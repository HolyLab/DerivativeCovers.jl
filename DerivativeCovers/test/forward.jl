# A forward propagation keeps the signs of the intermediate gradients and discards those of the
# adjoints, and these tests hold it there. Equality against `forward_corner`, which assembles that
# cover from a tape, is the acceptance criterion: it pins the construction, not merely the
# validity of what comes out.

# Array heads exercise the recorders `PROBLEMS` reaches only in part: `dotnode` for a bilinear
# pair, `linearnode` for a constant-coefficient combination, and both through `sum` and `*`.
const FORWARD_ARRAY = [
    ("softmax head", x -> sum(softmax(x) .* [1.0, -2.0, 0.5, 1.5]), [0.4, -1.1, 0.7, 0.2]),
    ("layernorm head", x -> sum(abs2, layernorm(x)) + layernorm(x)[1], [0.4, -1.1, 0.7, 0.2]),
    ("dot x,Ax", x -> dot(x, AQ * x) / 2, XQ),
    ("dot x,x", x -> dot(x, x) * dot(x, [1.0, 2.0, 3.0, 4.0, 5.0]), XQ),
    ("matvec chain", x -> sum(abs2, AQ * x) + dot(x, x), XQ),
]

const FORWARD_FIXTURES = vcat(PROBLEMS, FORWARD_ARRAY)

# Two routes to the same number differ in the order they accumulate it, and the split weights
# differ between them by the square root of the unsigned path sum to the output, so equality is
# to roundoff rather than to the bit. Measured worst case over these fixtures is 4e-16.
const FORWARD_RTOL = 1.0e-12

# The propagation carries both of the split's scale references, so it takes the same split a tape
# does at every block and the two are comparable coordinate by coordinate. The threshold and the
# declared scales are swept alongside, since the equality must hold for any of them.
const FORWARD_SPLITS = (
    (;), (; ratio = 0.5), (; ratio = 0.0), (; scales = nothing, ratio = 1.0e-4),
)

@testset "forward.jl" begin

    @testset "the cover is the tape's forward cover — $name" for
        (name, f, x0) in FORWARD_FIXTURES
        for kw in FORWARD_SPLITS
            fc = forward_cover(f, x0; kw...)
            @test fc.h_quad ≈ forward_corner(f, x0; kw...) rtol = FORWARD_RTOL
            @test fc.grad ≈ ForwardDiff.gradient(f, x0) rtol = FORWARD_RTOL
        end
    end

    @testset "the cover covers — $name" for (name, f, x0) in FORWARD_FIXTURES
        H = ForwardDiff.hessian(f, x0)
        for kw in FORWARD_SPLITS
            fc = forward_cover(f, x0; kw...)
            @test all(>=(0), fc.h_quad)
            @test max_coverage_ratio(fc.h_quad, H) <= GATE
        end
    end

    @testset "the declared scales propagate as a tape's sweep computes them" begin
        # `sigma` obeys the same recursion `sensitivity_scales` runs over a tape, so the value a
        # propagation reaches at its output is the value that sweep reaches at the output node.
        for (name, f, x0) in FORWARD_FIXTURES, weighted in (false, true)
            ws = weighted ? exp10.(range(-2, 2; length = length(x0))) : nothing
            captured = Ref{Any}(nothing)
            forward_cover(z -> (y = f(z); captured[] = y; y), x0; scales = ws)
            t, outidx = tape_function(f, x0)
            @test captured[].sigma ≈
                DerivativeCovers.sensitivity_scales(ws, t)[outidx] rtol = FORWARD_RTOL
        end
    end

    @testset "a slot is dead by its tangent, not by its declared scale" begin
        # `sigma` is a sum of products of first partials, so a chain of small factors underflows it
        # while the tangent it stands for is still there. Reading that as a dead slot would drop
        # the block's off-diagonal entry, which is in the Hessian and has to be covered.
        f = x -> (x[1] * 1.0e-100) * x[2]
        x0 = [1.0e100, 2.0]
        w = [1.0e-300, 1.0]
        captured = Ref{Any}(nothing)
        fc = forward_cover(
            z -> (u = z[1] * 1.0e-100; captured[] = u; u * z[2]), x0; scales = w
        )
        @test captured[].sigma == 0
        @test !DerivativeCovers.dead_slot(DerivativeCovers.forward_scale(captured[]))
        @test max_coverage_ratio(fc.h_quad, ForwardDiff.hessian(f, x0)) <= GATE
        @test fc.h_quad ≈ forward_corner(f, x0; scales = w) rtol = FORWARD_RTOL

        # The slots a tangent really does empty are dropped by both backends alike.
        deadprograms = (
            ("square times zero", x -> x[1]^2 * x[2], [0.0, 3.0]),
            ("unit cosine", x -> cos(x[1]) * x[2], [0.0, 3.0]),
            ("switched-off term", x -> (0.0 * x[1]^2) * (x[2] + x[2]^2)^2 + x[1] * x[2], [1.5, 2.0]),
            ("dead pair of a dot", x -> dot([x[1]^2, x[2]], [x[3], x[4]]), [0.0, 1.0, 2.0, 3.0]),
        )
        @testset "$name" for (name, fd, xd) in deadprograms
            fc = forward_cover(fd, xd)
            @test fc.h_quad ≈ forward_corner(fd, xd) rtol = FORWARD_RTOL
            @test max_coverage_ratio(fc.h_quad, ForwardDiff.hessian(fd, xd)) <= GATE
        end
    end

    @testset "every intermediate carries a cover" begin
        # `G` maintains its invariant at every variable, not only at the output, so the cover of a
        # sub-expression is available where the sub-expression is.
        inner(z) = tanh(z[1] * z[2]) + z[1]^2
        x0 = [0.6, -0.9]
        captured = Ref{Any}(nothing)
        outer(z) = (y = inner(z); captured[] = y; y * z[1])
        forward_cover(outer, x0)
        mid = forward_cover(captured[])
        @test mid.grad ≈ ForwardDiff.gradient(inner, x0) rtol = FORWARD_RTOL
        @test max_coverage_ratio(mid.h_quad, ForwardDiff.hessian(inner, x0)) <= GATE
    end

    @testset "the split keywords are checked as the tape's are" begin
        f = x -> x[1] * x[2]
        @test_throws DimensionMismatch forward_cover(f, [1.0, 2.0]; scales = [1.0])
        @test_throws "strictly positive" forward_cover(f, [1.0, 2.0]; scales = [1.0, 0.0])
        @test_throws "must lie in [0, 1)" forward_cover(f, [1.0, 2.0]; ratio = 1.0)
        @test forward_cover(f, [1.0, 2.0]; scales = [1.0, 4.0]) isa ForwardCover
    end

    @testset "the element type is preserved" begin
        f = x -> exp(x[1]) + x[1] * x[2]^2
        fc = forward_cover(f, Float32[0.5, 2.0])
        @test fc isa ForwardCover{Float32}
        @test eltype(fc.grad) === Float32
        @test eltype(fc.h_quad) === Float32
    end

    @testset "a deep chain needs no tape" begin
        # Working memory is per live variable, so a chain far longer than any of the fixtures
        # still runs; the dead intermediates are collectable as the propagation goes.
        function deep(x)
            v = x[1]
            for _ in 1:20_000
                v = tanh(v) + 0.5 * x[2]
            end
            return v
        end
        x0 = [0.3, 0.2]
        fc = forward_cover(deep, x0)
        @test all(isfinite, fc.h_quad)
        @test fc.grad ≈ ForwardDiff.gradient(deep, x0) rtol = FORWARD_RTOL
        @test max_coverage_ratio(fc.h_quad, ForwardDiff.hessian(deep, x0)) <= GATE
    end

    @testset "inputs share the context's zero cover channel — $name" for
        (name, f, x0) in FORWARD_FIXTURES
        # Propagation must leave the shared input channel unchanged.
        captured = Ref{Any}(nothing)
        forward_cover(z -> (captured[] = z; f(z)), x0)
        xs = captured[]
        ctx = xs[1].ctx
        @test all(x -> x.G === ctx.zerochannel, xs)
        @test iszero(ctx.zerochannel)
    end

    @testset "malformed programs are rejected" begin
        @test_throws "does not depend on the inputs" forward_cover(x -> zero(eltype(x)), [1.0])
        @test_throws "covers a scalar-valued program" forward_cover(x -> [x[1], x[2]], [1.0, 2.0])
        @test_throws "no inputs" forward_cover(x -> x, Float64[])
        # A constant carries no channels, so asking one for its cover names the reason.
        @test_throws "carries no tangent or cover channel" forward_cover(
            FVar{Float64, Vector{Float64}}(1.0)
        )
        # A declared sub-map belongs to recording; the propagation says so rather than
        # presenting a bare MethodError from an elemental it has no method for.
        @test_throws "available while recording" forward_cover(
            x -> submap(z -> [z[1] * z[2]], x)[1], [0.6, -0.9]
        )
    end

    @testset "a point that is not twice-differentiable is named by its values" begin
        # Without a tape there is no node number to report, so the elemental is identified by its
        # own value and those of its operands.
        divdiff = x -> (exp(x[1]) - exp(x[2])) / (x[1] - x[2])
        @test_throws "not twice-differentiable" forward_cover(divdiff, [0.5, 0.5])
        @test_throws "this elemental" forward_cover(divdiff, [0.5, 0.5])
        @test_throws "operand values" forward_cover(divdiff, [0.5, 0.5])

        # `check=false` propagates IEEE values instead, leaving the caller to test the cover.
        @test !all(isfinite, forward_cover(divdiff, [0.5, 0.5]; check = false).h_quad)
        @test all(isfinite, forward_cover(divdiff, [0.5, 1.5]).h_quad)
    end

    @testset "constants fold as they do on a tape" begin
        # The elemental table is shared, so the identities it declines to record — a zero shift, a
        # unit scaling — are declined here too, and a constant-only subexpression stays constant.
        f = x -> (c = zero(eltype(x)); (x[1] + c) * one(eltype(x)) * (x[2] + 2 * one(eltype(x))))
        g = x -> x[1] * (x[2] + 2)
        x0 = [0.7, -1.3]
        @test forward_cover(f, x0).h_quad == forward_cover(g, x0).h_quad
        @test forward_cover(f, x0).grad == forward_cover(g, x0).grad
    end

    # An integer starting point propagates in Float64, as recording one does. The one-based
    # declaration is pinned alongside every other entry point's, in `genericaxes.jl`.
    @testset "an integer starting point floats" begin
        @test forward_cover(x -> x[1] * x[2], [1, 2]) isa ForwardCover{Float64}
    end

    @testset "a static starting point propagates in static channels" begin
        # The channel storage is whatever the point is made of. The two storages run the same
        # recursions, so they agree to roundoff and differ only in what they allocate.
        @testset "$name" for (name, f, x0) in FORWARD_FIXTURES
            fs = forward_cover(f, SVector{length(x0)}(x0))
            fd = forward_cover(f, x0)
            @test fs.grad ≈ fd.grad rtol = FORWARD_RTOL
            @test fs.h_quad ≈ fd.h_quad rtol = FORWARD_RTOL
            @test fs.h_quad isa SVector{length(x0), Float64}
        end

        f = x -> exp(x[1]) + x[1] * x[2]^2
        @test forward_cover(f, SVector(0.5, 2.0)).h_quad ≈
            forward_cover(f, [0.5, 2.0]).h_quad rtol = FORWARD_RTOL
        @test forward_cover(f, SVector(0.5f0, 2.0f0)) isa ForwardCover{Float32}
        @test forward_cover(x -> x[1] * x[2], SVector(1, 2)) isa ForwardCover{Float64}

        # Constants carry no channels and so need none of the storage, whichever it is.
        g = x -> (c = zero(eltype(x)); (x[1] + c) * one(eltype(x)) * (x[2] + 2))
        @test forward_cover(g, SVector(0.7, -1.3)).h_quad ≈
            forward_cover(g, [0.7, -1.3]).h_quad rtol = FORWARD_RTOL

        # Static channels live in the variables themselves, so a propagation allocates a fixed
        # amount however long the program is. Two chains differing only in length pin that.
        function chain(L)
            return function (x)
                v = x[1]
                for _ in 1:L
                    v = tanh(v) + 0.5 * x[2]
                end
                return v
            end
        end
        alloc(f, x0) = (forward_cover(f, x0); @allocated forward_cover(f, x0))
        x0 = SVector(0.3, 0.2)
        @test alloc(chain(100), x0) == alloc(chain(400), x0)
    end
end
