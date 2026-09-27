# Recording over a box of intervals: covers hold at every point of the box, and the gradient
# encloses the gradient there.

using IntervalArithmetic: IntervalArithmetic, Interval, interval, in_interval, mid

@testset "interval element type" begin

    @testset "magtype" begin
        @test DerivativeCovers.magtype(Float32) === Float32
        @test DerivativeCovers.magtype(Complex{Float64}) === Float64
        @test DerivativeCovers.magtype(Interval{Float64}) === Float64
        @test DerivativeCovers.magnitude(interval(-3.0, 2.0)) === 3.0
        @test DerivativeCovers.separation(interval(0.0, 1.0), interval(0.5, 2.0)) == 0
        @test DerivativeCovers.separation(interval(0.0, 1.0), interval(1.5, 2.0)) == 0.5
    end

    inner = v -> [v[1] * v[2], exp(v[1])]
    # `submap` records onto a tape, so ForwardDiff differentiates the program with `inner` inlined.
    reference = Dict("declared sub-map" => x -> (y = inner(x[1:2]); y[1] + tanh(y[2]) * x[3]))
    box_programs = (
        (
            "exp-sin", x -> x[1]^2 * exp(x[2]) + sin(x[1] * x[2]),
            [0.4, 0.9], [0.6, 1.1],
        ),
        (
            "rosenbrock", x -> (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2,
            [-0.3, 0.8], [0.2, 1.2],
        ),
        (
            "dot and linear", x -> dot(x[1:2], x[3:4]) + tanh(dot([1.5, -0.5, 2.0, 1.0], x)),
            [0.1, -0.4, 0.3, 0.2], [0.3, -0.2, 0.6, 0.5],
        ),
        (
            "declared sub-map", x -> (y = submap(inner, [x[1], x[2]]); y[1] + tanh(y[2]) * x[3]),
            [0.7, -1.2, 0.5], [0.9, -1.0, 0.7],
        ),
    )

    # Roundoff slack on the covering inequality, as in the Float64 tests.
    slack = 1 + sqrt(eps())
    covers(H, h) = all(abs.(H) .<= h * h' .* slack)

    @testset "covers hold over the box — $name" for (name, f, lo, hi) in box_programs
        X = interval.(lo, hi)
        c = cover_hessian(f, X)
        d = cover_hessian(WithDiagnostics(), f, X)
        lat = cover_lattice(f, X)
        @test c isa HessianCover{Interval{Float64}, Float64}
        @test c.h_sum isa Vector{Float64}
        @test c.grad isa Vector{Interval{Float64}}
        @test d isa HessianDiagnostics{Interval{Float64}}
        @test d.B isa Matrix{Float64}
        @test lat isa CoverLattice{Interval{Float64}, Float64}
        @test c.h_sum ≈ d.h_sum rtol = 1.0e-12

        fref = get(reference, name, f)
        rng = Xoshiro(17)
        ok_cover = ok_grad = ok_diag = ok_lattice = true
        for _ in 1:200
            x = lo .+ rand(rng, length(lo)) .* (hi .- lo)
            H = ForwardDiff.hessian(fref, x)
            g = ForwardDiff.gradient(fref, x)
            ok_grad &= all(in_interval.(g, c.grad)) && all(in_interval.(g, d.grad)) &&
                all(in_interval.(g, lat.grad))
            ok_cover &= covers(H, c.h_sum)
            ok_diag &= covers(H, d.h_sum) && covers(H, d.h_quad) &&
                all(abs.(H) .<= d.B .* slack)
            ok_lattice &= all(abs.(H) .<= lat.B .* slack) &&
                all(abs.(H) .<= lat.B_signed .* slack) &&
                all(
                covers(H, getfield(lat, fld)) for fld in (
                        :h_sum, :h_quad, :h_sum_absadj, :h_quad_absadj,
                        :h_sum_signed, :h_quad_signed, :h_sum_forward, :h_quad_forward,
                    )
            )
        end
        @test ok_grad
        @test ok_cover
        @test ok_diag
        @test ok_lattice
    end

    samples(rng, lo, hi, k = 150) = [lo .+ rand(rng, length(lo)) .* (hi .- lo) for _ in 1:k]

    @testset "variance, hybrid and forward covers hold over the box — $name" for
        (name, f, lo, hi) in box_programs
        X = interval.(lo, hi)
        vc = cover_variance(f, X)
        @test vc isa VarianceCover{Interval{Float64}, Float64}
        hy = cover_hybrid(f, X, 2)
        hn = cover_hybrid(f, X, cyclic_partition(tape_function(f, X)[1], 3))
        @test hy isa Vector{Float64}
        # A propagation has no tape to sweep a declaration from.
        fc = name == "declared sub-map" ? nothing : forward_cover(f, X)
        fc === nothing || @test fc isa ForwardCover{Interval{Float64}, Float64}

        fref = get(reference, name, f)
        ok_grad = ok_cover = true
        for x in samples(Xoshiro(23), lo, hi)
            H = ForwardDiff.hessian(fref, x)
            g = ForwardDiff.gradient(fref, x)
            ok_grad &= all(in_interval.(g, vc.grad))
            ok_cover &= covers(H, vc.h_sum) && covers(H, hy) && covers(H, hn)
            if fc !== nothing
                ok_grad &= all(in_interval.(g, fc.grad))
                ok_cover &= covers(H, fc.h_quad)
            end
        end
        @test ok_grad
        @test ok_cover
    end

    vector_programs = (
        (
            "mixed", x -> [x[1] * x[2], exp(x[1]) - x[3], x[2] + 2x[3]],
            [0.4, 1.9, -1.1], [0.6, 2.1, -0.9],
        ),
        (
            "residuals", x -> [x[1] * x[2] - 1, exp(x[1]) - 2, sin(x[2]) * x[3]],
            [0.3, 0.8, -0.5], [0.5, 1.0, -0.3],
        ),
    )

    @testset "Jacobian covers hold over the box — $name" for (name, r, lo, hi) in vector_programs
        X = interval.(lo, hi)
        options = ((;), (; start = :equal), (; rounds = 0), (; a = [1.0, 2.0, 0.5]))
        cs = [cover_jacobian(r, X; kw...) for kw in options]
        ds = [cover_jacobian(WithDiagnostics(), r, X; kw...) for kw in options]
        @test all(c -> c isa JacobianCover{Float64}, cs)
        @test all(d -> d isa JacobianDiagnostics{Float64}, ds)
        hyb = [cover_jacobian_hybrid(r, X, G) for G in 1:3]
        @test all(p -> p.b isa Vector{Float64}, hyb)

        dominates(A, a, b) = all(A .<= a .* b' .* slack)
        ok = true
        for x in samples(Xoshiro(29), lo, hi)
            A = abs.(ForwardDiff.jacobian(r, x))
            ok &= all(c -> dominates(A, c.a, c.b_sum), cs)
            ok &= all(ds) do d
                dominates(A, d.a, d.b_sum) && dominates(A, d.a, d.b_quad) &&
                    dominates(A, d.a, d.b_max) && all(A .<= d.Utilde .* slack)
            end
            ok &= all(p -> dominates(A, p.a, p.b), hyb)
        end
        @test ok
    end

    @testset "least-squares covers hold over the box — $name" for (name, r, lo, hi) in vector_programs
        X = interval.(lo, hi)
        nc = cover_nls(r, X)
        nd = cover_nls(WithDiagnostics(), r, X)
        @test nc isa NLSCover{Interval{Float64}, Float64}
        @test nd isa NLSDiagnostics{Interval{Float64}, Float64}
        @test nd.Utilde isa Matrix{Float64}
        a, b = gn_ab(nd)
        @test a isa Vector{Float64}

        loss(x) = sum(abs2, r(x)) / 2
        ok_grad = ok_cover = true
        for x in samples(Xoshiro(31), lo, hi)
            J = ForwardDiff.jacobian(r, x)
            H = ForwardDiff.hessian(loss, x)
            g = ForwardDiff.gradient(loss, x)
            ok_grad &= all(in_interval.(g, nc.grad)) && all(in_interval.(g, nd.grad))
            ok_cover &= covers(H, nc.h) && covers(H, nd.h) && covers(J' * J, nc.h_gn) &&
                covers(J' * J, nd.h_gn) && all(abs.(J) .<= nd.Utilde .* slack) &&
                all(abs.(J) .<= a .* b' .* slack)
        end
        @test ok_grad
        @test ok_cover
    end

    # The partial magnitudes over the box bound those at each point, so the box's scale bounds
    # every pointwise scale.
    @testset "the Gram variance scale over the box bounds the pointwise scale — $name" for
        (name, r, lo, hi) in vector_programs
        X = interval.(lo, hi)
        t, outidxs = tape_function(r, X)
        w = [1.0, -2.0, 0.5]
        hv = DerivativeCovers.gramvariance(t, outidxs, w)
        @test hv isa Vector{Float64}
        @test all(hv .<= gramcover(t, outidxs, w) .* slack)
        nc = cover_nls(r, X)
        @test nc.h_var_gn isa Vector{Float64}
        @test all(nc.h_var_gn .<= nc.h_gn .* slack)
        ok = true
        for x in samples(Xoshiro(43), lo, hi)
            tx, ox = tape_function(r, x)
            ok &= all(DerivativeCovers.gramvariance(tx, ox, w) .<= hv .* slack)
        end
        @test ok
    end

    @testset "products on an interval tape enclose the pointwise products" begin
        _, r, lo, hi = vector_programs[2]
        t, outidxs = tape_function(r, interval.(lo, hi))
        v, u = [1.0, -0.5, 2.0], [0.3, 1.0, -1.0]
        Jv, Jtu = jvp(t, outidxs, v), vjp(t, outidxs, u)
        @test Jv isa Vector{Interval{Float64}}
        @test Jtu isa Vector{Interval{Float64}}
        _, f, flo, fhi = box_programs[1]
        tf, o = tape_function(f, interval.(flo, fhi))
        w = [1.0, -2.0]
        Hw = hvp(tf, o, w)
        @test Hw isa Vector{Interval{Float64}}
        ok = true
        for (x, y) in zip(samples(Xoshiro(37), lo, hi), samples(Xoshiro(41), flo, fhi))
            J = ForwardDiff.jacobian(r, x)
            ok &= all(in_interval.(J * v, Jv)) && all(in_interval.(J' * u, Jtu)) &&
                all(in_interval.(ForwardDiff.hessian(f, y) * w, Hw))
        end
        @test ok
    end

    @testset "a declaration must hold over the same sets as the tape" begin
        X = interval.([0.7, -1.2], [0.9, -1.0])
        d = declare_submap(inner, X)
        @test d.y isa Vector{Interval{Float64}}
        @test d.J isa Matrix{Interval{Float64}}
        @test d.S isa Matrix{Float64}
        t = Tape(X)
        xs = [TVar(t, i) for i in 1:2]
        @test length(submapnode(xs, d.y, d.J, d.S)) == 2
        # A declaration made at the midpoint says nothing about the rest of the box.
        p = declare_submap(inner, mid.(X))
        @test_throws "needs a declaration evaluated on the same sets" submapnode(xs, p.y, p.J, p.S)
        @test_throws "needs a declaration evaluated on the same sets" submapnode(xs, d.y, p.J, p.S)
    end

    @testset "pointwise estimates are refused over a box" begin
        _, r, lo, hi = vector_programs[1]
        t, outidxs = tape_function(r, interval.(lo, hi))
        @test_throws "power iteration is pointwise" scaled_singular_value(t, outidxs, ones(3), ones(3))
        _, f, flo, fhi = box_programs[1]
        tf, o = tape_function(f, interval.(flo, fhi))
        @test_throws "power iteration is pointwise" scaled_spectral_radius(tf, o, ones(2))
        @test_throws "must be a plain number" jacobian_shrink(ones(3), ones(3), interval(0.5, 0.6))
        @test_throws "must be a plain number" spectral_shrink(ones(2), interval(0.5, 0.6))
    end

    @testset "the cover norm of an interval gradient is its largest value" begin
        g = [interval(-2.0, 1.0), interval(0.5, 3.0)]
        @test cover_norm(g, [2.0, 4.0]) === 1.25
        @test cover_norm([interval(0.0), interval(1.0)], [0.0, 1.0]) === 1.0
        @test cover_norm([interval(-1.0, 0.0), interval(1.0)], [0.0, 1.0]) === Inf
    end

    @testset "a thin box reproduces the point cover — $name" for (name, f, lo, _) in box_programs
        c64 = cover_hessian(f, lo)
        c = cover_hessian(f, interval.(lo))
        @test c.h_sum ≈ c64.h_sum rtol = 1.0e-12
        @test mid.(c.grad) ≈ c64.grad rtol = 1.0e-12
        @test all(in_interval.(c64.grad, c.grad))
    end

    @testset "control flow the box cannot decide" begin
        X = interval.([-1.0, 1.0], [2.0, 2.0])
        @test_throws IntervalArithmetic.InconclusiveBooleanOperation cover_hessian(x -> x[1]^x[2], X)
        @test_throws "leaves the domain of an elemental" cover_hessian(x -> log(x[1]) + x[2], X)
        @test_throws "leaves the domain of an elemental" tape_function(x -> 1 / x[1] + x[2], X)
        # Defined on the whole box, but not differentiable at its lower edge.
        @test_throws "not differentiable everywhere on the box" cover_hessian(
            x -> sqrt(x[1]) * x[2], interval.([0.0, 1.0], [2.0, 2.0])
        )
    end
end
