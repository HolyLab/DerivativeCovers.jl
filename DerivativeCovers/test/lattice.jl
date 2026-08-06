# Discarding the signs of a channel can only inflate a cover, never invalidate one. These tests
# pin both halves of that: every corner of the lattice covers the Hessian, and the corners are
# ordered by how many channels kept their signs.

# A program in which each channel has something to cancel: `d` loses gradient magnitude along the
# two paths from `x[1]`, and `u` reaches the output through the two arms of `exp(u) - u` with
# opposite adjoint signs.
lattice_program(x) = (d = sin(x[1]) - x[1]; u = d * x[2]; exp(u) - u)
const LATTICE_X = [0.7, 1.3]

@testset "lattice.jl" begin

    @testset "every corner is a cover — $name" for (name, f, x0) in PROBLEMS
        H = ForwardDiff.hessian(f, x0)
        lat = cover_lattice(f, x0)

        @test lat.grad ≈ ForwardDiff.gradient(f, x0) rtol = 1.0e-12
        @test bound_ratio(H, lat.B) <= GATE
        @test bound_ratio(H, lat.B_signed) <= GATE

        corners = (
            lat.h_sum, lat.h_quad,
            lat.h_sum_absadj, lat.h_quad_absadj,
            lat.h_sum_signed, lat.h_quad_signed,
            lat.h_sum_forward, lat.h_quad_forward,
        )
        for h in corners
            @test all(>=(0), h)
            @test cover_ratio(H, h) <= GATE
        end
        # A corner's own elementwise bound is what its atoms were built to dominate.
        @test rank1_ratio(lat.B, lat.h_sum) <= GATE
        @test rank1_ratio(lat.B_signed, lat.h_sum_signed) <= GATE
    end

    @testset "signs only tighten — $name" for (name, f, x0) in PROBLEMS
        lat = cover_lattice(f, x0)

        # Quadrature is never looser than the sum, in every corner.
        @test all(lat.h_quad .<= lat.h_sum .* GATE)
        @test all(lat.h_quad_absadj .<= lat.h_sum_absadj .* GATE)
        @test all(lat.h_quad_signed .<= lat.h_sum_signed .* GATE)
        @test all(lat.h_quad_forward .<= lat.h_sum_forward .* GATE)

        # Signed tangents tighten: abs.(∇v) <= utilde, at a fixed adjoint channel.
        @test all(lat.B_signed .<= lat.B .* GATE)
        @test all(lat.h_sum_signed .<= lat.h_sum .* GATE)
        @test all(lat.h_quad_signed .<= lat.h_quad .* GATE)
        @test all(lat.h_sum_forward .<= lat.h_sum_absadj .* GATE)
        @test all(lat.h_quad_forward .<= lat.h_quad_absadj .* GATE)

        # Signed adjoints tighten: abs.(vbar) <= acheck, at a fixed tangent channel.
        @test all(lat.h_sum .<= lat.h_sum_absadj .* GATE)
        @test all(lat.h_quad .<= lat.h_quad_absadj .* GATE)
        @test all(lat.h_sum_signed .<= lat.h_sum_forward .* GATE)
        @test all(lat.h_quad_signed .<= lat.h_quad_forward .* GATE)
    end

    @testset "the shipped corner is `cover_hessian`'s — $name" for (name, f, x0) in PROBLEMS
        lat = cover_lattice(f, x0)
        res = cover_hessian(WithDiagnostics(), f, x0)
        @test lat.B == res.B
        @test lat.h_sum == res.h_sum
        @test lat.h_quad == res.h_quad
        @test lat.grad ≈ res.grad rtol = 1.0e-12
    end

    # The signed channel must be the *same* computation the lattice performs, not merely a close
    # one: any drift between them means the two have stopped sharing an implementation.
    @testset "the signed channel is a lattice corner — $name" for (name, f, x0) in PROBLEMS
        lat = cover_lattice(f, x0)
        sgn = cover_hessian(WithDiagnostics(), f, x0; channel = SignedTangents())
        @test sgn.B == lat.B_signed
        @test sgn.h_sum == lat.h_sum_signed
        @test sgn.h_quad == lat.h_quad_signed
        # The channel is the tangent channel; the gradient still comes off the adjoint sweep.
        @test sgn.grad == cover_hessian(WithDiagnostics(), f, x0).grad
    end

    @testset "the signed channel covers and tightens — $name" for (name, f, x0) in PROBLEMS
        H = ForwardDiff.hessian(f, x0)
        abs_ch = cover_hessian(WithDiagnostics(), f, x0)
        sgn = cover_hessian(WithDiagnostics(), f, x0; channel = SignedTangents())
        for h in (sgn.h_sum, sgn.h_quad)
            @test all(>=(0), h)
            @test rank1_ratio(sgn.B, h) <= GATE
            @test cover_ratio(H, h) <= GATE
        end
        # Tighter at every coordinate than the default channel, the two adjoint channels agreeing.
        @test all(sgn.h_sum .<= abs_ch.h_sum .* GATE)
        @test all(sgn.h_quad .<= abs_ch.h_quad .* GATE)
        # `h_sum_rev` stays the absolute route, so here it dominates rather than agrees.
        @test all(sgn.h_sum .<= sgn.h_sum_rev .* GATE)
        @test sgn.h_sum_rev == abs_ch.h_sum_rev
    end

    @testset "the signed channel is not free" begin
        # The O(L) sweep collapses the sum merge into one reverse pass, which a signed channel
        # cannot do; refusing is what keeps the mode's cost promise honest.
        @test_throws "CoverOnly cannot merge over SignedTangents" cover_hessian(
            CoverOnly(), lattice_program, LATTICE_X; channel = SignedTangents()
        )
        @test_throws "O(n*L)" cover_hessian(
            CoverOnly(), lattice_program, LATTICE_X; channel = SignedTangents()
        )
        # The default channel is the default, bitwise.
        @test cover_hessian(CoverOnly(), lattice_program, LATTICE_X; channel = AbsoluteTangents()).h_sum ==
            cover_hessian(CoverOnly(), lattice_program, LATTICE_X).h_sum
    end

    @testset "the signed channel threads the split keywords" begin
        for kw in ((; ), (; ratio = 0.5), (; scales = fill(3.0, length(LATTICE_X))))
            lat = cover_lattice(lattice_program, LATTICE_X; kw...)
            sgn = cover_hessian(
                WithDiagnostics(), lattice_program, LATTICE_X; kw..., channel = SignedTangents()
            )
            @test sgn.h_quad == lat.h_quad_signed
        end
    end

    @testset "each channel is separately visible" begin
        lat = cover_lattice(lattice_program, LATTICE_X)

        # Both inequalities are strict here, so neither substitution is silently a no-op.
        @test any(lat.h_quad_signed .< lat.h_quad)
        @test any(lat.h_quad .< lat.h_quad_absadj)
        @test any(lat.B_signed .< lat.B)

        # The two off-diagonal corners are incomparable: discarding tangent signs and discarding
        # adjoint signs cost different coordinates.
        @test any(lat.h_quad_forward .< lat.h_quad)
        @test any(lat.h_quad .< lat.h_quad_forward)
    end

    # What the adjoint channel is worth must be a function of the mechanism, not of whichever
    # fixture happens to exhibit it: `adjcancel`'s knob sets how nearly the output is stationary
    # in each product node, and the separation has to follow it.
    @testset "the adjoint separation follows the knob" begin
        deltas = (1.0, 1.0e-1, 1.0e-2, 1.0e-3)
        gains = map(deltas) do delta
            p = Problems.adjoint_cancellation(2; delta)
            lat = cover_lattice(p.f, p.x0)
            # The tangent channel is inert here, so this isolates the adjoint one.
            @test lat.h_quad == lat.h_quad_signed
            @test lat.h_quad_absadj == lat.h_quad_forward
            maximum(lat.h_quad_absadj ./ lat.h_quad)
        end
        @test issorted(gains)
        # The separation scales as 2/delta, so a tenfold knob is worth about tenfold.
        @test all(gains[i + 1] / gains[i] > 5 for i in 1:(length(gains) - 1))

        # The mechanism is a property of each node, so a chain of them holds the separation at
        # every size. This is what admits the family into a size sweep.
        wide = map((2, 4, 16, 64)) do n
            p = Problems.adjoint_cancellation(n; delta = 1.0e-2)
            lat = cover_lattice(p.f, p.x0)
            maximum(lat.h_quad_absadj ./ lat.h_quad)
        end
        @test all(≈(wide[1]), wide)
    end

    @testset "tangents are the signed twin of sensitivities" begin
        t, outidx = tape_function(lattice_program, LATTICE_X)
        u = DerivativeCovers.tangents(t)
        utilde = DerivativeCovers.sensitivities(t)

        @test length(u) == nnodes(t)
        # Inputs seed the identity; the output node carries the gradient.
        for a in 1:t.ninputs
            @test u[a] == [i == a for i in 1:t.ninputs]
        end
        @test u[outidx] ≈ ForwardDiff.gradient(lattice_program, LATTICE_X) rtol = 1.0e-12
        # The absolute recursion bounds the signed one node by node, strictly where paths cancel.
        for k in eachindex(u)
            @test all(abs.(u[k]) .<= utilde[k] .* GATE)
        end
        @test any(k -> any(abs.(u[k]) .< utilde[k]), eachindex(u))
    end

    @testset "element type is preserved" begin
        lat = cover_lattice(lattice_program, Float32.(LATTICE_X))
        @test lat isa CoverLattice{Float32}
        for fld in fieldnames(CoverLattice)
            @test eltype(getfield(lat, fld)) === Float32
        end
    end

    @testset "the split keywords thread through" begin
        for kw in ((; ), (; ratio = 0.5), (; scales = fill(3.0, length(LATTICE_X))))
            lat = cover_lattice(lattice_program, LATTICE_X; kw...)
            res = cover_hessian(WithDiagnostics(), lattice_program, LATTICE_X; kw...)
            @test lat.h_quad == res.h_quad
            # The ordering is a property of the construction, not of the default split.
            @test all(lat.h_quad_signed .<= lat.h_quad .* GATE)
            @test all(lat.h_quad .<= lat.h_quad_absadj .* GATE)
        end
    end

    @testset "a declared sub-map keeps the lattice ordered" begin
        # Only first partials enter the tangent channel, so a node with no recorded curvature
        # propagates like any other and its declared cover scales with whichever adjoint it sees.
        inner(z) = [z[1] * z[2], exp(z[1])]
        outer(y) = y[1]^2 + sin(y[2])
        x0 = [0.6, -0.9]
        f = x -> outer(submap(inner, x))
        H = ForwardDiff.hessian(x -> outer(inner(x)), x0)
        lat = cover_lattice(f, x0)
        for h in (lat.h_quad, lat.h_quad_signed, lat.h_quad_absadj, lat.h_quad_forward)
            @test cover_ratio(H, h) <= GATE
        end
        @test all(lat.h_quad_signed .<= lat.h_quad .* GATE)
        @test all(lat.h_quad .<= lat.h_quad_absadj .* GATE)
    end
end
