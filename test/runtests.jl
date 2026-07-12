using CoverAD
using ForwardDiff
using LinearAlgebra
using Test

include("programs.jl")

# The covering chain abs.(H) <= B <= sum_k s_k s_k' <= h*h' is exact in exact arithmetic;
# roundoff can push a measured ratio a hair above 1.
const GATE = 1 + 1e-10

@testset "CoverAD.jl" begin

    @testset "tape mechanics" begin
        f = x -> x[1] * x[2] + exp(x[3])
        x0 = [2.0, -3.0, 0.5]
        t, outidx = tape_function(f, x0)
        @test t.ninputs == 3
        @test t.vals[outidx] ≈ f(x0)
        @test outidx == nnodes(t)
        # Inputs come first and carry no parents; every parent precedes its child.
        @test all(isempty, t.parents[1:t.ninputs])
        @test all(k -> all(<(k), t.parents[k]), 1:nnodes(t))

        @test_throws "program output is not a TVar" tape_function(x -> 1.0, x0)
        @test_throws "different tapes" tape_function(x0) do xs
            t2, o2 = tape_function(f, x0)
            xs[1] * TVar(t2, o2)
        end
        # Elementals outside the supported set fail fast rather than silently linearizing.
        @test_throws MethodError tape_function(x -> abs(x[1]), x0)
    end

    @testset "gradient matches ForwardDiff — $name" for (name, f, x0) in PROBLEMS
        res = cover_hessian(f, x0)
        @test res.grad ≈ ForwardDiff.gradient(f, x0)
    end

    @testset "covering chain — $name" for (name, f, x0) in PROBLEMS
        H = ForwardDiff.hessian(f, x0)
        res = cover_hessian(f, x0)
        @test bound_ratio(H, res.B) <= GATE                  # abs.(H) <= B
        for h in (res.h_sum, res.h_quad)
            @test all(>=(0), h)
            @test rank1_ratio(res.B, h) <= GATE              # B <= h*h'
            @test cover_ratio(H, h) <= GATE                  # abs.(H) <= h*h'
        end
        # The quadrature merge is never looser than the sum merge, coordinatewise.
        @test all(res.h_quad .<= res.h_sum .* GATE)
        # The absolute-value reverse sweep reproduces the sum merge.
        @test res.h_sum_rev ≈ res.h_sum rtol=1e-12
        # For H ⪰ 0 the minimal cover is sqrt.(diag(H)); no valid cover can go below it.
        if isposdef(H)
            @test all(res.h_quad .>= sqrt.(diag(H)) .* (1 - 1e-10))
        end
    end

    @testset "exact cover of the factored rank-1 quadratic" begin
        # f = (w'x)²/2 has H = w*w', whose minimal cover is abs.(w). A single nonlinear node
        # carries the whole Hessian, so both merges attain it.
        res = cover_hessian(quad_factored(WQ), XQ)
        @test res.h_sum ≈ abs.(WQ)
        @test res.h_quad ≈ abs.(WQ)
        H = ForwardDiff.hessian(quad_factored(WQ), XQ)
        @test cover_ratio(H, res.h_sum) ≈ 1
    end

    @testset "elementwise bound on the cancellation pair" begin
        x0 = CANCEL_X
        n = length(x0)

        # Entrywise form: each Hessian entry is produced by exactly one tape node, so the
        # absolute-value bound has nothing to cancel and attains abs.(H).
        He = ForwardDiff.hessian(cancellation, x0)
        Be = cover_hessian(cancellation, x0).B
        @test Be ≈ abs.(He)

        # Summed form: the same Hessian I - e*e'/n, but its diagonal now arises by cancellation
        # between sum(x²)/2 and (sum x)²/(2n). The bound sums magnitudes instead, giving
        # B = I + e*e'/n and a diagonal excess of (n+1)/(n-1).
        Hs = ForwardDiff.hessian(cancellation_summed, x0)
        Bs = cover_hessian(cancellation_summed, x0).B
        @test Hs ≈ He
        @test Bs ≈ I + ones(n, n) ./ n
        @test maximum(Bs[i, i] / abs(Hs[i, i]) for i in 1:n) ≈ (n + 1) / (n - 1)
    end

    @testset "covariance" begin
        cases = [("rosenbrock n=2", rosenbrock, [-1.2, 1.0], [1e-6, 1e3]),
                 ("quad matvec", quad_matvec(AQ), XQ, [1e-6, 1e3, 1.0, 1e-2, 1e4]),
                 ("logsumexp n=8", logsumexp,
                  [-3.0, -1.0, -2.0, 0.0, 1.0, 2.0, -0.5, 3.0],
                  [1e-3, 1e2, 1.0, 1e-1, 1e4, 1e-2, 1e1, 1e3])]
        @testset "$name" for (name, f, x0, d) in cases
            res = cover_hessian(f, x0)

            # x -> D*x leaves every intermediate value and adjoint unchanged and puts a factor
            # 1/d[a] on each input edge, so h -> h ./ d exactly.
            resd = cover_hessian(xt -> f(xt ./ d), d .* x0)
            @test resd.h_sum ≈ res.h_sum ./ d rtol=1e-12
            @test resd.h_quad ≈ res.h_quad ./ d rtol=1e-12

            # f -> c*f scales every adjoint by c, hence every rho and every h by sqrt(c).
            c = 7.3
            resc = cover_hessian(x -> c * f(x), x0)
            @test resc.h_sum ≈ sqrt(c) .* res.h_sum rtol=1e-12
            @test resc.h_quad ≈ sqrt(c) .* res.h_quad rtol=1e-12
        end
    end

    @testset "balanced_split" begin
        for (c11, c12, c22, z1, z2) in ((0.0, 1.0, 0.0, 2.0, 5.0),      # multiplication node
                                        (1.0, 4.0, 2.0, 0.1, 10.0),
                                        (0.0, 3.0, 0.0, 0.0, 1.0),      # zero-value fallback
                                        (2.0, 5.0, 1.0, 1.0, 0.0))
            rho1, rho2 = balanced_split(c11, c12, c22, z1, z2)
            @test rho1 >= sqrt(c11) * (1 - 1e-12)
            @test rho2 >= sqrt(c22) * (1 - 1e-12)
            @test rho1 * rho2 >= c12 * (1 - 1e-12)
        end
        # With both parent values positive the split is balanced against them: for a pure
        # multiplication node (c11 = c22 = 0), rho1/rho2 = z2/z1 and rho1*rho2 = c12.
        rho1, rho2 = balanced_split(0.0, 1.0, 0.0, 2.0, 8.0)
        @test rho1 / rho2 ≈ 8.0 / 2.0
        @test rho1 * rho2 ≈ 1.0
    end

    @testset "metrics" begin
        H = [4.0 -2.0; -2.0 1.0]
        @test cover_ratio(H, [2.0, 1.0]) ≈ 1
        @test cover_ratio(H, [4.0, 2.0]) ≈ 0.25
        @test cover_ratio(H, [0.0, 1.0]) == Inf
        @test cover_ratio(zeros(2, 2), [0.0, 0.0]) == 0     # nothing to cover

        B = [4.0 2.0; 2.0 2.0]
        @test bound_ratio(H, B) ≈ 1                         # attained on the (1,1) entry
        @test rank1_ratio(B, [2.0, 1.0]) ≈ 2                # B[2,2] = 2 exceeds h[2]^2 = 1

        mx, gm, nextra = tightness([2.0, 8.0, 1.0], [1.0, 2.0, 0.0])
        @test mx ≈ 4
        @test gm ≈ sqrt(2 * 4)
        @test nextra == 1
        @test all(isnan, tightness([1.0], [0.0])[1:2])

        @test_throws DimensionMismatch cover_ratio(H, [1.0])
        @test_throws DimensionMismatch bound_ratio(H, zeros(3, 3))
        @test_throws DimensionMismatch tightness([1.0], [1.0, 2.0])
    end

end
