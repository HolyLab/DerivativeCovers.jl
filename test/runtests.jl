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

    @testset "node levels and partitions" begin
        f = x -> exp(x[1] * x[2]) + x[2]
        t, outidx = tape_function(f, [1.0, 2.0])
        levels = node_levels(t)
        @test all(iszero, levels[1:t.ninputs])
        @test all(k -> levels[k] > maximum(levels[t.parents[k]]; init=-1), (t.ninputs + 1):nnodes(t))

        # A tape shallower than `ngroups` gives every depth its own group.
        lmax = maximum(levels)
        groups = level_partition(t, 10 * lmax)
        for k in (t.ninputs + 1):nnodes(t), l in (t.ninputs + 1):nnodes(t)
            @test (groups[k] == groups[l]) == (levels[k] == levels[l])
        end
        @test level_partition(t, 1) == ones(Int, nnodes(t))
        @test_throws ArgumentError level_partition(t, 0)

        N = nnodes(t)
        @test cyclic_partition(t, 1) == ones(Int, N)
        @test cyclic_partition(t, N) == 1:N
        @test sort(cyclic_partition(t, 3)) == sort(repeat(1:3, outer=N)[1:N])
        @test_throws ArgumentError cyclic_partition(t, 0)
    end

    @testset "hybrid merge — $name" for (name, f, x0) in PROBLEMS
        H = ForwardDiff.hessian(f, x0)
        t, outidx = tape_function(f, x0)
        N = nnodes(t)
        res = cover_hessian(t, outidx)

        # One group sums every node: the sum merge. One node per group: the quadrature merge.
        @test cover_hybrid(t, outidx, 1) ≈ res.h_sum rtol=1e-12
        @test cover_hybrid(t, outidx, N) ≈ res.h_quad rtol=1e-12
        @test cover_hybrid(t, outidx, collect(1:N)) ≈ res.h_quad rtol=1e-12

        rng = MersenneTwister(hash(name))
        for groups in (cyclic_partition(t, 4), level_partition(t, 4),
                       rand(rng, 1:5, N), rand(rng, 1:2, N))
            h = cover_hybrid(t, outidx, groups)
            @test all(>=(0), h)
            @test cover_ratio(H, h) <= GATE                  # valid for any partition
            @test rank1_ratio(res.B, h) <= GATE
            # Bracketed by the two extremes it interpolates.
            @test all(res.h_quad .<= h .* GATE)
            @test all(h .<= res.h_sum .* GATE)

            # Coarsening a partition (merging groups pairwise) can only loosen the cover.
            coarse = (groups .+ 1) .÷ 2
            @test all(h .<= cover_hybrid(t, outidx, coarse) .* GATE)
        end
    end

    @testset "hybrid merge covariance" begin
        # Rescaling is applied inside the program, so it adds nodes to the tape and shifts the
        # depth levels the default partition bins. Comparing two tapes of the *same* shape — the
        # unscaled one written with the same wrapper — isolates the covariance from that shift.
        cases = [("rosenbrock n=2", rosenbrock, [-1.2, 1.0], [1e-6, 1e3]),
                 ("logsumexp n=8", logsumexp,
                  [-3.0, -1.0, -2.0, 0.0, 1.0, 2.0, -0.5, 3.0],
                  [1e-3, 1e2, 1.0, 1e-1, 1e4, 1e-2, 1e1, 1e3])]
        @testset "$name" for (name, f, x0, d) in cases
            e = ones(length(x0))
            for G in (2, 4)
                href = cover_hybrid(xt -> f(xt ./ e), x0, G)
                hd = cover_hybrid(xt -> f(xt ./ d), d .* x0, G)
                @test hd ≈ href ./ d rtol=1e-12

                c = 7.3
                hc = cover_hybrid(x -> c * f(x), x0, G)
                h1 = cover_hybrid(x -> 1.0 * f(x), x0, G)
                @test hc ≈ sqrt(c) .* h1 rtol=1e-12
            end
        end
    end

    @testset "hvp — $name" for (name, f, x0) in PROBLEMS
        H = ForwardDiff.hessian(f, x0)
        v = Float64[cos(3i) for i in eachindex(x0)]
        @test hvp(f, x0, v) ≈ H * v rtol=1e-10
        @test hvp(f, x0, zero(x0)) ≈ zero(x0) atol=1e-14
        @test_throws DimensionMismatch hvp(f, x0, [1.0; v])
    end

    @testset "spectral shrink — $name" for (name, f, x0) in PROBLEMS
        H = ForwardDiff.hessian(f, x0)
        t, outidx = tape_function(f, x0)
        h = cover_hessian(t, outidx).h_sum

        # Every entry of the scaled Hessian is bounded by its spectral norm.
        lambda = scaled_spectral_radius(H, h)
        @test lambda >= cover_ratio(H, h) * (1 - 1e-12)

        hs = spectral_shrink(h, lambda)
        @test cover_ratio(H, hs) <= GATE                     # shrinking preserves validity
        @test all(hs .<= h .* GATE)                          # and never inflates
        if lambda < 1
            @test hs ≈ h .* sqrt(lambda)
            @test all(hs[h .> 0] .< h[h .> 0])
        else
            @test hs == h
        end

        # The matrix-free estimate approaches the exact norm from below, where it applies at all:
        # a coordinate the Hessian ignores gets a zero cover, which the power iteration rejects.
        if all(>(0), h)
            est = scaled_spectral_radius(t, outidx, h; maxiter=1000, tol=1e-10)
            @test est <= lambda * GATE
            @test est ≈ lambda rtol=1e-4
        else
            @test_throws "strictly positive" scaled_spectral_radius(t, outidx, h)
        end
    end

    @testset "spectral shrink algebra" begin
        H = [4.0 -2.0; -2.0 1.0]
        h = [2.0, 1.0]
        # Hhat = [1 -1; -1 1] has spectral norm 2, so this cover is spectrally saturated.
        @test scaled_spectral_radius(H, h) ≈ 2
        @test spectral_shrink(h, 2.0) == h
        @test spectral_shrink(h, 0.25) ≈ h ./ 2
        @test spectral_shrink(h, 0.25; safety=4) == h        # safety = 4 saturates the cover

        # The scaled Hessian, and hence the shrink factor, is invariant under both rescalings.
        d = [1e-3, 1e5]
        @test scaled_spectral_radius(H .* (d * d'), h .* d) ≈ scaled_spectral_radius(H, h)
        @test scaled_spectral_radius(9 .* H, 3 .* h) ≈ scaled_spectral_radius(H, h)

        # A coordinate the Hessian ignores may carry a zero cover; one it does not, may not.
        @test scaled_spectral_radius([1.0 0.0; 0.0 0.0], [1.0, 0.0]) ≈ 1
        @test scaled_spectral_radius(zeros(2, 2), [0.0, 0.0]) == 0
        @test_throws "does not cover" scaled_spectral_radius(H, [2.0, 0.0])

        t, outidx = tape_function(rosenbrock, [-1.2, 1.0])
        @test_throws "strictly positive" scaled_spectral_radius(t, outidx, [1.0, 0.0])
        @test_throws "did not converge" scaled_spectral_radius(t, outidx, [1.0, 1.0];
                                                               maxiter=1, tol=1e-12)
        @test_throws ArgumentError spectral_shrink(h, -1.0)
        @test_throws ArgumentError spectral_shrink(h, 1.0; safety=0)
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
