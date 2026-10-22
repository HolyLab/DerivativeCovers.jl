# The aligned and cancelling programs share node values and derivative magnitudes. Only coefficient
# signs differ, isolating the effect of cancellation.
#
# Qualify the generator because programs.jl defines an unrelated fixture with the same name.

@testset "cancellation.jl" begin

    n, m, delta = 6, 4, 1.0e-2

    @testset "twin and Hessian agreement — $(aligned ? "aligned" : "cancel")" for aligned in (false, true)
        p = Problems.cancellation(n; m, delta, aligned, rng = MersenneTwister(11))
        @test p.f(p.x0) ≈ p.twin(p.x0)
        rng = MersenneTwister(12)
        for _ in 1:3
            x = randn(rng, n)
            @test p.f(x) ≈ p.twin(x)
        end
        H = exact_hessian(p)
        @test ForwardDiff.hessian(p.f, p.x0) ≈ H
        @test ForwardDiff.hessian(p.f, randn(MersenneTwister(13), n)) ≈ H
    end

    @testset "pair property" begin
        pc = Problems.cancellation(n; m, delta, aligned = false, rng = MersenneTwister(7))
        pa = Problems.cancellation(n; m, delta, aligned = true, rng = MersenneTwister(7))
        tc, oc = tape_function(pc.f, pc.x0)
        ta, oa = tape_function(pa.f, pa.x0)
        @test nnodes(tc) == nnodes(ta)

        rc = cover_hessian(WithDiagnostics(), tc, oc)
        ra = cover_hessian(WithDiagnostics(), ta, oa)
        # The gradient at x0 is identically delta*(q'x0)*q in both variants — every quartet's
        # linear form vanishes at x0 by construction, so it contributes nothing at first order —
        # but the two variants sum that cancellation in a different order (opposite-signed
        # terms), leaving roundoff at the last few bits.
        @test isapprox(rc.grad, ra.grad; rtol = 1.0e-12)
        @test rc.B == ra.B
        @test rc.h_sum == ra.h_sum
        @test rc.h_quad == ra.h_quad

        vc = cover_hessian(tc, oc)
        va = cover_hessian(ta, oa)
        @test vc.h_var == va.h_var
    end

    @testset "validity" for aligned in (false, true)
        p = Problems.cancellation(n; m, delta, aligned, rng = MersenneTwister(21))
        H = exact_hessian(p)
        t, outidx = tape_function(p.f, p.x0)
        res = cover_hessian(WithDiagnostics(), t, outidx)
        @test max_coverage_ratio(res.h_sum, H) <= GATE
        @test max_coverage_ratio(res.h_quad, H) <= GATE
        @test max_coverage_ratio(cover_hybrid(t, outidx, 4), H) <= GATE
    end

    @testset "closed forms" begin
        pc = Problems.cancellation(n; m, delta, aligned = false, rng = MersenneTwister(7))
        pa = Problems.cancellation(n; m, delta, aligned = true, rng = MersenneTwister(7))
        Hc, Ha = exact_hessian(pc), exact_hessian(pa)
        res = cover_hessian(WithDiagnostics(), pc.f, pc.x0)
        vc = cover_hessian(pc.f, pc.x0)

        @test res.h_quad ≈ sqrt.(diag(Ha)) rtol = 1.0e-8
        @test vc.h_var ≈ res.h_quad rtol = 1.0e-10

        hstar = sqrt.(diag(Hc))
        @test all(abs(Hc[i, j]) <= hstar[i] * hstar[j] * GATE for i in 1:n, j in 1:n)

        kappa = diag(Ha) ./ diag(Hc)
        @test res.h_quad ./ hstar ≈ sqrt.(kappa) rtol = 1.0e-8
    end

    @testset "looseness grows with m" begin
        function gm_looseness(mval)
            p = Problems.cancellation(16; m = mval, delta = 1.0e-2, aligned = false, rng = MersenneTwister(3))
            Hc = exact_hessian(p)
            hstar = sqrt.(diag(Hc))
            res = cover_hessian(WithDiagnostics(), p.f, p.x0)
            ratios = res.h_quad ./ hstar
            return exp(sum(log, ratios) / length(ratios))
        end
        @test gm_looseness(16) > 1.5 * gm_looseness(4)
    end

    @testset "support variant" begin
        support = 1:3
        p = Problems.cancellation(n; m = 4, delta = 1.0e-2, aligned = false, support, rng = MersenneTwister(3))
        Hc = exact_hessian(p)
        hstar = sqrt.(diag(Hc))
        res = cover_hessian(WithDiagnostics(), p.f, p.x0)
        for i in setdiff(1:n, support)
            @test res.h_quad[i] ≈ hstar[i] rtol = 1.0e-8
        end
        for i in support
            @test res.h_quad[i] >= 2 * hstar[i]
        end
    end

    @testset "sigma = 0.5" begin
        pc = Problems.cancellation(n; m, delta, aligned = false, sigma = 0.5, rng = MersenneTwister(7))
        pa = Problems.cancellation(n; m, delta, aligned = true, sigma = 0.5, rng = MersenneTwister(7))
        tc, oc = tape_function(pc.f, pc.x0)
        ta, oa = tape_function(pa.f, pa.x0)
        rc = cover_hessian(WithDiagnostics(), tc, oc)
        ra = cover_hessian(WithDiagnostics(), ta, oa)
        @test isapprox(rc.grad, ra.grad; rtol = 1.0e-12)
        @test rc.B == ra.B
        @test rc.h_sum == ra.h_sum
        @test rc.h_quad == ra.h_quad

        Hc = exact_hessian(pc)
        @test max_coverage_ratio(rc.h_sum, Hc) <= GATE
        @test max_coverage_ratio(rc.h_quad, Hc) <= GATE
    end

    @testset "argument errors" begin
        @test_throws ArgumentError Problems.cancellation(1)
        @test_throws ArgumentError Problems.cancellation(n; m = 0)
        @test_throws ArgumentError Problems.cancellation(n; delta = 0)
        @test_throws ArgumentError Problems.cancellation(n; support = [1])
        @test_throws ArgumentError Problems.cancellation(n; support = [1, 1, 2])
        @test_throws ArgumentError Problems.cancellation(n; support = [0, 1])
    end

    @testset "scale covariance" begin
        p = Problems.cancellation(n; m, delta, aligned = false, rng = MersenneTwister(1))
        d = exp.(0.3 .* randn(MersenneTwister(3), n))
        res = cover_hessian(WithDiagnostics(), p.f, p.x0)
        pr = rescale(p, d)
        resr = cover_hessian(WithDiagnostics(), pr.f, pr.x0)
        @test resr.h_sum ≈ res.h_sum ./ d rtol = 1.0e-10
    end

    @testset "magnitude methods pay the cancellation index — support $support" for support in (1:n, 1:3)
        pc = Problems.cancellation(n; m, delta, aligned = false, support, rng = MersenneTwister(7))
        pa = Problems.cancellation(n; m, delta, aligned = true, support, rng = MersenneTwister(7))
        Hc, Ha = exact_hessian(pc), exact_hessian(pa)
        tc, oc = tape_function(pc.f, pc.x0)
        ta, oa = tape_function(pa.f, pa.x0)
        rc = cover_hessian(WithDiagnostics(), tc, oc)

        # The sign flip that cancels the Hessian is invisible to the elementwise bound, which
        # matches the aligned Hessian on the diagonal.
        @test diag(rc.B) ≈ diag(Ha) rtol = 1.0e-8
        # The hybrid merge reads the same magnitudes as the other merges.
        @test cover_hybrid(tc, oc, 4) == cover_hybrid(ta, oa, 4)

        # A cover that cannot tell the variants apart must cover the aligned Hessian too, so on
        # the canceling variant it exceeds the minimal cover sqrt.(diag(Hc)) by at least
        # sqrt(kappa) on the support.
        hstar = sqrt.(diag(Hc))
        sqrtkappa = sqrt.(diag(Ha) ./ diag(Hc))
        vc = cover_hessian(tc, oc)
        for h in (rc.h_sum, rc.h_quad, cover_hybrid(tc, oc, 4), vc.h_var)
            @test all(h[support] .>= sqrtkappa[support] .* hstar[support] .* (1 - 1.0e-8))
        end

        # Hessian-vector products read signs. A probe along a coordinate no quartet touches sees
        # identical products; a probe inside the support separates the variants.
        for a in setdiff(1:n, support)
            e = zeros(n)
            e[a] = 1
            @test hvp(tc, oc, e) ≈ hvp(ta, oa, e) rtol = 1.0e-10
        end
        e = zeros(n)
        e[first(support)] = 1
        @test norm(hvp(tc, oc, e) - hvp(ta, oa, e)) > 1.0e-3 * norm(hvp(tc, oc, e))
    end

    @testset "the scaled spectral radius reads signs" begin
        pc = Problems.cancellation(n; m, delta, aligned = false, rng = MersenneTwister(7))
        pa = Problems.cancellation(n; m, delta, aligned = true, rng = MersenneTwister(7))
        Hc, Ha = exact_hessian(pc), exact_hessian(pa)
        hc = cover_hessian(pc.f, pc.x0).h_sum
        ha = cover_hessian(pa.f, pa.x0).h_sum
        @test hc == ha
        # On the canceling variant the scaled Hessian is far from saturated, so rescaling by its
        # norm separates covers that the sum merge returns identically.
        lc, la = scaled_spectral_radius(Hc, hc), scaled_spectral_radius(Ha, ha)
        @test lc < la < 1
        sc, sa = sqrt(lc) .* hc, sqrt(la) .* ha
        @test maximum(abs.(sc .- sa) ./ max.(sc, sa)) > 0.1
        @test max_coverage_ratio(sc, Hc) <= GATE
        @test max_coverage_ratio(sa, Ha) <= GATE
    end

    @testset "a recorded linear change of variables is the transported cover" begin
        # If abs.(H) <= h*h', then t = abs.(inv(A))' * h covers the Hessian of g(y) = f(inv(A)*y).
        # Recording inv(A)*y as linear nodes in front of f reproduces exactly that transport,
        # even when A is the eigenbasis that diagonalizes the canceling Hessian.
        pc = Problems.cancellation(n; m, delta, aligned = false, rng = MersenneTwister(7))
        Hc = exact_hessian(pc)
        A = permutedims(eigen(Symmetric(Hc)).vectors)
        Ainv = inv(A)
        g = ys -> pc.f([DerivativeCovers.linearnode(Ainv[a, :], ys) for a in 1:n])
        hx = cover_hessian(pc.f, pc.x0).h_sum
        hy = cover_hessian(g, A * pc.x0).h_sum
        @test hy ≈ abs.(permutedims(Ainv)) * hx rtol = 1.0e-13
        @test max_coverage_ratio(hy, permutedims(Ainv) * Hc * Ainv) <= GATE
    end

end
