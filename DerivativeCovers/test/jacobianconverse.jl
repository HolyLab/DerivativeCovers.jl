# Tests for the first-order (Jacobian) cancellation pair `Problems.jacobian_cancellation`. The
# family's claims: the aligned/cancel variants share every tape magnitude (so every cover built
# from magnitudes is identical on the two), the surrogate equals |J_aligned| exactly and is
# rank 1, and the balanced max-merge therefore pays exactly the per-column cancellation index
# kappa_j = |J_aligned|/|J_cancel| against the cancel member's minimal cover — no magnitude
# method can pay less, and the sum merge pays the output-multiplicity factor on top.

@testset "jacobianconverse.jl" begin

    # A cancel/aligned pair must share every random draw, so each member gets a fresh,
    # identically seeded rng.
    pair(; seed=1, kwargs...) =
        (Problems.jacobian_cancellation(8; aligned=false, rng=MersenneTwister(seed), kwargs...),
         Problems.jacobian_cancellation(8; aligned=true, rng=MersenneTwister(seed), kwargs...))

    @testset "analytic references — $name" for (name, kwargs) in [
        ("default", (;)),
        ("spread q", (; nout=6, m=5, sigma=0.6, seed=2)),
        ("anisotropic support", (; support=3:6, delta=1e-2, seed=3)),
    ]
        pm, pp = pair(; kwargs...)
        @test pm.x0 == pp.x0
        @test nvars(pm) == 8
        @test noutputs(pm) == noutputs(pp)
        @test has_twin(pm) && has_exact_jacobian(pm)
        for p in (pm, pp)
            J = exact_jacobian(p)
            @test ForwardDiff.jacobian(p.r, p.x0) ≈ J atol = 1e-10 * maximum(abs, J)
            @test ForwardDiff.jacobian(p.twin, p.x0) ≈ J atol = 1e-10 * maximum(abs, J)
            # The Jacobian is constant in x.
            @test ForwardDiff.jacobian(p.r, p.x0 .+ 0.7) ≈ J atol = 1e-10 * maximum(abs, J)
        end
        # Values agree at x0 up to the roundoff of the exactly-zero quartet forms, whose scale
        # is the total path magnitude |J_aligned|*|x0|, not the delta-scale output value.
        @test pm.r(pm.x0) ≈ pp.r(pp.x0) atol =
            1e-12 * maximum(abs.(exact_jacobian(pp)) * abs.(pm.x0))
    end

    @testset "indistinguishability: every magnitude object is bit-identical" begin
        pm, pp = pair(; nout=5, m=6, sigma=0.5, seed=4)
        cm = cover_jacobian(pm.r, pm.x0)
        cp = cover_jacobian(pp.r, pp.x0)
        @test cm.Utilde == cp.Utilde
        @test cm.b_sum == cp.b_sum && cm.b_quad == cp.b_quad && cm.b_max == cp.b_max
        tm, om = tape_function(pm.r, pm.x0)
        tp, op = tape_function(pp.r, pp.x0)
        @test cover_jacobian_hybrid(tm, om, 3) == cover_jacobian_hybrid(tp, op, 3)
        balm = balance_cover(cm.Utilde; merge=:max, arule=:geomean)
        balp = balance_cover(cp.Utilde; merge=:max, arule=:geomean)
        @test balm.a == balp.a && balm.b == balp.b
        # The signed Jacobian — outside the magnitude class — separates the pair decisively.
        @test maximum(abs, exact_jacobian(pp)) > 10 * maximum(abs, exact_jacobian(pm))
    end

    @testset "surrogate = |J_aligned| exactly, rank 1" begin
        pm, pp = pair(; nout=4, m=8, delta=1e-4, seed=5)
        U = jacobian_surrogate(tape_function(pm.r, pm.x0)...)
        Jp = exact_jacobian(pp)
        @test U ≈ abs.(Jp) rtol = 1e-12
        @test jacobian_representability(U) ≈ 1 rtol = 1e-10
        @test jacobian_minimax(U).excess ≈ 1 rtol = 1e-8
    end

    @testset "the floor: balanced max-merge pays exactly kappa, sum merge multiplies by q-mass" begin
        for kwargs in [(; nout=5, m=6, delta=1e-3, sigma=0.7, seed=6),
                       (; nout=3, m=4, delta=1e-1, support=2:5, seed=7)]
            pm, pp = pair(; kwargs...)
            Jm, Jp = exact_jacobian(pm), exact_jacobian(pp)
            kappa = vec(maximum(abs.(Jp) ./ abs.(Jm); dims=1))    # constant down each column
            cm = cover_jacobian(pm.r, pm.x0)

            # The minimax cover of the (rank-1) surrogate attains looseness kappa_j at every
            # entry of the cancel member — the theorem says no magnitude method beats it.
            bal = balance_cover(cm.Utilde; merge=:max, arule=:geomean)
            loose = (bal.a * bal.b') ./ abs.(Jm)
            @test loose ≈ ones(noutputs(pm)) * kappa' rtol = 1e-8

            # Off-support columns carry no cancellation: kappa = 1 there, exactly.
            supp = collect(get(kwargs, :support, 1:8))
            off = setdiff(1:8, supp)
            @test all(kappa[off] .≈ 1)
            isempty(off) || @test all(kappa[supp] .> 10)

            # Every shipped merge is in the magnitude class, so each pays at least the floor.
            for b in (cm.b_sum, cm.b_quad, cm.b_max)
                @test all((cm.a * b') ./ abs.(Jm) .>= (ones(noutputs(pm)) * kappa') .* (1 - 1e-10))
            end

            # The sum/max gap is the effective output multiplicity of q, uniform over inputs.
            qabs = abs.(Jm[:, findfirst(!iszero, vec(Jm[1, :]))])
            mult = sum(qabs) / maximum(qabs)
            @test cm.b_sum ./ cm.b_max ≈ fill(mult, 8) rtol = 1e-10
        end
    end

    @testset "single-path programs have no first-order cancellation" begin
        # Unique input→output paths make the surrogate equal |J| bit-for-bit: the abs sweep
        # multiplies and never adds two paths, and |a*b| == |a|*|b| exactly in floating point.
        # Bit-exactness holds against the tape's own signed sweep (identical operations up to
        # sign); ForwardDiff orders the arithmetic differently and can differ by an ulp.
        r = xs -> [tanh(xs[1] * xs[2]) * exp(xs[3]), sin(xs[4]) + xs[5] * xs[6]]
        x0 = [0.7, -0.3, 0.2, 0.9, -1.1, 0.4]
        t, outidxs = tape_function(r, x0)
        U = jacobian_surrogate(t, outidxs)
        @test U == abs.(adjoints(t, outidxs)[:, 1:t.ninputs])
        @test U ≈ abs.(ForwardDiff.jacobian(r, x0)) rtol = 1e-14
    end

    @testset "argument validation" begin
        @test_throws "n must be at least 2" Problems.jacobian_cancellation(1)
        @test_throws "nout must be positive" Problems.jacobian_cancellation(4; nout=0)
        @test_throws "m must be positive" Problems.jacobian_cancellation(4; m=0)
        @test_throws "delta must be positive" Problems.jacobian_cancellation(4; delta=0.0)
        @test_throws "support must be a subset" Problems.jacobian_cancellation(4; support=3:5)
        @test_throws "at least 2 entries" Problems.jacobian_cancellation(4; support=2:2)
        @test_throws "no analytic Jacobian" begin
            p = Problems.JacobianProblem("bare", xs -> [xs[1]], [1.0, 2.0], 1)
            Problems.exact_jacobian(p)
        end
    end
end
