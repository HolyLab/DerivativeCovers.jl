# The aligned and cancelling Jacobian fixtures share all tape magnitudes. These tests isolate the
# unavoidable cost of signed cancellation for magnitude-only covers.

@testset "jacobianconverse.jl" begin

    # Separate identically seeded RNGs keep both variants on the same random draws.
    pair(; seed = 1, kwargs...) =
        (
        Problems.jacobian_cancellation(8; aligned = false, rng = MersenneTwister(seed), kwargs...),
        Problems.jacobian_cancellation(8; aligned = true, rng = MersenneTwister(seed), kwargs...),
    )

    @testset "analytic references — $name" for (name, kwargs) in [
            ("default", (;)),
            ("spread q", (; nout = 6, m = 5, sigma = 0.6, seed = 2)),
            ("anisotropic support", (; support = 3:6, delta = 1.0e-2, seed = 3)),
        ]
        pm, pp = pair(; kwargs...)
        @test pm.x0 == pp.x0
        @test nvars(pm) == 8
        @test noutputs(pm) == noutputs(pp)
        @test has_twin(pm) && has_exact_jacobian(pm)
        for p in (pm, pp)
            J = exact_jacobian(p)
            @test ForwardDiff.jacobian(p.r, p.x0) ≈ J atol = 1.0e-10 * maximum(abs, J)
            @test ForwardDiff.jacobian(p.twin, p.x0) ≈ J atol = 1.0e-10 * maximum(abs, J)
            # The Jacobian is constant in x.
            @test ForwardDiff.jacobian(p.r, p.x0 .+ 0.7) ≈ J atol = 1.0e-10 * maximum(abs, J)
        end
        # Values agree at x0 up to the roundoff of the exactly-zero quartet forms, whose scale
        # is the total path magnitude |J_aligned|*|x0|, not the delta-scale output value.
        @test pm.r(pm.x0) ≈ pp.r(pp.x0) atol =
            1.0e-12 * maximum(abs.(exact_jacobian(pp)) * abs.(pm.x0))
    end

    @testset "indistinguishability: every magnitude object is bit-identical" begin
        pm, pp = pair(; nout = 5, m = 6, sigma = 0.5, seed = 4)
        cm = cover_jacobian(WithDiagnostics(), pm.r, pm.x0)
        cp = cover_jacobian(WithDiagnostics(), pp.r, pp.x0)
        @test cm.Utilde == cp.Utilde
        @test cm.b_sum == cp.b_sum && cm.b_quad == cp.b_quad && cm.b_max == cp.b_max
        tm, om = tape_function(pm.r, pm.x0)
        tp, op = tape_function(pp.r, pp.x0)
        @test cover_jacobian_hybrid(tm, om, 3).b == cover_jacobian_hybrid(tp, op, 3).b
        balm = balance_cover(cm.Utilde; merge = MaxMerge(), arule = :geomean)
        balp = balance_cover(cp.Utilde; merge = MaxMerge(), arule = :geomean)
        @test balm.a == balp.a && balm.b == balp.b
        # The signed Jacobian — outside the magnitude class — separates the pair decisively.
        @test maximum(abs, exact_jacobian(pp)) > 10 * maximum(abs, exact_jacobian(pm))
    end

    @testset "surrogate = |J_aligned| exactly, rank 1" begin
        pm, pp = pair(; nout = 4, m = 8, delta = 1.0e-4, seed = 5)
        U = jacobian_surrogate(tape_function(pm.r, pm.x0)...)
        Jp = exact_jacobian(pp)
        @test U ≈ abs.(Jp) rtol = 1.0e-12
        @test jacobian_representability(U) ≈ 1 rtol = 1.0e-10
        @test jacobian_minimax(U).excess ≈ 1 rtol = 1.0e-8
    end

    @testset "the floor: balanced max-merge pays exactly kappa, sum merge multiplies by q-mass" begin
        for kwargs in [
                (; nout = 5, m = 6, delta = 1.0e-3, sigma = 0.7, seed = 6),
                (; nout = 3, m = 4, delta = 1.0e-1, support = 2:5, seed = 7),
            ]
            pm, pp = pair(; kwargs...)
            Jm, Jp = exact_jacobian(pm), exact_jacobian(pp)
            kappa = vec(maximum(abs.(Jp) ./ abs.(Jm); dims = 1))    # constant down each column
            # Uniform output weights: the merge-gap identity below is a statement about them.
            cm = cover_jacobian(WithDiagnostics(), pm.r, pm.x0; a = ones(noutputs(pm)))

            # The minimax cover of the (rank-1) surrogate attains looseness kappa_j at every
            # entry of the cancel member — the theorem says no magnitude method beats it.
            bal = balance_cover(cm.Utilde; merge = MaxMerge(), arule = :geomean)
            loose = (bal.a * bal.b') ./ abs.(Jm)
            @test loose ≈ ones(noutputs(pm)) * kappa' rtol = 1.0e-8

            # Off-support columns carry no cancellation: kappa = 1 there, exactly.
            supp = collect(get(kwargs, :support, 1:8))
            off = setdiff(1:8, supp)
            @test all(kappa[off] .≈ 1)
            isempty(off) || @test all(kappa[supp] .> 10)

            # Every shipped merge is in the magnitude class, so each pays at least the floor.
            for b in (cm.b_sum, cm.b_quad, cm.b_max)
                @test all((cm.a * b') ./ abs.(Jm) .>= (ones(noutputs(pm)) * kappa') .* (1 - 1.0e-10))
            end

            # The sum/max gap is the effective output multiplicity of q, uniform over inputs.
            qabs = abs.(Jm[:, findfirst(!iszero, vec(Jm[1, :]))])
            mult = sum(qabs) / maximum(qabs)
            @test cm.b_sum ./ cm.b_max ≈ fill(mult, 8) rtol = 1.0e-10
        end
    end

    @testset "off the support, no scalar correction can tighten a magnitude cover" begin
        pm, pp = pair(; nout = 4, m = 8, delta = 1.0e-3, sigma = 0.5, support = 3:6, seed = 2)
        Jm, Jp = exact_jacobian(pm), exact_jacobian(pp)
        kappa = vec(maximum(abs.(Jp) ./ abs.(Jm); dims = 1))
        off = setdiff(1:8, 3:6)
        @test all(==(1), kappa[off])
        @test all(kappa[3:6] .> 10)
        # At every off-support input some output is covered with no slack, so c*b fails to cover
        # for any c < 1, and the on-support excess kappa cannot be traded away by a scalar.
        cm = cover_jacobian(WithDiagnostics(), pm.r, pm.x0; a = ones(4))
        bal = balance_cover(cm.Utilde; merge = MaxMerge(), arule = :geomean, start = :equal, rounds = nothing)
        for (a, b) in ((cm.a, cm.b_max), (bal.a, bal.b)), j in off
            @test minimum(a[k] * b[j] / abs(Jm[k, j]) for k in axes(Jm, 1)) <= GATE
        end
    end

    @testset "signed products separate the pair; blind directions do not" begin
        pm, pp = pair(; nout = 4, m = 8, delta = 1.0e-3, sigma = 0.5, support = 3:6, seed = 2)
        jvp(p, v) = ForwardDiff.derivative(s -> p.r(p.x0 .+ s .* v), 0.0)
        vjp(p, w) = ForwardDiff.gradient(x -> dot(w, p.r(x)), p.x0)
        e1, e3 = zeros(8), zeros(8)
        e1[1] = 1
        e3[3] = 1
        # A Jacobian-vector product along an input no quartet touches is blind to the sign; one
        # along a support input is not.
        @test jvp(pm, e1) ≈ jvp(pp, e1) rtol = 1.0e-10
        @test maximum(abs, jvp(pp, e3)) > 10 * maximum(abs, jvp(pm, e3))
        # Every column of the canceling Jacobian is proportional to the output scales q, so a
        # vector-Jacobian product along any w orthogonal to q is blind, and one along q is not.
        Jm = exact_jacobian(pm)
        q = Jm[:, findfirst(!iszero, vec(Jm[1, :]))]
        w = nullspace(q')[:, 1]
        @test maximum(abs, vjp(pm, w) .- vjp(pp, w)) < 1.0e-10 * maximum(abs, vjp(pp, q))
        @test maximum(abs, vjp(pp, q)) > 10 * maximum(abs, vjp(pm, q))
    end

    @testset "the sum/max gap is bounded by the outputs reaching each input" begin
        # Equal output scales: the effective multiplicity is the output count itself.
        pm, _ = pair(; nout = 5, m = 4, delta = 1.0e-3, sigma = 0.0, seed = 3)
        cm = cover_jacobian(WithDiagnostics(), pm.r, pm.x0; a = ones(5))
        @test cm.b_sum ./ cm.b_max ≈ fill(5, 8) rtol = 1.0e-12

        # Windowed residuals: output k depends on x[k:k+w-1], so at most w outputs reach an input
        # and b_sum <= w * b_max there.
        rng = MersenneTwister(4)
        n, w = 12, 3
        nout = n - w + 1
        coeff = ifelse.(rand(rng, nout, w) .< 0.5, -1.0, 1.0) .* (abs.(randn(rng, nout, w)) .+ 0.1)
        r = xs -> map(1:nout) do k
            acc = coeff[k, 1] * xs[k]
            for l in 2:w
                acc = acc + coeff[k, l] * xs[k + l - 1]
            end
            sin(acc) + acc^2 / 2
        end
        jc = cover_jacobian(WithDiagnostics(), r, randn(rng, n); a = ones(nout))
        reach = vec(count(>(0), jc.Utilde; dims = 1))
        @test maximum(reach) == w
        @test all(jc.b_sum .<= reach .* jc.b_max .* GATE)
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
        @test U ≈ abs.(ForwardDiff.jacobian(r, x0)) rtol = 1.0e-14
    end

    @testset "argument validation" begin
        @test_throws "n must be at least 2" Problems.jacobian_cancellation(1)
        @test_throws "nout must be positive" Problems.jacobian_cancellation(4; nout = 0)
        @test_throws "m must be positive" Problems.jacobian_cancellation(4; m = 0)
        @test_throws "delta must be positive" Problems.jacobian_cancellation(4; delta = 0.0)
        @test_throws "support must be a subset" Problems.jacobian_cancellation(4; support = 3:5)
        @test_throws "at least 2 entries" Problems.jacobian_cancellation(4; support = 2:2)
        @test_throws "no analytic Jacobian" begin
            p = Problems.JacobianProblem("bare", xs -> [xs[1]], [1.0, 2.0], 1)
            Problems.exact_jacobian(p)
        end
    end
end
