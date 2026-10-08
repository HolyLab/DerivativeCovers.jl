# Compare Jacobian covers with ForwardDiff. Every merge must remain valid for positive output
# weights, with b_max <= b_quad <= b_sum and grouped merges between the endpoints.

@testset "jacobian.jl" begin

    W = randn(MersenneTwister(1), 5, 3)
    t5 = randn(MersenneTwister(2), 5)
    RESIDUAL_MAPS = [
        ("scalarized", xs -> [xs[1] * xs[2], sin(xs[3]), exp(xs[1]) + xs[3]], [0.6, -0.4, 0.9]),
        ("array-native matvec", xs -> W * xs, [0.6, -0.4, 0.9]),
        ("nonlinear rows", xs -> [tanh(y) - t5[k] for (k, y) in enumerate(W * xs)], [0.3, -0.2, 0.5]),
        ("per-coordinate square", xs -> [xs[j]^2 for j in 1:3], [0.7, -0.5, 1.2]),
    ]

    @testset "valid cover, in the merge chain — $name" for (name, r, x0) in RESIDUAL_MAPS
        J = ForwardDiff.jacobian(r, x0)
        jc = cover_jacobian(WithDiagnostics(), r, x0; a = ones(size(J, 1)))
        @test jc.m == size(J, 1)
        @test jc.n == size(J, 2)

        # Every merge covers J, for the default a = ones.
        for b in (jc.b_sum, jc.b_quad, jc.b_max)
            @test all(>=(0), b)
            @test jacobian_ratio(J, jc.a, b) <= GATE
        end
        # The ℓ¹ ≥ ℓ² ≥ ℓ∞ chain, coordinatewise.
        @test all(jc.b_max .<= jc.b_quad .* GATE)
        @test all(jc.b_quad .<= jc.b_sum .* GATE)
        # The surrogate bounds |J|, and b_max is its exact per-input max.
        @test all(jc.Utilde .>= abs.(J) .- 1.0e-12)
        @test jc.b_max ≈ vec(maximum(jc.Utilde ./ jc.a; dims = 1))

        # Group-max hybrid: one group is the sum merge, one output per group the exact merge, and
        # an intermediate partition is bracketed by and covers between them.
        t, outidxs = tape_function(r, x0)
        # The sweep tier reaches the same `a` and `b_sum` without forming the surrogate.
        jsweep = cover_jacobian(t, outidxs; a = ones(size(J, 1)))
        @test jsweep.a == jc.a
        @test jsweep.b_sum ≈ jc.b_sum
        @test jsweep.roots === nothing && jsweep.nties === nothing
        # The endpoint identities are statements about the uniform weights `jc` carries.
        @test cover_jacobian_hybrid(t, outidxs, 1; a = jc.a).b ≈ jc.b_sum
        @test cover_jacobian_hybrid(t, outidxs, jc.m; a = jc.a).b ≈ jc.b_max
        hyb = cover_jacobian_hybrid(t, outidxs, 2; a = jc.a).b
        @test jacobian_ratio(J, jc.a, hyb) <= GATE
        @test all(jc.b_max .<= hyb .* GATE)
        @test all(hyb .<= jc.b_sum .* GATE)
    end

    @testset "hand-checkable linear residuals" begin
        # r = [x₁ + 2x₂, 3x₁], J = [1 2; 3 0]. With a = ones the minimal b is the column max of |J|.
        r = xs -> [xs[1] + 2 * xs[2], 3 * xs[1]]
        x0 = [0.5, -0.3]
        jc = cover_jacobian(WithDiagnostics(), r, x0; a = ones(2))
        @test jc.Utilde ≈ [1.0 2.0; 3.0 0.0]
        @test jc.b_max ≈ [3.0, 2.0]
        @test jc.b_sum ≈ [4.0, 2.0]
        @test jc.b_quad ≈ [sqrt(10.0), 2.0]
    end

    @testset "valid for any positive output weights" begin
        r = xs -> [xs[1] * xs[2], sin(xs[3]), exp(xs[1]) + xs[3]]
        x0 = [0.6, -0.4, 0.9]
        J = ForwardDiff.jacobian(r, x0)
        a = [2.0, 0.5, 3.0]
        jc = cover_jacobian(WithDiagnostics(), r, x0; a)
        for b in (jc.b_sum, jc.b_quad, jc.b_max)
            @test jacobian_ratio(J, a, b) <= GATE
        end
        # Reweighting a rescales the surrogate rows but not the covering property.
        @test jc.b_max ≈ vec(maximum(jc.Utilde ./ a; dims = 1))

        @test_throws DimensionMismatch cover_jacobian(r, x0; a = [1.0, 1.0])
        @test_throws "strictly positive" cover_jacobian(r, x0; a = [1.0, 0.0, 1.0])
    end

    @testset "the b-side is covariant under input rescaling" begin
        # x → D*x sends J → J*inv(D), so |J| and the surrogate lose a factor d[j] on input j and
        # b → b ./ d, with the output weights a untouched. This is the property equilibration rests on.
        r = xs -> [xs[1] * xs[2], sin(xs[3]) + xs[1], exp(xs[2])]
        x0 = [0.6, -0.4, 0.9]
        d = [1.0e-3, 1.0e2, 5.0]
        jc = cover_jacobian(WithDiagnostics(), r, x0; a = ones(3))
        jcd = cover_jacobian(WithDiagnostics(), xs -> r(xs ./ d), d .* x0; a = ones(3))
        @test jcd.b_sum ≈ jc.b_sum ./ d rtol = 1.0e-10
        @test jcd.b_quad ≈ jc.b_quad ./ d rtol = 1.0e-10
        @test jcd.b_max ≈ jc.b_max ./ d rtol = 1.0e-10
    end

    @testset "duplicated and input-node outputs cover correctly" begin
        r = xs -> (y = xs[1] * xs[2]; [y, y, xs[3]])   # one node in two output slots, plus an input
        x0 = [0.7, -0.5, 1.2]
        J = ForwardDiff.jacobian(r, x0)
        jc = cover_jacobian(WithDiagnostics(), r, x0; a = ones(3))
        @test jc.m == 3
        @test jc.Utilde[1, :] == jc.Utilde[2, :]        # identical rows for the aliased output
        @test jacobian_ratio(J, jc.a, jc.b_max) <= GATE
    end

    @testset "jacobian_ratio" begin
        J = [1.0 2.0; 3.0 0.0]
        @test jacobian_ratio(J, [1.0, 3.0], [1.0, 2.0]) ≈ 1        # a = [1,3], b = [1,2] is exact
        @test jacobian_ratio(J, [2.0, 6.0], [2.0, 4.0]) ≈ 0.25     # doubling both sides
        @test jacobian_ratio(J, [1.0, 1.0], [0.0, 5.0]) == Inf     # J[1,1] over a vanishing product
        @test jacobian_ratio(zeros(2, 2), [1.0, 1.0], [1.0, 1.0]) == 0
        @test_throws DimensionMismatch jacobian_ratio(J, [1.0], [1.0, 2.0])
    end

    @testset "jacobian_representability lower-bounds any rank-1 cover" begin
        rng = MersenneTwister(3)
        # A rank-1 magnitude matrix is covered exactly, so the forced excess is 1.
        u, v = exp.(randn(rng, 5)), exp.(randn(rng, 3))
        @test jacobian_representability(u * v') ≈ 1

        # A 2×2 block: excess ≥ √defect, defect = (M₁₁M₂₂)/(M₁₂M₂₁).
        M = [1.0 2.0; 3.0 10.0]
        @test jacobian_representability(M) ≈ sqrt((1.0 * 10.0) / (2.0 * 3.0))

        # Two disjoint blocks force a large rank-1 fill: rows/cols across blocks give defect 1/ε².
        eps = 0.01
        B = [1.0 1.0 eps eps; 1.0 1.0 eps eps; eps eps 1.0 1.0; eps eps 1.0 1.0]
        @test jacobian_representability(B) ≈ 1 / eps

        # It is a genuine lower bound: no rank-1 cover of M can beat it.
        Mr = exp.(randn(rng, 6, 4))
        rep = jacobian_representability(Mr)
        excess(a, b) = maximum((a[k] * b[j]) / Mr[k, j] for k in 1:6, j in 1:4)
        jc = cover_jacobian(WithDiagnostics(), xs -> Mr * xs, zeros(4); a = ones(6), check = false)
        # Mr*xs has Jacobian Mr (all entries positive), so its b_max cover has excess ≥ rep.
        @test rep <= excess(jc.a, jc.b_max) * (1 + 1.0e-8)

        @test_throws "nonnegative" jacobian_representability([1.0 -2.0; 3.0 4.0])
    end

    @testset "jacobian_minimax is the exact minimax rank-1 cover" begin
        rng = MersenneTwister(5)

        # Reference: the optimal excess is the max over bipartite support cycles of the
        # alternating geometric mean, enumerable outright at these sizes.
        function brute_minimax(M)
            m, n = size(M)
            best = 0.0
            for r in 2:min(m, n)
                for rows in Iterators.product(fill(1:m, r)...),
                        cols in Iterators.product(fill(1:n, r)...)

                    (allunique(rows) && allunique(cols)) || continue
                    all(M[rows[i], cols[i]] > 0 for i in 1:r) || continue
                    all(M[rows[i], cols[mod1(i + 1, r)]] > 0 for i in 1:r) || continue
                    v = sum(
                        log(M[rows[i], cols[i]]) - log(M[rows[i], cols[mod1(i + 1, r)]])
                            for i in 1:r
                    ) / r
                    best = max(best, v)
                end
            end
            return exp(best)
        end

        for trial in 1:12
            m, n = rand(rng, 2:4), rand(rng, 2:4)
            M = exp.(3 .* randn(rng, m, n))
            trial > 6 && (M .*= rand(rng, m, n) .< 0.7)
            mm = jacobian_minimax(M)
            # Valid, attains the enumerated optimum, and is never beaten by the 2×2 bound.
            @test jacobian_ratio(M, mm.a, mm.b) <= GATE
            @test mm.excess ≈ brute_minimax(M) rtol = 1.0e-8
            @test mm.excess >= jacobian_representability(M) / GATE
        end

        # Rank-1 magnitudes are covered exactly; a tree-support pattern too (no cycles at all).
        u, v = exp.(randn(rng, 4)), exp.(randn(rng, 5))
        @test jacobian_minimax(u * v').excess ≈ 1
        T = [2.0 5.0 0.0; 0.0 3.0 7.0]
        mmT = jacobian_minimax(T)
        @test mmT.excess ≈ 1
        @test jacobian_ratio(T, mmT.a, mmT.b) <= GATE

        # On the 2×2 block matrix the 2-cycle bound is the whole story.
        eps = 0.01
        B = [1.0 1.0 eps eps; 1.0 1.0 eps eps; eps eps 1.0 1.0; eps eps 1.0 1.0]
        @test jacobian_minimax(B).excess ≈ 1 / eps rtol = 1.0e-8

        # A 3-cycle can dominate every 2-cycle: all 2×2 defects of C are mild, the 6-cycle is not.
        C = [1.0 1.0e3 0.0; 0.0 1.0 1.0e3; 1.0e3 0.0 1.0]
        @test jacobian_representability(C) ≈ 1        # no row pair shares two columns
        @test jacobian_minimax(C).excess ≈ 1.0e3 rtol = 1.0e-8

        # Zero rows and columns get exact zeros; the excess ignores them.
        Z = [0.0 0.0 0.0; 0.0 2.0 4.0; 0.0 4.0 2.0]
        mmZ = jacobian_minimax(Z)
        @test mmZ.a[1] == 0 && mmZ.b[1] == 0
        @test mmZ.excess ≈ 2.0 rtol = 1.0e-8

        # Diagonal rescaling moves the pair covariantly and leaves the excess invariant.
        M = exp.(2 .* randn(rng, 5, 4))
        C2, D2 = exp2.(rand(rng, -10:10, 5)), exp2.(rand(rng, -10:10, 4))
        @test jacobian_minimax(C2 .* M .* D2').excess ≈ jacobian_minimax(M).excess rtol = 1.0e-9

        @test_throws "nonnegative" jacobian_minimax([1.0 -2.0; 3.0 4.0])
        @test_throws "atol must be positive" jacobian_minimax([1.0 2.0; 3.0 4.0]; atol = 0)
    end

    @testset "cover_jacobian rejects a scalar program" begin
        @test_throws "vector-valued program" cover_jacobian(xs -> xs[1] * xs[2], [1.0, 2.0])
    end

    @testset "abs_forward is the transpose of the reverse sweep" begin
        r = xs -> [tanh(y) for y in W * xs]
        x0 = [0.3, -0.2, 0.5]
        t, outidxs = tape_function(r, x0)
        U = jacobian_surrogate(t, outidxs)
        # Seeding e_j at input j and reading the outputs gives column j of the surrogate.
        for j in 1:3
            theta = zeros(nnodes(t))
            theta[j] = 1.0
            @test abs_forward(t, theta)[outidxs] ≈ U[:, j]
        end
        # Linearity in the seeds: a weighted input seed gives the same weighted column combination.
        wts = [0.7, 2.0, 0.1]
        theta = zeros(nnodes(t))
        theta[1:3] = wts
        @test abs_forward(t, theta)[outidxs] ≈ U * wts
        @test_throws DimensionMismatch abs_forward(t, zeros(nnodes(t) - 1))
    end

    @testset "marginal_weights: covariant a-rule, valid for any positive b" begin
        r = xs -> [xs[1] * xs[2], sin(xs[3]) + xs[1], exp(xs[2]), xs[2] * xs[3]]
        x0 = [0.6, -0.4, 0.9]
        t, outidxs = tape_function(r, x0)
        U = jacobian_surrogate(t, outidxs)
        J = ForwardDiff.jacobian(r, x0)
        for b in ([1.0, 1.0, 1.0], [0.7, 2.0, 0.1], [1.0e-3, 1.0e2, 5.0])
            a = marginal_weights(t, outidxs, b)
            @test a ≈ marginal_weights(U, b)                       # tape path = matrix path
            @test a ≈ vec(sum(U ./ b'; dims = 2))
            @test jacobian_ratio(J, a, b) <= GATE                  # every (marginal a, b) covers
            @test jacobian_ratio(U, a, b) <= GATE
        end
        @test_throws DimensionMismatch marginal_weights(t, outidxs, [1.0, 2.0])
        @test_throws "nonnegative" marginal_weights(t, outidxs, [1.0, -1.0, 1.0])

        # A zero b is accepted only on a dead input.
        rdead = xs -> [xs[1]^2, 2.0 * xs[1]]                       # xs[2] unused
        td, od = tape_function(rdead, [0.7, 1.3])
        a = marginal_weights(td, od, [1.0, 0.0])
        @test a ≈ marginal_weights(jacobian_surrogate(td, od), [1.0, 0.0])
        @test_throws "no output weighting" marginal_weights(td, od, [0.0, 1.0])
        @test_throws "no output weighting" marginal_weights(U, [0.0, 1.0, 1.0])
    end

    @testset "balance_cover: valid at every iterate, stationary at the fixed point" begin
        # Run to convergence with uniform targets, which is Sinkhorn scaling and converges only on
        # total-support patterns, so both maps here have fully dense surrogates.
        conv = (; targets = :uniform, rounds = nothing)
        maps = [
            ("nonlinear rows", xs -> [tanh(y) - t5[k] for (k, y) in enumerate(W * xs)], [0.3, -0.2, 0.5]),
            (
                "dense scalarized", xs -> [
                    xs[1] * xs[2] + xs[3], sin(xs[3]) + xs[1] * xs[2],
                    exp(xs[1]) + xs[2] * xs[3],
                ], [0.6, -0.4, 0.9],
            ),
        ]
        @testset "$name" for (name, r, x0) in maps
            t, outidxs = tape_function(r, x0)
            U = jacobian_surrogate(t, outidxs)
            J = ForwardDiff.jacobian(r, x0)

            bal = balance_cover(t, outidxs; conv...)
            @test jacobian_ratio(J, bal.a, bal.b) <= GATE
            @test jacobian_ratio(U, bal.a, bal.b) <= GATE
            @test bal.delta <= 1.0e-10

            # The tape path is the matrix path applied to the surrogate. The two forms start from
            # different covariant starts, which fix different gauges, so the comparison is on the
            # products — the gauge-independent content of the pair.
            balm = balance_cover(U; conv...)
            @test bal.a .* bal.b' ≈ balm.a .* balm.b' rtol = 1.0e-8

            # The default construction — covariant start, path-count targets, four rounds, sum
            # merge — needs no convergence and covers.
            bdef = balance_cover(t, outidxs)
            @test bdef.iters == 4
            @test jacobian_ratio(U, bdef.a, bdef.b) <= GATE
            @test jacobian_ratio(J, bdef.a, bdef.b) <= GATE

            # Every iterate of the alternation is a valid cover: replay it from the pieces.
            a = ones(length(outidxs))
            b = vec(sum(U ./ a; dims = 1))
            for _ in 1:5
                a = marginal_weights(t, outidxs, b)
                @test jacobian_ratio(U, a, b) <= GATE              # after the a-step
                b = vec(sum(U ./ a; dims = 1))
                @test jacobian_ratio(U, a, b) <= GATE              # after the b-step
            end

            # Stationarity: one more full round changes every product by at most ~rtol.
            a2 = marginal_weights(t, outidxs, bal.b)
            b2 = vec(sum(U ./ a2; dims = 1))
            @test a2 .* b2' ≈ bal.a .* bal.b' rtol = 1.0e-8

            # :quad, :max, and the geomean a-rule reach their own fixed points and stay valid.
            # (Fixed points of different rules are not comparable entrywise; tightness
            # comparisons live in the measurement scripts.)
            for kwargs in (
                    (; merge = QuadMerge(), conv...), (; merge = MaxMerge(), conv...),
                    (; arule = :geomean, rounds = nothing),
                    (; merge = MaxMerge(), arule = :geomean, rounds = nothing),
                )
                balv = balance_cover(t, outidxs; kwargs...)
                @test jacobian_ratio(U, balv.a, balv.b) <= GATE
                @test jacobian_ratio(J, balv.a, balv.b) <= GATE
            end
        end
    end

    @testset "balance_cover: check is honored on every call form" begin
        # A dense surrogate: every merge/arule combination converges on it, so any failure here
        # is about `check` rather than about the alternation.
        rmap = xs -> [tanh(y) - t5[k] for (k, y) in enumerate(W * xs)]
        t, outidxs = tape_function(rmap, [0.3, -0.2, 0.5])
        U = jacobian_surrogate(t, outidxs)

        # The matrix form and the tape form's delegating branch (any merge/arule other than
        # sum/marginal) both take `check`; skipping verification leaves the pair unchanged.
        for kwargs in (
                (;), (; merge = MaxMerge()), (; arule = :geomean),
                (; merge = QuadMerge(), arule = :geomean),
                (; targets = :uniform, rounds = nothing),
            )
            balm = balance_cover(U; check = false, kwargs...)
            @test balm.a == balance_cover(U; kwargs...).a
            @test balm.b == balance_cover(U; kwargs...).b

            balt = balance_cover(t, outidxs; check = false, kwargs...)
            @test balt.a == balance_cover(t, outidxs; kwargs...).a
            @test balt.b == balance_cover(t, outidxs; kwargs...).b
        end
    end

    @testset "balance_cover: support-deficient patterns" begin
        # Ũ has support (a positive diagonal) but not total support: entry (3,3) lies on no
        # positive diagonal, so the Sinkhorn alternation drifts without bound while every
        # iterate still covers. The geomean a-rule is the convergent alternative.
        r = xs -> [xs[1] * xs[2], sin(xs[3]), exp(xs[1]) + xs[3]]
        x0 = [0.6, -0.4, 0.9]
        t, outidxs = tape_function(r, x0)
        U = jacobian_surrogate(t, outidxs)
        J = ForwardDiff.jacobian(r, x0)
        @test_throws "total support" balance_cover(
            t, outidxs;
            targets = :uniform, rounds = nothing, maxiter = 50
        )
        @test_throws "total support" balance_cover(
            U;
            targets = :uniform, rounds = nothing, maxiter = 50
        )
        for kwargs in (
                (; arule = :geomean, rounds = nothing),
                (; arule = :geomean, merge = MaxMerge(), rounds = nothing),
                (; targets = :flow, rounds = nothing),
            )
            bal = balance_cover(t, outidxs; kwargs...)
            @test jacobian_ratio(U, bal.a, bal.b) <= GATE
            @test jacobian_ratio(J, bal.a, bal.b) <= GATE
            @test bal.delta <= 1.0e-10
        end
        # The two-way log fit is exact in one round on a complete (dense) pattern.
        rng = MersenneTwister(9)
        Ud = exp.(randn(rng, 4, 3))
        bald = balance_cover(Ud; arule = :geomean, start = :equal, rounds = nothing)
        @test bald.iters <= 2
    end

    @testset "balance_cover: drift is reported as drift" begin
        # Without total support the weights separate without bound. Every iterate still covers,
        # but its excess grows with the round count, so running longer returns a worse cover and
        # eventually leaves the floating-point range. Given room to run, that is diagnosed as
        # divergence rather than left to exhaust the budget.
        Utri = [1.0 1.0; 0.0 1.0]
        uni = (; targets = :uniform, rounds = nothing)
        drift = try
            balance_cover(Utri; uni..., maxiter = 10^6)
            nothing
        catch e
            e
        end
        @test drift isa ErrorException
        @test occursin("diverged", drift.msg)
        @test occursin("worse cover", drift.msg)
        @test occursin("geomean", drift.msg)
        # Caught by the divergence check, not by the (large) budget.
        rounds = parse(Int, match(r"diverged after (\d+) rounds", drift.msg)[1])
        @test rounds < 10^4
        # Below the detection point the plain non-convergence report still stands.
        @test_throws "did not converge" balance_cover(Utri; uni..., maxiter = 20)
        # The convergent alternative on the same pattern.
        balg = balance_cover(Utri; arule = :geomean, rounds = nothing)
        @test jacobian_ratio(Utri, balg.a, balg.b) <= GATE

        # A pattern that merely converges slowly must not be mistaken for drift: this banded map
        # needs thousands of rounds under the sum merge and still reaches its fixed point.
        chain = xs -> [tanh(xs[i]) * xs[i + 1] for i in 1:(length(xs) - 1)]
        tc, oc = tape_function(chain, 0.7 .* randn(MersenneTwister(3), 16))
        Uc = jacobian_surrogate(tc, oc)
        slow = balance_cover(Uc; uni..., start = :equal, maxiter = 10^5)
        @test slow.iters > 2000
        @test jacobian_ratio(Uc, slow.a, slow.b) <= GATE
    end

    @testset "balance_cover: an output no input reaches" begin
        # A constant output has an identically zero surrogate row, so the alternation drives its
        # weight to exactly zero. The tape form admits that only after an indicator sweep confirms
        # no input reaches the output: a zero weight on a live output could cover nothing.
        r = xs -> [xs[1] * xs[2], 3 * one(eltype(xs)), xs[1] + xs[2]]
        x0 = [0.7, 1.3]
        t, outidxs = tape_function(r, x0)
        U = jacobian_surrogate(t, outidxs)
        @test U[2, :] == zeros(2)

        bal = balance_cover(t, outidxs; rounds = nothing)
        @test bal.a[2] == 0
        @test all(>(0), bal.a[[1, 3]])
        @test jacobian_ratio(U, bal.a, bal.b) <= GATE

        # The tape route and the materialized surrogate reach the same products, up to the gauge
        # each start leaves behind. Every entry here lies on one path, so the path counts and the
        # support counts are the same targets.
        balm = balance_cover(U; rounds = nothing)
        @test balm.a[2] == 0
        @test bal.a .* bal.b' ≈ balm.a .* balm.b' rtol = 1.0e-8

        # `cover_jacobian` accepts the zero weight the dead output gets, from either start, and so
        # does the grouped merge.
        for start in (:covariant, :equal)
            bs = balance_cover(t, outidxs; start)
            jc = cover_jacobian(WithDiagnostics(), t, outidxs; a = bs.a, check = true)
            @test jc.a[2] == 0
            @test all(isfinite, jc.b_sum)
            @test jacobian_ratio(U, jc.a, jc.b_max) <= GATE
            @test cover_jacobian_hybrid(t, outidxs, 3; a = bs.a).b ≈ jc.b_max
            hyb = cover_jacobian_hybrid(t, outidxs, 2; a = bs.a).b
            @test jacobian_ratio(U, bs.a, hyb) <= GATE
            @test all(jc.b_max .<= hyb .* GATE)
            @test all(hyb .<= jc.b_sum .* GATE)
        end

        # A zero weight on a live output cannot cover it, whichever entry point sees it.
        @test_throws "strictly positive" cover_jacobian(t, outidxs; a = [0.0, 0.0, 1.0])
        @test_throws "strictly positive" cover_jacobian_hybrid(t, outidxs, 2; a = [0.0, 0.0, 1.0])
        @test_throws "strictly positive" cover_jacobian(t, outidxs; a = [1.0, 0.0, -1.0])
        @test_throws "must be finite" cover_jacobian(t, outidxs; a = [1.0, 0.0, NaN])
        @test_throws "must be finite" cover_jacobian(t, outidxs; a = [1.0, 0.0, Inf])
    end

    @testset "balance_cover: rank-1 exactness and the Sinkhorn fixed point" begin
        rng = MersenneTwister(7)
        u, v = exp.(randn(rng, 6)), exp.(randn(rng, 4))
        U = u * v'
        # The annotation supplies the rule check: there is no list of admissible spellings to
        # fall out of date, and SumMerge() is the default in both call forms.
        @test_throws TypeError balance_cover(U; merge = :max)
        @test_throws "arule must be :marginal or :geomean" balance_cover(U; arule = :sinkhorn)
        @test balance_cover(U; merge = SumMerge()).b == balance_cover(U).b
        @test SumMerge() isa MergeRule && QuadMerge() isa MergeRule && MaxMerge() isa MergeRule

        # The max-merge fixed point recovers a rank-1 magnitude matrix exactly.
        balx = balance_cover(U; merge = MaxMerge(), targets = :uniform, rounds = nothing)
        @test balx.a .* balx.b' ≈ U
        # The sum-merge fixed point with uniform targets is the Sinkhorn scaling: every row and
        # column of the ratio matrix R = U ./ (a .* b') sums to 1 after its own half-step; on
        # rank-1 the uniform excess is exactly m.
        bals = balance_cover(U; targets = :uniform, rounds = nothing)
        @test maximum((bals.a .* bals.b') ./ U) ≈ 6 rtol = 1.0e-8
        @test minimum((bals.a .* bals.b') ./ U) ≈ 6 rtol = 1.0e-8
        R = U ./ (bals.a .* bals.b')
        @test vec(sum(R; dims = 1)) ≈ ones(4) rtol = 1.0e-8              # b-step ran last

        # Zero rows and columns persist as exact zero weights.
        Uz = [1.0 0.0 2.0; 0.0 0.0 0.0; 3.0 0.0 1.0]
        balz = balance_cover(Uz)
        @test balz.a[2] == 0
        @test balz.b[2] == 0
        @test jacobian_ratio(Uz, balz.a, balz.b) <= GATE

        @test_throws "did not converge" balance_cover(
            exp.(randn(rng, 3, 5));
            targets = :uniform, rounds = nothing, maxiter = 1
        )
        @test_throws "strictly positive" balance_cover(U; targets = :uniform, a0 = zeros(6))
        @test_throws "nonnegative" balance_cover([1.0 -1.0; 1.0 1.0])
        # `rounds` and `maxiter` are alternative bounds, not a pair.
        @test_throws "cannot be combined" balance_cover(U; rounds = 2, maxiter = 10)

        # The function form tapes the program and hands the tape to the two-argument form, so on a
        # vector-valued program the two agree exactly; a scalar output has no rows to balance.
        rvec = xs -> [
            xs[1] * xs[2] + xs[3], sin(xs[3]) + xs[1] * xs[2],
            exp(xs[1]) + xs[2] * xs[3],
        ]
        xvec = [0.6, -0.4, 0.9]
        balf = balance_cover(rvec, xvec)
        balt = balance_cover(tape_function(rvec, xvec)...)
        @test balf.a == balt.a
        @test balf.b == balt.b
        @test_throws "vector-valued program" balance_cover(xs -> xs[1] * xs[2], [1.0, 2.0])
    end

    @testset "balanced cover covariance, bit-exact for power-of-two factors" begin
        rbase = xs -> [xs[1] * xs[2], sin(xs[3]) + xs[1], exp(xs[2]), xs[2] * xs[3]]
        x0 = [0.6, -0.4, 0.9]

        # Input rescaling x → d .* x: a is untouched and b picks up exactly 1 ./ d. The wrapper
        # is applied on both sides so the two tapes have identical shape.
        d = [0.25, 4.0, 2.0]
        e = ones(3)
        te, oe = tape_function(xs -> rbase(xs ./ e), x0)
        td, od = tape_function(xs -> rbase(xs ./ d), d .* x0)
        bale = balance_cover(te, oe)
        bald = balance_cover(td, od)
        @test bald.a == bale.a
        @test bald.b == bale.b ./ d
        @test bald.iters == bale.iters

        # Output rescaling r → c .* r with the equal start transformed alongside (a0 = c): every
        # product a_k*b_j picks up exactly c_k. The equal start itself is not c-equivariant, so a
        # finite run from it matches only to the convergence tolerance, while the covariant start
        # is equivariant at any round count.
        c = [2.0, 0.5, 8.0, 0.25]
        t1, o1 = tape_function(xs -> ones(4) .* rbase(xs), x0)
        tc, oc = tape_function(xs -> c .* rbase(xs), x0)
        uni = (; targets = :uniform, rounds = nothing)
        bal1 = balance_cover(t1, o1; uni..., start = :equal)
        balc = balance_cover(tc, oc; uni..., a0 = c)
        @test balc.a .* balc.b' == (c .* bal1.a) .* bal1.b'
        balc0 = balance_cover(tc, oc; uni..., start = :equal)
        @test balc0.a .* balc0.b' ≈ (c .* bal1.a) .* bal1.b' rtol = 1.0e-8

        # The default construction is c-equivariant at its fixed four-round budget.
        dflt1 = balance_cover(t1, o1)
        dfltc = balance_cover(tc, oc)
        @test dfltc.a .* dfltc.b' ≈ (c .* dflt1.a) .* dflt1.b' rtol = 1.0e-10

        # The marginal rule alone is exactly d-invariant and c-equivariant.
        b = [0.7, 2.0, 0.1]
        @test marginal_weights(td, od, b ./ d) == marginal_weights(te, oe, b)
        @test marginal_weights(tc, oc, b) == c .* marginal_weights(t1, o1, b)
    end

    @testset "the surrogate and the b-merges transform exactly" begin
        # x -> d .* x and r -> c .* r send the surrogate to diag(c) * Ũ * diag(d)^-1. The first-order
        # sweeps have no zero-value guard, so this holds at every point, bit-exactly for
        # power-of-two factors. Wrappers on both sides give the compared tapes the same shape.
        rbase = xs -> [
            xs[1] * xs[2] + xs[3], sin(xs[3]) + xs[1] * xs[2],
            exp(xs[1]) + xs[2] * xs[3], tanh(xs[1]) + xs[2] + xs[3],
        ]
        x0 = [0.6, -0.4, 0.9]
        surrogate(r, x) = jacobian_surrogate(tape_function(r, x)...)
        for (d, c, exact) in (
                ([0.25, 4.0, 2.0], [2.0, 0.5, 8.0, 0.25], true),
                ([1.0e-3, 7.3, 260.0], [3.7, 0.11, 42.0, 1.9], false),
            )
            Ue = surrogate(xs -> rbase(xs ./ ones(3)), x0)
            Ud = surrogate(xs -> rbase(xs ./ d), d .* x0)
            Uo = surrogate(xs -> ones(4) .* rbase(xs), x0)
            Uc = surrogate(xs -> c .* rbase(xs), x0)
            if exact
                @test Ud == Ue ./ d'
                @test Uc == c .* Uo
            else
                @test Ud ≈ Ue ./ d' rtol = 1.0e-12
                @test Uc ≈ c .* Uo rtol = 1.0e-12
            end
        end

        # With the output weights held fixed, every b-merge picks up exactly 1 ./ d.
        d = [0.25, 4.0, 2.0]
        te, oe = tape_function(xs -> rbase(xs ./ ones(3)), x0)
        td, od = tape_function(xs -> rbase(xs ./ d), d .* x0)
        jce = cover_jacobian(WithDiagnostics(), te, oe; a = ones(4))
        jcd = cover_jacobian(WithDiagnostics(), td, od; a = ones(4))
        @test jcd.b_sum == jce.b_sum ./ d
        @test jcd.b_quad == jce.b_quad ./ d
        @test jcd.b_max == jce.b_max ./ d
        @test cover_jacobian_hybrid(td, od, 2; a = ones(4)).b ==
            cover_jacobian_hybrid(te, oe, 2; a = ones(4)).b ./ d

        # The marginal rule is invariant under a generic input rescaling to roundoff.
        dg = [1.0e-3, 7.3, 260.0]
        tg, og = tape_function(xs -> rbase(xs ./ dg), dg .* x0)
        b = [0.7, 2.0, 0.1]
        @test marginal_weights(tg, og, b ./ dg) ≈ marginal_weights(te, oe, b) rtol = 1.0e-12

        # Sinkhorn balancing from the equal start: every product a_k*b_j and the round count are
        # unchanged by a power-of-two input rescaling.
        uni = (; targets = :uniform, start = :equal, rounds = nothing)
        bale = balance_cover(te, oe; uni...)
        bald = balance_cover(td, od; uni...)
        @test bald.a .* bald.b' == bale.a .* (bale.b ./ d)'
        @test bald.iters == bale.iters
    end

    @testset "output weightings that are not covariant" begin
        # Weighting outputs by the residual magnitude |r_k| gives a = 0 at a zero-residual
        # solution of a consistent least-squares problem, where J is not zero: no b covers.
        rres = xs -> [xs[1] - 1.0, xs[1] * xs[2] - 2.0]
        xsol = [1.0, 2.0]
        @test all(iszero, rres(xsol))
        @test jacobian_ratio(ForwardDiff.jacobian(rres, xsol), abs.(rres(xsol)), [1.0, 1.0]) == Inf

        # The unweighted row sum a_k = sum_j Ũ_kj adds entries in the units of different inputs,
        # so an input rescaling moves each output's weight by a different factor.
        rdisj = xs -> [xs[1]^2, sin(xs[2]), exp(xs[3]), xs[1] * xs[2]]
        x0 = [0.6, -0.4, 0.9]
        d = [1.0e-3, 7.3, 260.0]
        rowsum(r, x) = vec(sum(jacobian_surrogate(tape_function(r, x)...); dims = 2))
        moved = rowsum(xs -> rdisj(xs ./ d), d .* x0) ./ rowsum(xs -> rdisj(xs ./ ones(3)), x0)
        @test maximum(moved) / minimum(moved) > 10
        # The marginal rule weights each entry by 1/b_j and is unmoved.
        b = [0.7, 2.0, 0.1]
        te, oe = tape_function(xs -> rdisj(xs ./ ones(3)), x0)
        td, od = tape_function(xs -> rdisj(xs ./ d), d .* x0)
        @test marginal_weights(td, od, b ./ d) ≈ marginal_weights(te, oe, b) rtol = 1.0e-12
    end

    @testset "balance_cover: max-merge limit cycle fails fast to geomean" begin
        # This pattern has total support, but max-merge marginal balancing enters a limit cycle.
        # Stall detection should fail early and recommend the convergent geomean rule.
        Ucyc = [
            0.0 0.5 0.0 0.0
            2.0 0.5 0.0 0.0
            0.0 0.0 0.25 4.0
            0.0 0.0 0.5 2.0
            0.5 0.0 1.0 0.5
        ]

        uni = (; targets = :uniform, rounds = nothing)
        @test balance_cover(Ucyc; uni...).iters < 1000        # :sum/marginal (Sinkhorn) converges

        stall = try
            balance_cover(Ucyc; uni..., merge = MaxMerge(), maxiter = 100_000)
            nothing
        catch e
            e
        end
        @test stall isa ErrorException
        @test occursin("limit cycle", stall.msg)
        @test occursin("geomean", stall.msg)
        # Detected as a stall, not by exhausting the (large) budget.
        @test occursin("stalled", stall.msg)

        bal = balance_cover(Ucyc; merge = MaxMerge(), arule = :geomean, rounds = nothing)
        @test jacobian_ratio(Ucyc, bal.a, bal.b) <= GATE

        # A fixed budget has nothing to stall: it returns what the rounds reached, and `delta`
        # reports the plateau rather than raising it as an error.
        budgeted = balance_cover(Ucyc; targets = :uniform, merge = MaxMerge(), rounds = 200)
        @test budgeted.iters == 200
        @test budgeted.delta > 0
        @test jacobian_ratio(Ucyc, budgeted.a, budgeted.b) <= GATE
    end

    @testset "balance_cover: geomean round budget on slow supports" begin
        # This ill-conditioned support converges under geomean but needs its larger default budget.
        L = 20
        Uslow = zeros(L, L)
        for k in 1:L
            Uslow[k, k] = 1.0
            k < L && (Uslow[k, k + 1] = 2.0)
            k > 1 && (Uslow[k, k - 1] = 0.5)
        end

        bal = balance_cover(
            Uslow;
            merge = MaxMerge(), arule = :geomean, start = :equal, rounds = nothing
        )
        @test jacobian_ratio(Uslow, bal.a, bal.b) <= GATE
        @test bal.iters > 1000                                # exceeds the marginal budget

        # The same fit under a fixed budget stops where the budget runs out, still covering.
        short = balance_cover(Uslow; merge = MaxMerge(), arule = :geomean, rounds = 4)
        @test short.iters == 4
        @test jacobian_ratio(Uslow, short.a, short.b) <= GATE

        undersized = try
            balance_cover(
                Uslow;
                merge = MaxMerge(), arule = :geomean, start = :equal, rounds = nothing,
                maxiter = 10
            )
            nothing
        catch e
            e
        end
        @test undersized isa ErrorException
        @test occursin("maxiter", undersized.msg)
        @test occursin("geomean", undersized.msg)
    end

end
