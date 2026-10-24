# Path-count targets, tape potentials, and the covariant output weighting they combine into.
#
# The defining property is covariance: rescaling the program's inputs, outputs or intermediates
# must move the products `a_k*b_j` by exactly the rescaling factors, at every round count.

@testset "covariant.jl" begin

    # The tape recorded with node ℓ's value rescaled by s[ℓ]: an incoming slot picks up s[ℓ] and
    # loses its parent's scale, which is how `d_{ℓ,i} = ∂φ_ℓ/∂v_{p_i}` transforms.
    function rescale_tape(t::Tape{T}, s::AbstractVector) where {T}
        ds = [copy(d) for d in t.ds]
        for k in 1:nnodes(t), i in eachindex(t.parents[k])
            ds[k][i] = t.ds[k][i] * s[k] / s[t.parents[k][i]]
        end
        return Tape{T}(
            T[t.vals[k] * s[k] for k in 1:nnodes(t)], [copy(p) for p in t.parents],
            ds, copy(t.Cs), t.ninputs
        )
    end

    # Worst discrepancy, in logs, between the rescaled products and the transformation law, over
    # the support of the surrogate.
    function product_drift(t, outidxs, s, a, b, a2, b2)
        U = jacobian_surrogate(t, outidxs)
        worst = 0.0
        for j in 1:t.ninputs, k in eachindex(outidxs)
            U[k, j] > 0 || continue
            expected = log(s[outidxs[k]]) - log(s[j])
            worst = max(worst, abs(log(a2[k] * b2[j]) - log(a[k] * b[j]) - expected))
        end
        return worst
    end

    triangular = xs -> [xs[1] * xs[2], sin(xs[3]), exp(xs[1]) + xs[3]]
    hall = xs -> [3 * xs[1], xs[1]^2, sin(xs[1]), xs[1] * xs[2]]
    zeroderiv = xs -> [xs[1]^2, xs[1] * xs[2]]          # ∂(x₁²)/∂x₁ = 0 at the point below
    repeated = xs -> [xs[1] * xs[1], xs[2] + xs[1]]     # one node in two parent slots
    twoblocks = xs -> [xs[1] * xs[2], exp(xs[3]) + xs[4]]
    # One node in two output slots, and that same node feeding a third output.
    aliased = xs -> (y = xs[1] * xs[2]; [y, y, y + xs[3]])
    chain = xs -> [tanh(xs[i]) * xs[i + 1] for i in 1:(length(xs) - 1)]

    PROGRAMS = [
        ("triangular", triangular, [0.6, -0.4, 0.9]),
        ("hall 4x2", hall, [0.7, 1.3]),
        ("zero derivative", zeroderiv, [0.0, 1.0]),
        ("repeated slots", repeated, [0.7, -1.1]),
        ("two blocks", twoblocks, [0.7, -0.4, 0.5, 1.2]),
        ("aliased outputs", aliased, [0.7, -1.1, 0.5]),
        ("chain n=6", chain, [0.7, -0.5, 1.1, -0.9, 0.4, 1.3]),
        ("dense", xs -> [xs[1] * xs[2] + xs[3], sin(xs[3]) + xs[1] * xs[2], exp(xs[1]) + xs[2] * xs[3]], [0.6, -0.4, 0.9]),
    ]

    liveinfo(t, outidxs) = DerivativeCovers._live_info(t, outidxs)
    flowtargets(t, outidxs) =
        DerivativeCovers._flow_targets(t, outidxs, liveinfo(t, outidxs))
    supportcounts(t, outidxs) =
        DerivativeCovers._support_counts(t, outidxs, liveinfo(t, outidxs))

    # The weight matrix a slot weighting implies, one reverse sweep per output, so the sweep-based
    # marginals can be checked against it entry by entry.
    function flow_weight_matrix(t, outidxs, kind)
        DC = DerivativeCovers
        li = liveinfo(t, outidxs)
        w = kind === :fwd ? DC.ForwardFlow(li) : DC.ReverseFlow(li)
        W = zeros(length(outidxs), t.ninputs)
        for (k, o) in enumerate(outidxs)
            li.live[o] || continue
            seed = zeros(nnodes(t))
            seed[o] = kind === :fwd ? 1.0 : 1 / li.outdeg[o]
            W[k, :] .= DC._reverse_sweep(w, t, seed)[1:t.ninputs]
        end
        return W
    end

    @testset "flow targets are the marginals of a pattern-matching weight matrix — $name" for
        (name, r, x0) in PROGRAMS
        t, outidxs = tape_function(r, x0)
        U = jacobian_surrogate(t, outidxs)
        li = liveinfo(t, outidxs)
        Wf, Wr = flow_weight_matrix(t, outidxs, :fwd), flow_weight_matrix(t, outidxs, :rev)

        # The two constants the flows are built to have, taken rather than swept.
        @test all(sum(Wf; dims = 2)[k] ≈ 1 for k in 1:length(outidxs) if li.liverow[k])
        @test all(sum(Wr; dims = 1)[j] ≈ 1 for j in 1:t.ninputs if li.livecol[j])

        # Both weight matrices carry the surrogate's pattern, so the mixture's marginals are
        # realizable on it. The mixture is read straight off the explicit matrices.
        @test (Wf .> 0) == (U .> 0)
        @test (Wr .> 0) == (U .> 0)
        rt, ct = flowtargets(t, outidxs)
        @test rt ≈ vec(sum(Wf; dims = 2)) ./ li.nrow .+ vec(sum(Wr; dims = 2)) ./ li.ncol
        @test ct ≈ vec(sum(Wf; dims = 1)) ./ li.nrow .+ vec(sum(Wr; dims = 1)) ./ li.ncol
        # The totals agree by construction, without any rescaling.
        @test sum(rt) ≈ sum(ct) rtol = 1.0e-12
        @test sum(rt) ≈ 2 rtol = 1.0e-12
        @test findall(iszero, rt) == findall(k -> all(iszero, view(U, k, :)), 1:length(outidxs))
        @test findall(iszero, ct) == findall(j -> all(iszero, view(U, :, j)), 1:t.ninputs)
        @test all(<=(2 + 1.0e-12), rt) && all(<=(2 + 1.0e-12), ct)
    end

    @testset "support counts equal the surrogate's nonzero counts — $name" for
        (name, r, x0) in PROGRAMS
        t, outidxs = tape_function(r, x0)
        U = jacobian_surrogate(t, outidxs)
        rs, cs = supportcounts(t, outidxs)
        @test rs == [count(!iszero, view(U, k, :)) for k in 1:length(outidxs)]
        @test cs == [count(!iszero, view(U, :, j)) for j in 1:t.ninputs]
        @test sum(rs) == sum(cs)
    end

    @testset "support counts span several bit blocks" begin
        # 130 inputs and 130 outputs need three forward and three reverse passes.
        nwide = 130
        wide = xs -> [xs[k] * xs[mod1(k + 1, nwide)] for k in 1:nwide]
        t, outidxs = tape_function(wide, collect(0.5:0.01:(0.5 + 0.01 * (nwide - 1))))
        U = jacobian_surrogate(t, outidxs)
        rs, cs = supportcounts(t, outidxs)
        @test rs == [count(!iszero, view(U, k, :)) for k in 1:nwide]
        @test cs == [count(!iszero, view(U, :, j)) for j in 1:nwide]
    end

    @testset "a node in two output slots is weighted twice" begin
        t, outidxs = tape_function(aliased, [0.7, -1.1, 0.5])
        U = jacobian_surrogate(t, outidxs)
        @test outidxs[1] == outidxs[2]                  # the same node fills two output slots
        @test U[1, :] == U[2, :]
        rt, ct = flowtargets(t, outidxs)
        @test rt[1] ≈ rt[2]
        rs, cs = supportcounts(t, outidxs)
        @test rs[1] == rs[2]
        @test sum(rt) ≈ sum(ct)
        @test sum(rs) == sum(cs)
    end

    @testset "flow targets do not sacrifice few-path entries" begin
        # A repeated-doubling chain gives one output exponentially many paths and the other two.
        # Weighting by path count would drive that second output's tightness to nothing.
        function doubler(N)
            return xs -> begin
                y, z = xs[1], xs[2]
                for _ in 1:N
                    y = y * z + y
                end
                return [y, z * xs[1]]
            end
        end
        x0 = [0.5, 0.001]
        t, outidxs = tape_function(doubler(60), x0)
        U = jacobian_surrogate(t, outidxs)
        q(a, b) = [U[k, j] / (a[k] * b[j]) for k in 1:2, j in 1:2 if U[k, j] > 0]

        cflow = cover_jacobian(t, outidxs)
        cones = cover_jacobian(t, outidxs; a = ones(2))
        @test minimum(q(cflow.a, cflow.b_sum)) >= 0.01
        @test exp(sum(log, q(cflow.a, cflow.b_sum)) / length(q(cflow.a, cflow.b_sum))) >
            exp(sum(log, q(cones.a, cones.b_sum)) / length(q(cones.a, cones.b_sum)))

        # A depth that overflowed path counts runs with finite weights.
        tdeep, odeep = tape_function(doubler(1100), x0)
        cdeep = cover_jacobian(tdeep, odeep)
        @test all(isfinite, cdeep.a) && all(>(0), cdeep.a)
        @test all(isfinite, cdeep.b_sum)
    end

    @testset "targets: pattern-feasible where uniform is not" begin
        t, outidxs = tape_function(triangular, [0.6, -0.4, 0.9])
        U = jacobian_surrogate(t, outidxs)
        J = ForwardDiff.jacobian(triangular, [0.6, -0.4, 0.9])
        @test_throws "total support" balance_cover(
            t, outidxs;
            targets = :uniform, rounds = nothing, maxiter = 50
        )

        bal = balance_cover(t, outidxs; targets = :flow, rounds = nothing)
        @test bal.delta <= 1.0e-10
        @test max_coverage_ratio(bal.a, bal.b, U) <= GATE
        @test max_coverage_ratio(bal.a, bal.b, J) <= GATE

        # The matrix-free tape path and the explicit matrix path reach the same pair.
        rt, ct = flowtargets(t, outidxs)
        balm = balance_cover(U; targets = (rt, ct), rounds = nothing, a0 = bal.a)
        @test bal.a ≈ balm.a rtol = 1.0e-8
        @test bal.b ≈ balm.b rtol = 1.0e-8
        # `:support` is the matrix form's built-in pattern-feasible choice.
        bals = balance_cover(U; targets = :support, rounds = nothing)
        @test max_coverage_ratio(bals.a, bals.b, U) <= GATE

        # Non-sum merges take the same normalized alternation and only change the final b-step,
        # so their products are no larger than the sum merge's. (The comparison is on products:
        # the delegating path starts elsewhere and so fixes a different gauge.)
        for merge in (QuadMerge(), MaxMerge())
            balq = balance_cover(t, outidxs; targets = :flow, rounds = nothing, merge)
            @test max_coverage_ratio(balq.a, balq.b, U) <= GATE
            @test all(balq.a .* balq.b' .<= bal.a .* bal.b' .* GATE)
        end
    end

    @testset "targets: a rectangular pattern no uniform scaling can match" begin
        # Three of the four outputs depend on x₁ alone, so the uniform row targets exceed what
        # column 1 can supply.
        x0 = [0.7, 1.3]
        t, outidxs = tape_function(hall, x0)
        U = jacobian_surrogate(t, outidxs)
        J = ForwardDiff.jacobian(hall, x0)
        @test_throws ErrorException balance_cover(
            t, outidxs;
            targets = :uniform, rounds = nothing, maxiter = 200
        )

        bal = balance_cover(t, outidxs; targets = :flow, rounds = nothing)
        @test bal.delta <= 1.0e-10
        @test max_coverage_ratio(bal.a, bal.b, U) <= GATE
        @test max_coverage_ratio(bal.a, bal.b, J) <= GATE
    end

    @testset "targets: explicit marginals are validated" begin
        t, outidxs = tape_function(triangular, [0.6, -0.4, 0.9])
        U = jacobian_surrogate(t, outidxs)
        @test_throws DimensionMismatch balance_cover(t, outidxs; targets = ([1.0, 1.0], ones(3)))
        @test_throws DimensionMismatch balance_cover(U; targets = (ones(3), [1.0, 1.0]))
        @test_throws "not positive on a live output" balance_cover(
            t, outidxs;
            targets = ([1.0, 0.0, 2.0], ones(3))
        )
        @test_throws "target totals must agree" balance_cover(
            t, outidxs;
            targets = (ones(3), [1.0, 1.0, 2.0])
        )
        @test_throws "must be :flow, :support" balance_cover(t, outidxs; targets = :pathcounts)
        @test_throws "needs a tape" balance_cover(U; targets = :flow)
        @test_throws "prescribes no marginals" balance_cover(
            t, outidxs;
            targets = :flow, arule = :geomean
        )

        # A dead row's target must be zero, matching the weight it will keep.
        td, od = tape_function(xs -> [xs[1] * xs[2], 3 * one(eltype(xs))], [0.7, 1.3])
        @test_throws "nonzero on a dead output" balance_cover(
            td, od;
            targets = ([1.0, 1.0], [1.0, 1.0])
        )
        bald = balance_cover(td, od; targets = :flow, rounds = nothing)
        @test bald.a[2] == 0
        @test max_coverage_ratio(bald.a, bald.b, jacobian_surrogate(td, od)) <= GATE
    end

    @testset "tape_potentials and the covariant start cover — $name" for (name, r, x0) in PROGRAMS
        t, outidxs = tape_function(r, x0)
        U = jacobian_surrogate(t, outidxs)
        J = ForwardDiff.jacobian(r, x0)
        pot = tape_potentials(t, outidxs)

        @test length(pot.theta) == nnodes(t)
        @test all(isnan, pot.theta[.!pot.live])
        @test all(isfinite, pot.theta[pot.live])
        @test all(>=(1), pot.nties)
        @test length(pot.roots) == length(pot.nties)
        @test all(v -> pot.live[v], pot.roots)

        for rounds in (0, 1, 4), cgiters in (0, 5)
            cw = balance_cover(t, outidxs; rounds, cgiters, check = true)
            @test max_coverage_ratio(cw.a, cw.b, U) <= GATE
            @test max_coverage_ratio(cw.a, cw.b, J) <= GATE
            @test all(>=(0), cw.a)
            @test all(>=(0), cw.b)
        end
    end

    @testset "covariance under node rescaling — $name" for (name, r, x0) in PROGRAMS
        t, outidxs = tape_function(r, x0)
        rng = MersenneTwister(11)
        s = 2.0 .^ rand(rng, -10:10, nnodes(t))
        t2 = rescale_tape(t, s)

        for rounds in (0, 1, 4), cgiters in (0, 5)
            cw = balance_cover(t, outidxs; rounds, cgiters)
            cw2 = balance_cover(t2, outidxs; rounds, cgiters)
            @test product_drift(t, outidxs, s, cw.a, cw.b, cw2.a, cw2.b) < 1.0e-10
        end
    end

    @testset "a start of ones is not covariant at a finite budget" begin
        # The path-count targets are invariant and the alternation is equivariant, so only the
        # start can break covariance — and a constant start does, until convergence.
        t, outidxs = tape_function(triangular, [0.6, -0.4, 0.9])
        s = ones(nnodes(t))
        for (k, o) in enumerate(outidxs)
            s[o] = 2.0^(3k)                     # distinct per output: a rescaling, not a gauge
        end
        t2 = rescale_tape(t, s)
        m = length(outidxs)
        bal = balance_cover(t, outidxs; start = :equal, rounds = 4)
        bal2 = balance_cover(t2, outidxs; start = :equal, rounds = 4)
        @test product_drift(t, outidxs, s, bal.a, bal.b, bal2.a, bal2.b) > 1.0e-3

        # The potential start closes that gap at the same budget.
        cw = balance_cover(t, outidxs; rounds = 4)
        cw2 = balance_cover(t2, outidxs; rounds = 4)
        @test product_drift(t, outidxs, s, cw.a, cw.b, cw2.a, cw2.b) < 1.0e-10
    end

    @testset "live components match the bipartite support components" begin
        t, outidxs = tape_function(twoblocks, [0.7, -0.4, 0.5, 1.2])
        pot, comp = DerivativeCovers._tape_potentials(t, outidxs)
        sc = support_components(t, outidxs)
        @test length(pot.roots) == 2
        @test ncomponents(sc) == 2

        # The two labelings agree up to renumbering: each live output and input lands in the same
        # part of both partitions.
        pairing = Dict{Int, Int}()
        for (k, o) in enumerate(outidxs)
            pairing[comp[o]] = rowcomponent(sc, k)
        end
        for (k, o) in enumerate(outidxs)
            @test pairing[comp[o]] == rowcomponent(sc, k)
        end
        for j in 1:t.ninputs
            @test pairing[comp[j]] == colcomponent(sc, j)
        end
    end

    @testset "roots: the rule, the override, and the ties" begin
        t, outidxs = tape_function(chain, [0.7, -0.5, 1.1, -0.9, 0.4, 1.3])
        pot = tape_potentials(t, outidxs)
        @test length(pot.roots) == 1
        # A chain ties massively, so the root is settled by the node index rather than the graph.
        @test pot.nties[1] > 1

        # An explicit root is honored and moves the potentials by a constant per component.
        other = findfirst(v -> pot.live[v] && v != pot.roots[1], 1:nnodes(t))
        potr = tape_potentials(t, outidxs; root = other)
        @test potr.roots == [other]
        @test potr.theta[other] == 0
        @test potr.nties == [1]                  # a supplied root broke no tie

        dead = findfirst(!, pot.live)
        if dead !== nothing
            @test_throws "not a live node" tape_potentials(t, outidxs; root = dead)
        end
        @test_throws "not a live node" tape_potentials(t, outidxs; root = nnodes(t) + 1)
        @test_throws "at most one per component" tape_potentials(
            t, outidxs;
            root = [pot.roots[1], other]
        )
        @test_throws "cgiters must be nonnegative" tape_potentials(t, outidxs; cgiters = -1)
        @test_throws "rounds must be nonnegative" balance_cover(t, outidxs; rounds = -1)
    end

    @testset "converged potentials are root independent" begin
        t, outidxs = tape_function(triangular, [0.6, -0.4, 0.9])
        U = jacobian_surrogate(t, outidxs)
        pot = tape_potentials(t, outidxs)
        other = findfirst(v -> pot.live[v] && v != pot.roots[1], 1:nnodes(t))

        # CG run far past convergence stops at the residual floor rather than dividing roundoff by
        # roundoff, so the result stays finite and stays covariant.
        cw1 = balance_cover(t, outidxs; rounds = 0, cgiters = 10_000)
        cw2 = balance_cover(t, outidxs; rounds = 0, cgiters = 10_000, root = other)
        @test all(isfinite, tape_potentials(t, outidxs; cgiters = 10_000).theta[pot.live])
        for j in 1:t.ninputs, k in eachindex(outidxs)
            U[k, j] > 0 || continue
            @test log(cw1.a[k] * cw1.b[j]) ≈ log(cw2.a[k] * cw2.b[j]) atol = 1.0e-8
        end

        s = 2.0 .^ rand(MersenneTwister(5), -10:10, nnodes(t))
        t2 = rescale_tape(t, s)
        cws = balance_cover(t2, outidxs; rounds = 0, cgiters = 10_000)
        @test product_drift(t, outidxs, s, cw1.a, cw1.b, cws.a, cws.b) < 1.0e-10
    end

    @testset "dead inputs and dead outputs stay at zero" begin
        # x₃ enters nothing, and the middle output is a constant.
        r = xs -> [xs[1] * xs[2], 3 * one(eltype(xs)), xs[1] + xs[2]]
        t, outidxs = tape_function(r, [0.7, 1.3, -0.5])
        U = jacobian_surrogate(t, outidxs)
        rt, ct = flowtargets(t, outidxs)
        @test rt[2] == 0
        @test ct[3] == 0

        cw = balance_cover(t, outidxs; check = true)
        @test cw.a[2] == 0
        @test cw.b[3] == 0
        @test all(>(0), cw.a[[1, 3]])
        @test max_coverage_ratio(cw.a, cw.b, U) <= GATE
        @test isnan(tape_potentials(t, outidxs).theta[3])
    end

    @testset "cover_jacobian: the two tiers, the two starts, and explicit weights" begin
        t, outidxs = tape_function(triangular, [0.6, -0.4, 0.9])
        U = jacobian_surrogate(t, outidxs)

        # The tiers agree on what they share, for every start and budget.
        for start in (:covariant, :equal), rounds in (0, 1, 4)
            c = cover_jacobian(t, outidxs; start, rounds)
            d = cover_jacobian(WithDiagnostics(), t, outidxs; start, rounds)
            @test c.a == d.a
            @test c.b_sum == d.b_sum
            @test c.m == d.m == 3 && c.n == d.n == 3
            @test max_coverage_ratio(c.a, c.b_sum, U) <= GATE
            @test all(d.b_max .<= d.b_quad .* GATE)
            @test all(d.b_quad .<= d.b_sum .* GATE)
            @test (start === :covariant) == (c.roots !== nothing)
            @test (start === :covariant) == (c.nties !== nothing)
        end

        # `a = ones` reproduces the plain uniform-weight cover both tiers used to give.
        d1 = cover_jacobian(WithDiagnostics(), t, outidxs; a = ones(3))
        @test d1.a == ones(3)
        @test d1.b_max ≈ vec(maximum(U; dims = 1))
        @test d1.b_sum ≈ vec(sum(U; dims = 1))
        @test d1.b_quad ≈ sqrt.(vec(sum(abs2, U; dims = 1)))
        @test cover_jacobian(t, outidxs; a = ones(3)).b_sum ≈ d1.b_sum

        # Explicit weights replace the construction outright.
        @test_throws "start and rounds do not apply" cover_jacobian(
            t, outidxs;
            a = ones(3), start = :equal
        )
        @test_throws "start and rounds do not apply" cover_jacobian(
            WithDiagnostics(), t, outidxs; a = ones(3), rounds = 2
        )
        @test_throws DimensionMismatch cover_jacobian(t, outidxs; a = ones(2))

        # The sweep tier reaches no surrogate, with `check` either way.
        for check in (false, true)
            @test cover_jacobian(t, outidxs; check).b_sum ==
                cover_jacobian(t, outidxs; check = !check).b_sum
        end
    end

    @testset "cover_jacobian: the starts carry their own symmetry" begin
        t, outidxs = tape_function(triangular, [0.6, -0.4, 0.9])
        rng = MersenneTwister(23)
        s = 2.0 .^ rand(rng, -10:10, nnodes(t))
        t2 = rescale_tape(t, s)
        for rounds in (0, 1, 4)
            c = cover_jacobian(t, outidxs; rounds)
            c2 = cover_jacobian(t2, outidxs; rounds)
            @test product_drift(t, outidxs, s, c.a, c.b_sum, c2.a, c2.b_sum) < 1.0e-10
        end

        # The equal start is permutation equivariant: reversing the inputs and the outputs of the
        # program permutes the products and nothing else.
        rev = xs -> reverse(triangular(reverse(xs)))
        tr, orev = tape_function(rev, reverse([0.6, -0.4, 0.9]))
        for rounds in (0, 1, 4)
            c = cover_jacobian(t, outidxs; start = :equal, rounds)
            cr = cover_jacobian(tr, orev; start = :equal, rounds)
            @test reverse(cr.a) ≈ c.a rtol = 1.0e-12
            @test reverse(cr.b_sum) ≈ c.b_sum rtol = 1.0e-12
        end
    end

    @testset "cover_jacobian_hybrid: the same weighting as cover_jacobian" begin
        t, outidxs = tape_function(triangular, [0.6, -0.4, 0.9])
        U = jacobian_surrogate(t, outidxs)

        # The default weights are the ones cover_jacobian chooses, for every start and budget.
        for start in (:covariant, :equal), rounds in (0, 4)
            c = cover_jacobian(WithDiagnostics(), t, outidxs; start, rounds)
            @test cover_jacobian_hybrid(t, outidxs, 1; start, rounds).b ≈ c.b_sum
            @test cover_jacobian_hybrid(t, outidxs, 3; start, rounds).b ≈ c.b_max
            h2 = cover_jacobian_hybrid(t, outidxs, 2; start, rounds)
            @test h2.a == c.a
            hyb = h2.b
            @test max_coverage_ratio(c.a, hyb, U) <= GATE
            @test all(c.b_max .<= hyb .* GATE)
            @test all(hyb .<= c.b_sum .* GATE)
        end

        # Uniform weights reproduce the plain grouped merge.
        @test cover_jacobian_hybrid(t, outidxs, 1; a = ones(3)).b ≈ vec(sum(U; dims = 1))
        @test cover_jacobian_hybrid(t, outidxs, 3; a = ones(3)).b ≈ vec(maximum(U; dims = 1))
        # The function form tapes the program and agrees with the tape form.
        @test cover_jacobian_hybrid(triangular, [0.6, -0.4, 0.9], 2).b ==
            cover_jacobian_hybrid(t, outidxs, 2).b
        @test cover_jacobian_hybrid(triangular, [0.6, -0.4, 0.9], [1, 1, 2]).b ==
            cover_jacobian_hybrid(t, outidxs, [1, 1, 2]).b
        @test_throws "vector-valued program" cover_jacobian_hybrid(
            xs -> xs[1] * xs[2], [1.0, 2.0], 2
        )

        @test_throws "start and rounds do not apply" cover_jacobian_hybrid(
            t, outidxs, 2; a = ones(3), start = :equal
        )
        @test_throws "start and rounds do not apply" cover_jacobian_hybrid(
            t, outidxs, [1, 1, 2]; a = ones(3), rounds = 2
        )
        @test_throws DimensionMismatch cover_jacobian_hybrid(t, outidxs, 2; a = ones(2))

        # A dead output carries a zero weight and seeds nothing.
        rd = xs -> [xs[1] * xs[2], 3 * one(eltype(xs)), xs[1] + xs[2]]
        td, od = tape_function(rd, [0.7, 1.3])
        Ud = jacobian_surrogate(td, od)
        for start in (:covariant, :equal)
            ad = cover_jacobian(td, od; start).a
            @test ad[2] == 0
            bd = cover_jacobian_hybrid(td, od, 2; start).b
            @test all(isfinite, bd)
            @test max_coverage_ratio(ad, bd, Ud) <= GATE
        end
    end

    @testset "cover_jacobian: dead rows and columns through both tiers" begin
        # x₃ enters nothing and the middle output is a constant.
        r = xs -> [xs[1] * xs[2], 3 * one(eltype(xs)), xs[1] + xs[2]]
        t, outidxs = tape_function(r, [0.7, 1.3, -0.5])
        U = jacobian_surrogate(t, outidxs)
        for start in (:covariant, :equal)
            c = cover_jacobian(t, outidxs; start)
            d = cover_jacobian(WithDiagnostics(), t, outidxs; start)
            for x in (c, d)
                @test x.a[2] == 0
                @test x.b_sum[3] == 0
                @test all(isfinite, x.a) && all(isfinite, x.b_sum)
                @test max_coverage_ratio(x.a, x.b_sum, U) <= GATE
            end
            @test all(isfinite, d.b_quad) && all(isfinite, d.b_max)
            @test max_coverage_ratio(d.a, d.b_max, U) <= GATE
        end
    end

    @testset "edge cases: underflow, budgets, sentinels" begin
        # A log-scale spread too wide for centering on the mean: centering on the midrange keeps
        # both ends representable.
        wide = xs -> [1.0e300 * xs[1], 1.0e-300 * xs[2], 1.0e-300 * xs[1] * xs[2]]
        t, outidxs = tape_function(wide, [1.0, 1.0])
        c = cover_jacobian(t, outidxs)
        @test all(isfinite, c.a) && all(>(0), c.a)
        @test all(isfinite, c.b_sum)
        @test max_coverage_ratio(c.a, c.b_sum, jacobian_surrogate(t, outidxs)) <= GATE
        # A spread nothing can hold is reported as a range failure naming the weight, never as the
        # zero-weight symptom it used to surface as.
        wider = try
            cover_jacobian(xs -> [1.0e300 * exp(xs[1]), 1.0e-300 * exp(-xs[2])], [700.0, 700.0])
            nothing
        catch e
            sprint(showerror, e)
        end
        @test wider !== nothing
        @test occursin("floating-point range", wider) || occursin("representable", wider)
        @test !occursin("is zero on live output", wider)

        # A structurally live row whose surrogate entries underflow is reported as underflow,
        # not blamed on the caller's targets or treated as a dead output.
        under = xs -> [(1.0e-200 * xs[1]) * 1.0e-200, xs[1] * xs[2]]
        tu, ou = tape_function(under, [1.0, 1.0])
        @test flowtargets(tu, ou)[1][1] > 0                    # structurally live
        @test all(iszero, jacobian_surrogate(tu, ou)[1, :])    # numerically zero
        @test_throws "underflowed to zero" balance_cover(tu, ou; merge = QuadMerge())
        @test_throws "underflowed to zero" cover_jacobian(WithDiagnostics(), tu, ou)
        # The sweep tier never forms the surrogate, so it is unaffected.
        @test all(isfinite, cover_jacobian(tu, ou).b_sum)

        # `maxiter` alone asks for a convergence run rather than colliding with the default
        # round count.
        Ud = exp.(randn(MersenneTwister(31), 3, 4))
        @test balance_cover(Ud; merge = MaxMerge(), maxiter = 20_000).iters > 0
        @test balance_cover(Ud; rtol = 1.0e-12).iters > 0
        @test balance_cover(Ud).iters == 4
        @test_throws "cannot be combined" balance_cover(Ud; rounds = 2, maxiter = 10)

        # `a0` and an explicit `start` are incompatible, whichever start was named.
        @test_throws "cannot be combined with start" balance_cover(
            Ud; a0 = ones(3), start = :covariant
        )
        @test_throws "cannot be combined with start" balance_cover(Ud; a0 = ones(3), start = :equal)

        # The stall hint names a `targets` value the originating call form accepts.
        Ucyc = [
            0.0 0.5 0.0 0.0
            2.0 0.5 0.0 0.0
            0.0 0.0 0.25 4.0
            0.0 0.0 0.5 2.0
            0.5 0.0 1.0 0.5
        ]
        tcyc, ocyc = tape_function(xs -> Ucyc * xs, ones(4))
        stall = try
            balance_cover(
                tcyc, ocyc; merge = MaxMerge(), targets = :uniform, rounds = nothing,
                maxiter = 100_000
            )
            nothing
        catch e
            e
        end
        @test stall isa ErrorException
        @test occursin("targets=:flow", stall.msg)
        @test !occursin("targets=:support", stall.msg)

        # A supplied root breaks no tie, so its component reports one.
        tch, och = tape_function(chain, [0.7, -0.5, 1.1, -0.9, 0.4, 1.3])
        pot = tape_potentials(tch, och)
        @test pot.nties[1] > 1
        other = findfirst(v -> pot.live[v] && v != pot.roots[1], 1:nnodes(tch))
        @test tape_potentials(tch, och; root = other).nties == [1]

        # A factor that underflows to zero on a live row or column is not a
        # cover, so it is an error on every path and whatever `check` says.
        thin = xs -> [1.0e-150 * xs[1], xs[2]]
        tt, ot = tape_function(thin, [1.0, 1.0])
        for call in (
                () -> cover_jacobian(tt, ot; a = [1.0e200, 1.0]),
                () -> cover_jacobian(tt, ot; a = [1.0e200, 1.0], check = false),
                () -> cover_jacobian(WithDiagnostics(), tt, ot; a = [1.0e200, 1.0], check = false),
                () -> cover_jacobian_hybrid(tt, ot, 2; a = [1.0e200, 1.0], check = false),
            )
            thrown = try
                call()
                nothing
            catch e
                sprint(showerror, e)
            end
            @test thrown !== nothing
            @test occursin("underflow", thrown)
            @test occursin("b[1]", thrown) || occursin("input 1", thrown)
        end
        # The same program with weights that keep both sides in range is fine.
        @test all(>(0), cover_jacobian(tt, ot).b_sum)

        # `rounds = nothing` means a convergence run at every entry point, and
        # leaving it unsaid means the fixed budget.
        @test balance_cover(tt, ot).iters == 4
        @test cover_jacobian(tt, ot; rounds = nothing) isa JacobianCover
        @test balance_cover(tt, ot; rounds = 0).iters == 0

        # A zero budget merges the start whatever the targets, and still validates them.
        for start in (:covariant, :equal)
            z = [balance_cover(tt, ot; rounds = 0, start, targets) for targets in (:flow, :support, :uniform)]
            @test z[1].a == z[2].a == z[3].a
            @test z[1].b == z[2].b == z[3].b
            @test all(w -> w.iters == 0 && w.delta == Inf, z)
        end
        # Equal up to the power-of-two gauge, which leaves every product bit-identical.
        z = balance_cover(tt, ot; rounds = 0, start = :equal)
        @test z.a * z.b' == ones(length(ot)) * cover_jacobian(tt, ot; a = ones(length(ot))).b_sum'
        @test_throws "targets must be" balance_cover(tt, ot; rounds = 0, targets = :bogus)
        @test_throws "target r has length 1, expected 2" balance_cover(
            tt, ot; rounds = 0, targets = ([1.0], [1.0, 2.0, 3.0])
        )

        # `rtol` tunes a convergence run, so it cannot join a fixed budget.
        @test_throws "cannot be combined" balance_cover(tt, ot; rounds = 2, rtol = 1.0e-9)
        @test_throws "cannot be combined" balance_cover(tt, ot; rounds = 2, maxiter = 10)

        # A documented `a0` — zero exactly on dead outputs — is accepted under
        # uniform targets too, and a zero on a live output is refused there as well.
        rd2 = xs -> [xs[1] * xs[2], 3 * one(eltype(xs)), xs[1] + xs[2]]
        td2, od2 = tape_function(rd2, [0.7, 1.3])
        @test balance_cover(
            td2, od2; targets = :uniform, a0 = [1.0, 0.0, 1.0], rounds = 2
        ).a[2] == 0
        @test_throws "strictly positive on every live output" balance_cover(
            td2, od2; targets = :uniform, a0 = [0.0, 0.0, 1.0], rounds = 2
        )

        # The gauge is chosen from both factors at once, so a spread that fits
        # only when split between them survives.
        split = xs -> [1.0e-160 * xs[1], 1.0e160 * xs[2]]
        ts, os = tape_function(split, [1.0, 1.0])
        cs = cover_jacobian(ts, os)
        @test all(isfinite, cs.a) && all(>(0), cs.a)
        @test all(isfinite, cs.b_sum) && all(>(0), cs.b_sum)

        # A live input whose path products underflow is refused by name rather
        # than passing a numeric reach test as dead.
        @test_throws "reaches an output" marginal_weights(tt, ot, [0.0, 1.0])

        # The materialized path reports an underflowed entry, not a target that
        # the caller is told is realizable.
        under2 = xs -> [(1.0e-200 * xs[1]) * 1.0e-200, xs[1] * xs[2]]
        tu2, ou2 = tape_function(under2, [1.0, 1.0])
        @test_throws "underflowed to zero" balance_cover(tu2, ou2; merge = MaxMerge())

        # A non-finite weight is rejected where it enters, by name.
        @test_throws "must be finite" cover_jacobian(tu, ou; a = [Inf, 1.0])
        @test_throws "must be finite" cover_jacobian(tu, ou; a = [1.0, NaN])
        @test_throws "must be finite" cover_jacobian_hybrid(tu, ou, 2; a = [Inf, 1.0])
    end

    @testset "cover_jacobian takes the covariant weights" begin
        t, outidxs = tape_function(triangular, [0.6, -0.4, 0.9])
        jc = cover_jacobian(WithDiagnostics(), t, outidxs)
        @test all(jc.b_max .<= jc.b_quad .* GATE)
        @test max_coverage_ratio(jc.a, jc.b_max, jc.Utilde) <= GATE
    end

    @testset "the convenience program form" begin
        x0 = [0.6, -0.4, 0.9]
        t, outidxs = tape_function(triangular, x0)
        @test balance_cover(triangular, x0).a ≈ balance_cover(t, outidxs).a
        @test tape_potentials(triangular, x0).theta == tape_potentials(t, outidxs).theta
        @test_throws "vector-valued program" balance_cover(xs -> xs[1] * xs[2], x0)
        @test_throws "vector-valued program" tape_potentials(xs -> xs[1] * xs[2], x0)
    end

    # The sign-averaged pair written out densely on `U2`: the targeted alternation on squared
    # factors, then the unnormalized final b-step, returned as square roots.
    function dense_signaveraged_pair(U2, a0, r, c, rounds)
        inv0(x) = x > 0 ? 1 / x : 0.0
        divt(v, tg) = [tg[i] > 0 ? v[i] / tg[i] : 0.0 for i in eachindex(v, tg)]
        A = a0 .^ 2
        B = divt(U2' * inv0.(A), c)
        for _ in 1:rounds
            A = divt(U2 * inv0.(B), r)
            B = divt(U2' * inv0.(A), c)
        end
        return sqrt.(A), sqrt.(U2' * inv0.(A))
    end
    # Worst log-discrepancy between two pairs' products over the support of `U`.
    function product_gap(U, a, b, a2, b2)
        worst = 0.0
        for j in axes(U, 2), k in axes(U, 1)
            U[k, j] > 0 || continue
            worst = max(worst, abs(log(a[k] * b[j]) - log(a2[k] * b2[j])))
        end
        return worst
    end
    potential_start(t, outidxs) =
        DerivativeCovers._potential_start(t, outidxs, nothing, 0, liveinfo(t, outidxs))[1]

    @testset "sign-averaged pair: the dense iteration on the squared channel — $name" for
        (name, r, x0) in PROGRAMS
        t, outidxs = tape_function(r, x0)
        U2 = squared_surrogate(t, outidxs)
        @test U2 ≈ squared_path_sums(t, outidxs) rtol = 1.0e-14
        rt, ct = flowtargets(t, outidxs)
        li = liveinfo(t, outidxs)
        a0 = [li.liverow[k] ? 1.5^k : 0.0 for k in eachindex(li.liverow)]
        vm = SignAveragedMerge()
        for rounds in (0, 1, 4)
            bal = balance_cover(t, outidxs; merge = vm, a0, rounds, check = true)
            ad, bd = dense_signaveraged_pair(U2, a0, rt, ct, rounds)
            @test product_gap(U2, bal.a, bal.b, ad, bd) < 1.0e-12
            @test bal.iters == rounds
            balm = balance_cover(U2; merge = vm, a0, targets = (rt, ct), rounds)
            @test product_gap(U2, balm.a, balm.b, ad, bd) < 1.0e-12
            # The final b-step is the sign-averaged Gram cover for the weights `1/a²`.
            w = [ak > 0 ? 1 / ak^2 : 0.0 for ak in bal.a]
            @test bal.b ≈ DerivativeCovers.gramvariance(t, outidxs, w) rtol = 1.0e-12
            # Every live column of `U2 ./ (a.^2 .* b'.^2)` sums to one, so the squares cover `U2`.
            S = [U2[k, j] > 0 ? U2[k, j] / (bal.a[k] * bal.b[j])^2 : 0.0 for k in axes(U2, 1), j in axes(U2, 2)]
            @test all(sum(S; dims = 1)[j] ≈ 1 for j in axes(U2, 2) if li.livecol[j])
            @test max_coverage_ratio(bal.a .^ 2, bal.b .^ 2, U2) <= GATE
        end

        # Run to convergence, the normalized iterate `(a, b ./ sqrt.(γ))` meets both targets.
        bal = balance_cover(t, outidxs; merge = vm, rounds = nothing, check = true)
        @test bal.delta <= 1.0e-10
        bn2 = [ct[j] > 0 ? bal.b[j]^2 / ct[j] : 0.0 for j in eachindex(ct)]
        for k in eachindex(rt)
            rt[k] > 0 || continue
            @test sum(U2[k, j] / (bal.a[k]^2 * bn2[j]) for j in eachindex(ct) if ct[j] > 0) ≈ rt[k] rtol = 1.0e-8
        end
        # The fixed point has unique products, whatever the start or call form.
        for other in (
                balance_cover(t, outidxs; merge = vm, rounds = nothing, start = :equal),
                balance_cover(U2; merge = vm, targets = (rt, ct), rounds = nothing),
            )
            @test product_gap(U2, bal.a, bal.b, other.a, other.b) < 1.0e-8
        end
    end

    @testset "sign-averaged pair: the default pair on Ũ.^2 where every entry has one path — $name" for
        (name, r, x0) in PROGRAMS
        t, outidxs = tape_function(r, x0)
        U, U2 = jacobian_surrogate(t, outidxs), squared_surrogate(t, outidxs)
        U2 ≈ U .^ 2 || continue                      # some entry has several paths
        rt, ct = flowtargets(t, outidxs)
        # The sign-averaged pair's squares are the default pair of `Ũ.^2` started from the squared
        # start.
        # Both regauge their squared factors identically, so the agreement is entrywise.
        p0 = potential_start(t, outidxs)
        for rounds in (0, 1, 4, nothing)
            v = balance_cover(t, outidxs; merge = SignAveragedMerge(), rounds)
            d = balance_cover(U .^ 2; targets = (rt, ct), a0 = p0 .^ 2, rounds)
            @test v.a .^ 2 ≈ d.a rtol = 1.0e-10
            @test v.b .^ 2 ≈ d.b rtol = 1.0e-10
            ve = balance_cover(t, outidxs; merge = SignAveragedMerge(), start = :equal, rounds)
            de = balance_cover(U .^ 2; targets = (rt, ct), start = :equal, rounds)
            @test ve.a .^ 2 ≈ de.a rtol = 1.0e-10
            @test ve.b .^ 2 ≈ de.b rtol = 1.0e-10
        end
    end

    @testset "sign-averaged pair: covariance under node rescaling — $name" for (name, r, x0) in PROGRAMS
        t, outidxs = tape_function(r, x0)
        s = 2.0 .^ rand(MersenneTwister(13), -10:10, nnodes(t))
        t2 = rescale_tape(t, s)
        for rounds in (0, 1, 4, nothing), cgiters in (0, 5)
            v = balance_cover(t, outidxs; merge = SignAveragedMerge(), rounds, cgiters)
            v2 = balance_cover(t2, outidxs; merge = SignAveragedMerge(), rounds, cgiters)
            @test product_drift(t, outidxs, s, v.a, v.b, v2.a, v2.b) < 1.0e-10
        end
        # The matrix form's covariant start transforms with `Ũ⁽²⁾ -> C² Ũ⁽²⁾ D⁻²`.
        U2 = squared_surrogate(t, outidxs)
        c = 2.0 .^ rand(MersenneTwister(17), -10:10, length(outidxs))
        d = 2.0 .^ rand(MersenneTwister(19), -10:10, t.ninputs)
        for rounds in (0, 1, 4, nothing)
            v = balance_cover(U2; merge = SignAveragedMerge(), rounds)
            v2 = balance_cover(c .^ 2 .* U2 ./ (d .^ 2)'; merge = SignAveragedMerge(), rounds)
            @test product_gap(U2, v.a .* c, v.b ./ d, v2.a, v2.b) < 1.0e-10
        end
    end

    @testset "sign-averaged pair: errors name the squared quantities" begin
        # A path product of 1e-200 is representable; its square is not.
        tu, ou = tape_function(xs -> [1.0e-100 * (1.0e-100 * xs[1]) + 0 * xs[2], xs[2]], [1.0, 1.0])
        @test balance_cover(tu, ou).a[1] > 0
        for kw in ((;), (; rounds = nothing))
            @test_throws "squared input weight b[1]^2 is zero on an input that reaches an output" balance_cover(
                tu, ou; merge = SignAveragedMerge(), kw...
            )
        end
        @test_throws "entry of the squared channel Ũ⁽²⁾ underflowed to zero" balance_cover(
            tu, ou; merge = SignAveragedMerge(), arule = :geomean
        )
        # The final factor check names the squared factors too.
        tq, oq = tape_function(xs -> [xs[1] * xs[2], xs[2]], [1.0, 2.0])
        li = liveinfo(tq, oq)
        check! = DerivativeCovers._check_cover_factors
        @test_throws "squared output weight a[1]^2 is 0.0" check!([0.0, 1.0], [1.0, 1.0], li; signaveraged = true)
        @test_throws "squared input weight b[1]^2 is 0.0" check!([1.0, 1.0], [0.0, 1.0], li; signaveraged = true)
        @test_throws "output weight a[1] is 0.0" check!([0.0, 1.0], [1.0, 1.0], li)
    end

    @testset "sign-averaged pair: does not cover J" begin
        # Two paths reach J = 2x from x, each with partial x: Ũ = 2|x| but Ũ⁽²⁾ = 2x².
        t, outidxs = tape_function(xs -> [xs[1] * xs[1]], [3.0])
        U, U2 = jacobian_surrogate(t, outidxs), squared_surrogate(t, outidxs)
        @test U2[1, 1] ≈ 18 && U[1, 1] ≈ 6
        v = balance_cover(t, outidxs; merge = SignAveragedMerge(), rounds = nothing, check = true)
        @test v.a[1] * v.b[1] ≈ sqrt(18)
        @test max_coverage_ratio(v.a, v.b, U) ≈ 6 / sqrt(18)          # > 1: not a cover of Ũ
        @test max_coverage_ratio(v.a .^ 2, v.b .^ 2, U2) <= GATE

        # Dead rows and columns keep zero weights.
        rd = xs -> [xs[1] * xs[2], 3 * one(eltype(xs)), xs[1] + xs[2]]
        td, od = tape_function(rd, [0.7, 1.3, -0.5])
        for kw in ((;), (; rounds = nothing), (; targets = :support), (; arule = :geomean, rounds = nothing))
            vd = balance_cover(td, od; merge = SignAveragedMerge(), check = true, kw...)
            @test vd.a[2] == 0 && vd.b[3] == 0
            @test all(>(0), vd.a[[1, 3]]) && all(>(0), vd.b[1:2])
        end

        # The geometric-mean fit reads the same `Ũ⁽²⁾` on a tape as in the matrix form.
        tt, ot = tape_function(triangular, [0.6, -0.4, 0.9])
        vg = balance_cover(tt, ot; merge = SignAveragedMerge(), arule = :geomean, rounds = nothing)
        vgm = balance_cover(squared_surrogate(tt, ot); merge = SignAveragedMerge(), arule = :geomean, rounds = nothing)
        @test vg.a ≈ vgm.a rtol = 1.0e-10
        @test vg.b ≈ vgm.b rtol = 1.0e-10

        # A start whose square leaves the floating-point range is refused by name.
        tw, ow = tape_function(xs -> [1.0e300 * xs[1], 1.0e-300 * xs[2], 1.0e-300 * xs[1] * xs[2]], [1.0, 1.0])
        @test all(>(0), balance_cover(tw, ow).a)
        @test_throws "squared start is not representable" balance_cover(tw, ow; merge = SignAveragedMerge())
    end

    # The targeted alternation run to convergence on a tape, relaxed or plain.
    function converge_targets(
            t, outidxs; relax, signaveraged = false, maxiter = 100_000, a0 = potential_start(t, outidxs)
        )
        DC = DerivativeCovers
        li = liveinfo(t, outidxs)
        rt, ct = flowtargets(t, outidxs)
        a = DC._balance_start(a0, length(outidxs), Float64, li.liverow)
        signaveraged && (a = DC._squared_start(a))
        slot = signaveraged ? DC._sqslot : DC._absslot
        bnum = a -> DC._sum_merge_b(t, outidxs, a, li, slot)
        anum = b -> DC._marginal_weights(t, outidxs, b, li, slot)
        rtol = eps()^(2 // 3)
        a, iters, delta = DC._balance_targets(anum, bnum, a, rt, ct, maxiter, rtol, nothing; relax)
        b = bnum(a)
        signaveraged && (a = sqrt.(a); b = sqrt.(b))
        return (; a, b, iters, delta)
    end
    # Rows `w1[i] x[i] + w2[i] x[i+1]`, cyclically: one long cycle through the support, on which
    # the plain alternation converges slowly.
    function cyclic_bidiagonal(n, sigma, seed)
        rng = MersenneTwister(seed)
        w1, w2 = exp.(sigma .* randn(rng, n)), exp.(sigma .* randn(rng, n))
        return xs -> [w1[i] * xs[i] + w2[i] * xs[mod1(i + 1, n)] for i in 1:n], randn(rng, n)
    end

    @testset "a fixed budget runs the plain alternation, bit for bit" begin
        # Pinned from the plain iteration; a run to convergence is the only one relaxed.
        r = xs -> [xs[1] * xs[2] + xs[3], sin(xs[3]) + xs[1] * xs[2], exp(xs[1]) + xs[2] * xs[3]]
        t, outidxs = tape_function(r, [0.6, -0.4, 0.9])
        bits(v) = reinterpret.(UInt64, v)
        bal = balance_cover(t, outidxs)
        @test bits(bal.a) == [0x3ff5c5ca9ce627c4, 0x3ff144945e0db3d3, 0x3ffffa796368de49]
        @test bits(bal.b) == [0x3ff938539fd17928, 0x3ff7279b5b8a300f, 0x3ff82cf03b1c691e]
        bal = balance_cover(t, outidxs; merge = SignAveragedMerge())
        @test bits(bal.a) == [0x3fedb3afda9817ac, 0x3fe709649b7d5f97, 0x3ff8a09c9c2afcce]
        @test bits(bal.b) == [0x3ff607bf2e50884b, 0x3ff34bb64da045d9, 0x3ff67a9eec38e833]
        bal = balance_cover(t, outidxs; rounds = 7, targets = :support)
        @test bits(bal.a) == [0x3fe555a348fae680, 0x3fe1583015fb5304, 0x3ff07fb66d5d58b7]
        @test bits(bal.b) == [0x4008d6fdff7ef330, 0x400709a8e554d395, 0x40084701d0030474]
        bal = balance_cover(jacobian_surrogate(t, outidxs); rounds = 5)
        @test bits(bal.a) == [0x3fe8087d82ac9a58, 0x3fe389d2861f4957, 0x3ff295eaca0f2aee]
        @test bits(bal.b) == [0x40060cff0d6f5edc, 0x40047372778c69dd, 0x40058d262909c733]
    end

    @testset "overrelaxation: the plain fixed point, in fewer rounds" begin
        r, x0 = cyclic_bidiagonal(50, 2.0, 5)
        t, outidxs = tape_function(r, x0)
        U, U2 = jacobian_surrogate(t, outidxs), squared_surrogate(t, outidxs)
        for signaveraged in (false, true)
            plain = converge_targets(t, outidxs; relax = false, signaveraged)
            relaxed = converge_targets(t, outidxs; relax = true, signaveraged)
            @test plain.delta <= eps()^(2 // 3) && relaxed.delta <= eps()^(2 // 3)
            @test product_gap(signaveraged ? U2 : U, plain.a, plain.b, relaxed.a, relaxed.b) < 1.0e-8
            @test relaxed.iters < plain.iters / 5
            # The public entry point takes the relaxed run.
            bal = balance_cover(t, outidxs; rounds = nothing, merge = signaveraged ? SignAveragedMerge() : SumMerge())
            @test bal.iters == relaxed.iters
            @test bal.a == relaxed.a
        end

        # Rescaling the tape leaves the relaxed run's products covariant.
        s = 2.0 .^ rand(MersenneTwister(29), -10:10, nnodes(t))
        t2 = rescale_tape(t, s)
        for merge in (SumMerge(), SignAveragedMerge())
            bal = balance_cover(t, outidxs; rounds = nothing, merge)
            bal2 = balance_cover(t2, outidxs; rounds = nothing, merge)
            @test product_drift(t, outidxs, s, bal.a, bal.b, bal2.a, bal2.b) < 1.0e-8
        end

        # Relaxation amplifies roundoff by about `1/(2 - ω)`, so `ω` stays where that is below
        # `rtol`, and a tolerance near roundoff turns it off.
        ωmax = DerivativeCovers._relax_omega_max
        @test ωmax(Float64, eps()^(2 // 3)) == 1.995
        @test 1 < ωmax(Float32, eps(Float32)^(2 // 3)) < 1.995
        @test ωmax(Float64, 1.0e-15) == 1

        # A run to convergence that exhausts its budget says so, relaxed or plain.
        @test_throws "did not converge in 3 rounds" balance_cover(t, outidxs; rounds = nothing, maxiter = 3)
        @test_throws "did not converge in 3 rounds" converge_targets(t, outidxs; relax = false, maxiter = 3)
        # The residual it reports is that of the iterate reached, relaxed rounds included.
        function reported_delta(maxiter, merge)
            msg = try
                balance_cover(t, outidxs; rounds = nothing, maxiter, merge)
                ""
            catch err
                err isa ErrorException || rethrow()
                err.msg
            end
            return parse(Float64, match(r"delta = (\S+) >", msg)[1])
        end
        for merge in (SumMerge(), SignAveragedMerge())
            @test reported_delta(100, merge) < reported_delta(4, merge) / 10
        end
    end

    @testset "overrelaxation: a plateau does not set the factor" begin
        # From equal weights both maps hold the imbalance nearly constant for several rounds and
        # then contract quickly. A factor estimated on the plateau would sit near its cap, where
        # the relaxed rate `ω - 1` is far slower than the plain rounds. On the first map the
        # contraction is still improving after `_RELAX_NESTIMATE` rounds; on the second it looks
        # steady by then, and the bound on the first `ω` is what keeps the run short.
        powell = xs -> [1.0e4 * xs[1] * xs[2] - 1, exp(-xs[1]) + exp(-xs[2]) - 1.0001]
        W = [0.03 0.008 0.8; 4.0e5 400.0 0.006; 3.0 0.02 1.0]
        for (r, x0) in ((powell, [0.0, 1.0]), (xs -> W * xs, ones(3))), signaveraged in (false, true)
            t, outidxs = tape_function(r, x0)
            a0 = ones(length(outidxs))
            plain = converge_targets(t, outidxs; relax = false, signaveraged, a0)
            relaxed = converge_targets(t, outidxs; relax = true, signaveraged, a0)
            @test plain.delta <= eps()^(2 // 3) && relaxed.delta <= eps()^(2 // 3)
            @test relaxed.iters < plain.iters
            merge = signaveraged ? SignAveragedMerge() : SumMerge()
            bal = balance_cover(t, outidxs; rounds = nothing, start = :equal, merge)
            @test bal.iters == relaxed.iters
        end
    end

    @testset "overrelaxation: a window is judged against the window before it" begin
        # On the squared channel of this map the plain rounds need about 50,000 rounds. The
        # imbalance level rises with `ω`, so a window judged against the smallest imbalance of the
        # plain rounds would show no progress for many windows while the run converges, and the
        # stall rule would halve `ω` when it should rise.
        t, outidxs = tape_function(xs -> [4000.0 0.3; 9.0 9.0] * xs, ones(2))
        a0 = ones(2)
        relaxed = converge_targets(t, outidxs; relax = true, signaveraged = true, a0)
        @test relaxed.delta <= eps()^(2 // 3)
        @test relaxed.iters < 1500
        @test balance_cover(t, outidxs; rounds = nothing, start = :equal, merge = SignAveragedMerge()).iters == relaxed.iters

        # The first window after `ω` changes is judged against one at the lower `ω`, whose lower
        # level makes it read as slow. Without a limit on one window's raise this run takes more
        # than a thousand rounds.
        W = [100.0 3.0e7 0.0; 0.0 100.0 0.7; 200.0 100.0 2.0]
        r = xs -> [sum(W[i, j] * exp(-xs[j]) for j in 1:3 if W[i, j] != 0) + xs[i] * xs[mod1(i + 1, 3)] for i in 1:3]
        t, outidxs = tape_function(r, [0.5, 2.0, 1.0])
        plain = converge_targets(t, outidxs; relax = false, signaveraged = true)
        relaxed = converge_targets(t, outidxs; relax = true, signaveraged = true)
        @test relaxed.delta <= eps()^(2 // 3)
        @test relaxed.iters < 2 * plain.iters
    end

    @testset "overrelaxation: a step that would overshoot is shortened" begin
        step! = DerivativeCovers._relaxed_step!
        # From `x = 1000` toward `F = 1`, `ω = 1.99` in log space would land near 1e-3; the
        # safeguard halves `ω - 1` until the step's excess is no larger than the plain one allows.
        x = [1.0e3]
        @test step!(x, [1.0], [1.0], 1.99) ≈ 0.999
        @test 1.0e3^(1 - 1.99) < x[1] < 1
        @test x[1] ≈ 1.0e3^(-0.2475)
        # A small step is taken in full, and a dead entry is left alone.
        x = [1.01, 5.0]
        step!(x, [1.0, 2.0], [1.0, 0.0], 1.5)
        @test x ≈ [1.01^(-0.5), 5.0]
    end

    # `outidxs` names node numbers, so its axes are the caller's; the results are indexed by
    # output position. `x0` lays out the tape's inputs, which are one-based.
    @testset "generic axes" begin
        offsetcopy(A) = OffsetArray(A, -2)
        x0 = [0.6, -0.4, 0.9]
        t, outidxs = tape_function(triangular, x0)
        oidx = offsetcopy(collect(outidxs))
        voidx = view(collect(outidxs), :)

        for idx in (oidx, voidx)
            rt, ct = flowtargets(t, idx)
            rtref, ctref = flowtargets(t, outidxs)
            @test rt == rtref && ct == ctref
            @test axes(rt) == (Base.OneTo(3),)
            @test supportcounts(t, idx) == supportcounts(t, outidxs)
            @test tape_potentials(t, idx).theta == tape_potentials(t, outidxs).theta
            cw, cwref = balance_cover(t, idx), balance_cover(t, outidxs)
            @test cw.a == cwref.a
            @test axes(cw.a) == (Base.OneTo(3),)
            @test cw.b == cwref.b
            @test balance_cover(t, idx).a == balance_cover(t, outidxs).a
            @test balance_cover(t, idx; start = :equal).a ==
                balance_cover(t, outidxs; start = :equal).a
            vw, vwref = balance_cover(t, idx; merge = SignAveragedMerge()),
                balance_cover(t, outidxs; merge = SignAveragedMerge())
            @test vw.a == vwref.a && vw.b == vwref.b
            @test axes(vw.a) == (Base.OneTo(3),)
            @test squared_surrogate(t, idx) == squared_surrogate(t, outidxs)
        end

        @test_throws "offset arrays are not supported" balance_cover(
            triangular,
            offsetcopy(x0)
        )
        @test_throws "offset arrays are not supported" tape_potentials(triangular, offsetcopy(x0))
        @test_throws "offset arrays are not supported" balance_cover(
            jacobian_surrogate(t, outidxs);
            targets = (offsetcopy(ones(3)), ones(3))
        )
    end

end
