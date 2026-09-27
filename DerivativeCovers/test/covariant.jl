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
        @test jacobian_ratio(U, bal.a, bal.b) <= GATE
        @test jacobian_ratio(J, bal.a, bal.b) <= GATE

        # The matrix-free tape path and the explicit matrix path reach the same pair.
        rt, ct = flowtargets(t, outidxs)
        balm = balance_cover(U; targets = (rt, ct), rounds = nothing, a0 = bal.a)
        @test bal.a ≈ balm.a rtol = 1.0e-8
        @test bal.b ≈ balm.b rtol = 1.0e-8
        # `:support` is the matrix form's built-in pattern-feasible choice.
        bals = balance_cover(U; targets = :support, rounds = nothing)
        @test jacobian_ratio(U, bals.a, bals.b) <= GATE

        # Non-sum merges take the same normalized alternation and only change the final b-step,
        # so their products are no larger than the sum merge's. (The comparison is on products:
        # the delegating path starts elsewhere and so fixes a different gauge.)
        for merge in (QuadMerge(), MaxMerge())
            balq = balance_cover(t, outidxs; targets = :flow, rounds = nothing, merge)
            @test jacobian_ratio(U, balq.a, balq.b) <= GATE
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
        @test jacobian_ratio(U, bal.a, bal.b) <= GATE
        @test jacobian_ratio(J, bal.a, bal.b) <= GATE
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
        @test jacobian_ratio(jacobian_surrogate(td, od), bald.a, bald.b) <= GATE
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
            @test jacobian_ratio(U, cw.a, cw.b) <= GATE
            @test jacobian_ratio(J, cw.a, cw.b) <= GATE
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
        @test jacobian_ratio(U, cw.a, cw.b) <= GATE
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
            @test jacobian_ratio(U, c.a, c.b_sum) <= GATE
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
            @test jacobian_ratio(U, c.a, hyb) <= GATE
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
            @test jacobian_ratio(Ud, ad, bd) <= GATE
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
                @test jacobian_ratio(U, x.a, x.b_sum) <= GATE
            end
            @test all(isfinite, d.b_quad) && all(isfinite, d.b_max)
            @test jacobian_ratio(U, d.a, d.b_max) <= GATE
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
        @test jacobian_ratio(jacobian_surrogate(t, outidxs), c.a, c.b_sum) <= GATE
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
        @test jacobian_ratio(jc.Utilde, jc.a, jc.b_max) <= GATE
    end

    @testset "the convenience program form" begin
        x0 = [0.6, -0.4, 0.9]
        t, outidxs = tape_function(triangular, x0)
        @test balance_cover(triangular, x0).a ≈ balance_cover(t, outidxs).a
        @test tape_potentials(triangular, x0).theta == tape_potentials(t, outidxs).theta
        @test_throws "vector-valued program" balance_cover(xs -> xs[1] * xs[2], x0)
        @test_throws "vector-valued program" tape_potentials(xs -> xs[1] * xs[2], x0)
    end

    # The variance pair written out densely on `U2`: the targeted alternation on squared factors,
    # then the unnormalized final b-step, returned as square roots.
    function dense_variance_pair(U2, a0, r, c, rounds)
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

    @testset "variance pair: the dense iteration on the squared channel — $name" for
        (name, r, x0) in PROGRAMS
        t, outidxs = tape_function(r, x0)
        U2 = variance_surrogate(t, outidxs)
        @test U2 ≈ squared_path_sums(t, outidxs) rtol = 1.0e-14
        rt, ct = flowtargets(t, outidxs)
        li = liveinfo(t, outidxs)
        a0 = [li.liverow[k] ? 1.5^k : 0.0 for k in eachindex(li.liverow)]
        vm = VarianceMerge()
        for rounds in (0, 1, 4)
            bal = balance_cover(t, outidxs; merge = vm, a0, rounds, check = true)
            ad, bd = dense_variance_pair(U2, a0, rt, ct, rounds)
            @test product_gap(U2, bal.a, bal.b, ad, bd) < 1.0e-12
            @test bal.iters == rounds
            balm = balance_cover(U2; merge = vm, a0, targets = (rt, ct), rounds)
            @test product_gap(U2, balm.a, balm.b, ad, bd) < 1.0e-12
            # The final b-step is the Gram variance scale for the weights `1/a²`.
            w = [ak > 0 ? 1 / ak^2 : 0.0 for ak in bal.a]
            @test bal.b ≈ DerivativeCovers.gramvariance(t, outidxs, w) rtol = 1.0e-12
            # Every live column of `U2 ./ (a.^2 .* b'.^2)` sums to one, so the squares cover `U2`.
            S = [U2[k, j] > 0 ? U2[k, j] / (bal.a[k] * bal.b[j])^2 : 0.0 for k in axes(U2, 1), j in axes(U2, 2)]
            @test all(sum(S; dims = 1)[j] ≈ 1 for j in axes(U2, 2) if li.livecol[j])
            @test jacobian_ratio(U2, bal.a .^ 2, bal.b .^ 2) <= GATE
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

    @testset "variance pair: the default pair on Ũ.^2 where every entry has one path — $name" for
        (name, r, x0) in PROGRAMS
        t, outidxs = tape_function(r, x0)
        U, U2 = jacobian_surrogate(t, outidxs), variance_surrogate(t, outidxs)
        U2 ≈ U .^ 2 || continue                      # some entry has several paths
        rt, ct = flowtargets(t, outidxs)
        # The variance pair's squares are the default pair of `Ũ.^2` started from the squared start.
        # Both regauge their squared factors identically, so the agreement is entrywise.
        p0 = potential_start(t, outidxs)
        for rounds in (0, 1, 4, nothing)
            v = balance_cover(t, outidxs; merge = VarianceMerge(), rounds)
            d = balance_cover(U .^ 2; targets = (rt, ct), a0 = p0 .^ 2, rounds)
            @test v.a .^ 2 ≈ d.a rtol = 1.0e-10
            @test v.b .^ 2 ≈ d.b rtol = 1.0e-10
            ve = balance_cover(t, outidxs; merge = VarianceMerge(), start = :equal, rounds)
            de = balance_cover(U .^ 2; targets = (rt, ct), start = :equal, rounds)
            @test ve.a .^ 2 ≈ de.a rtol = 1.0e-10
            @test ve.b .^ 2 ≈ de.b rtol = 1.0e-10
        end
    end

    @testset "variance pair: covariance under node rescaling — $name" for (name, r, x0) in PROGRAMS
        t, outidxs = tape_function(r, x0)
        s = 2.0 .^ rand(MersenneTwister(13), -10:10, nnodes(t))
        t2 = rescale_tape(t, s)
        for rounds in (0, 1, 4, nothing), cgiters in (0, 5)
            v = balance_cover(t, outidxs; merge = VarianceMerge(), rounds, cgiters)
            v2 = balance_cover(t2, outidxs; merge = VarianceMerge(), rounds, cgiters)
            @test product_drift(t, outidxs, s, v.a, v.b, v2.a, v2.b) < 1.0e-10
        end
        # The matrix form's covariant start transforms with `Ũ⁽²⁾ -> C² Ũ⁽²⁾ D⁻²`.
        U2 = variance_surrogate(t, outidxs)
        c = 2.0 .^ rand(MersenneTwister(17), -10:10, length(outidxs))
        d = 2.0 .^ rand(MersenneTwister(19), -10:10, t.ninputs)
        for rounds in (0, 1, 4, nothing)
            v = balance_cover(U2; merge = VarianceMerge(), rounds)
            v2 = balance_cover(c .^ 2 .* U2 ./ (d .^ 2)'; merge = VarianceMerge(), rounds)
            @test product_gap(U2, v.a .* c, v.b ./ d, v2.a, v2.b) < 1.0e-10
        end
    end

    @testset "variance pair: a scale, not a cover" begin
        # Two paths reach J = 2x from x, each with partial x: Ũ = 2|x| but Ũ⁽²⁾ = 2x².
        t, outidxs = tape_function(xs -> [xs[1] * xs[1]], [3.0])
        U, U2 = jacobian_surrogate(t, outidxs), variance_surrogate(t, outidxs)
        @test U2[1, 1] ≈ 18 && U[1, 1] ≈ 6
        v = balance_cover(t, outidxs; merge = VarianceMerge(), rounds = nothing, check = true)
        @test v.a[1] * v.b[1] ≈ sqrt(18)
        @test jacobian_ratio(U, v.a, v.b) ≈ 6 / sqrt(18)          # > 1: not a cover of Ũ
        @test jacobian_ratio(U2, v.a .^ 2, v.b .^ 2) <= GATE

        # Dead rows and columns keep zero weights.
        rd = xs -> [xs[1] * xs[2], 3 * one(eltype(xs)), xs[1] + xs[2]]
        td, od = tape_function(rd, [0.7, 1.3, -0.5])
        for kw in ((;), (; rounds = nothing), (; targets = :support), (; arule = :geomean, rounds = nothing))
            vd = balance_cover(td, od; merge = VarianceMerge(), check = true, kw...)
            @test vd.a[2] == 0 && vd.b[3] == 0
            @test all(>(0), vd.a[[1, 3]]) && all(>(0), vd.b[1:2])
        end

        # The geometric-mean fit reads the same `Ũ⁽²⁾` on a tape as in the matrix form.
        tt, ot = tape_function(triangular, [0.6, -0.4, 0.9])
        vg = balance_cover(tt, ot; merge = VarianceMerge(), arule = :geomean, rounds = nothing)
        vgm = balance_cover(variance_surrogate(tt, ot); merge = VarianceMerge(), arule = :geomean, rounds = nothing)
        @test vg.a ≈ vgm.a rtol = 1.0e-10
        @test vg.b ≈ vgm.b rtol = 1.0e-10

        # A start whose square leaves the floating-point range is refused by name.
        tw, ow = tape_function(xs -> [1.0e300 * xs[1], 1.0e-300 * xs[2], 1.0e-300 * xs[1] * xs[2]], [1.0, 1.0])
        @test all(>(0), balance_cover(tw, ow).a)
        @test_throws "squared start is not representable" balance_cover(tw, ow; merge = VarianceMerge())
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
            vw, vwref = balance_cover(t, idx; merge = VarianceMerge()),
                balance_cover(t, outidxs; merge = VarianceMerge())
            @test vw.a == vwref.a && vw.b == vwref.b
            @test axes(vw.a) == (Base.OneTo(3),)
            @test variance_surrogate(t, idx) == variance_surrogate(t, outidxs)
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
