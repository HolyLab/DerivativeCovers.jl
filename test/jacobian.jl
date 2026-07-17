# Tests for the asymmetric Jacobian covers of jacobian.jl. Residual maps are synthetic and small;
# ground truth is ForwardDiff's exact Jacobian. The claims specific to this construction are that
# every merge is a valid cover of the Jacobian for any positive output weights `a`, that the merges
# fall in the chain b_max ≤ b_quad ≤ b_sum with the group-max hybrid interpolating between the
# endpoints, and that the surrogate Ũ bounds |J| entrywise.

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
        jc = cover_jacobian(r, x0)
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
        @test all(jc.Utilde .>= abs.(J) .- 1e-12)
        @test jc.b_max ≈ vec(maximum(jc.Utilde ./ jc.a; dims=1))

        # Group-max hybrid: one group is the sum merge, one output per group the exact merge, and
        # an intermediate partition is bracketed by and covers between them.
        t, outidxs = tape_function(r, x0)
        @test cover_jacobian_hybrid(t, outidxs, 1) ≈ jc.b_sum
        @test cover_jacobian_hybrid(t, outidxs, jc.m) ≈ jc.b_max
        hyb = cover_jacobian_hybrid(t, outidxs, 2)
        @test jacobian_ratio(J, jc.a, hyb) <= GATE
        @test all(jc.b_max .<= hyb .* GATE)
        @test all(hyb .<= jc.b_sum .* GATE)
    end

    @testset "hand-checkable linear residuals" begin
        # r = [x₁ + 2x₂, 3x₁], J = [1 2; 3 0]. With a = ones the minimal b is the column max of |J|.
        r = xs -> [xs[1] + 2 * xs[2], 3 * xs[1]]
        x0 = [0.5, -0.3]
        jc = cover_jacobian(r, x0)
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
        jc = cover_jacobian(r, x0; a)
        for b in (jc.b_sum, jc.b_quad, jc.b_max)
            @test jacobian_ratio(J, a, b) <= GATE
        end
        # Reweighting a rescales the surrogate rows but not the covering property.
        @test jc.b_max ≈ vec(maximum(jc.Utilde ./ a; dims=1))

        @test_throws DimensionMismatch cover_jacobian(r, x0; a=[1.0, 1.0])
        @test_throws "strictly positive" cover_jacobian(r, x0; a=[1.0, 0.0, 1.0])
    end

    @testset "the b-side is covariant under input rescaling" begin
        # x → D*x sends J → J*inv(D), so |J| and the surrogate lose a factor d[j] on input j and
        # b → b ./ d, with the output weights a untouched. This is the property equilibration rests on.
        r = xs -> [xs[1] * xs[2], sin(xs[3]) + xs[1], exp(xs[2])]
        x0 = [0.6, -0.4, 0.9]
        d = [1e-3, 1e2, 5.0]
        jc = cover_jacobian(r, x0)
        jcd = cover_jacobian(xs -> r(xs ./ d), d .* x0)
        @test jcd.b_sum ≈ jc.b_sum ./ d rtol=1e-10
        @test jcd.b_quad ≈ jc.b_quad ./ d rtol=1e-10
        @test jcd.b_max ≈ jc.b_max ./ d rtol=1e-10
    end

    @testset "duplicated and input-node outputs cover correctly" begin
        r = xs -> (y = xs[1] * xs[2]; [y, y, xs[3]])   # one node in two output slots, plus an input
        x0 = [0.7, -0.5, 1.2]
        J = ForwardDiff.jacobian(r, x0)
        jc = cover_jacobian(r, x0)
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
        jc = cover_jacobian(xs -> Mr * xs, zeros(4); check=false)   # some valid rank-1 cover of |J|=Mr
        # Mr*xs has Jacobian Mr (all entries positive), so its b_max cover has excess ≥ rep.
        @test rep <= excess(jc.a, jc.b_max) * (1 + 1e-8)

        @test_throws "nonnegative" jacobian_representability([1.0 -2.0; 3.0 4.0])
    end

    @testset "cover_jacobian rejects a scalar program" begin
        @test_throws "vector-valued program" cover_jacobian(xs -> xs[1] * xs[2], [1.0, 2.0])
    end

end
