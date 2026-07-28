# Tests for the tape forms of the Gram-matrix cover. The construction, its gauge invariance, and
# the roundoff guarantee are MatrixCovers'; what is DC's is that the O(L) tape form — which never
# forms the surrogate — reproduces MatrixCovers' matrix form bit for bit, and that the pair it is
# fed genuinely covers the Jacobian the program has. Ground truth is the formed J'*W*J.

@testset "gram.jl" begin

    # Outputs 1,2 depend on inputs 1,2; output 3 on inputs 3,4. Two support components.
    prog = xs -> [sin(xs[1]) * xs[2] + xs[1]^2, xs[2] * exp(xs[1]), xs[3] * xs[4] + cos(xs[4])]
    x0 = [0.3, 0.7, 1.1, -0.4]
    t, outidxs = tape_function(prog, x0)
    U = DerivativeCovers.jacobian_surrogate(t, outidxs)
    jc = cover_jacobian(t, outidxs)
    a, b = jc.a, jc.b_max
    J = ForwardDiff.jacobian(prog, x0)

    @testset "the tape form matches the matrix form" begin
        w = [0.5, 2.0, 1.5]
        W = randn(MersenneTwister(3), 3, 3)                  # indefinite and unsymmetric on purpose
        @test gramcover(a, b, t, outidxs) == gramcover(a, b, U)
        @test gramcover(a, b, t, outidxs, w) == gramcover(a, b, U, w)
        @test gramcover(a, b, t, outidxs, Diagonal(w)) == gramcover(a, b, U, w)
        @test gramcover(a, b, t, outidxs, W) == gramcover(a, b, U, W)
    end

    @testset "covers J'*W*J entrywise" begin
        w = [0.5, 2.0, 1.5]
        W = randn(MersenneTwister(3), 3, 3)
        for (s, G) in ((gramcover(a, b, t, outidxs), J'J),
                       (gramcover(a, b, t, outidxs, w), J' * Diagonal(w) * J),
                       (gramcover(a, b, t, outidxs, W), J' * W * J))
            @test all(>=(0), s)
            @test all(abs(G[j, k]) <= s[j] * s[k] * GATE for j in axes(G, 1), k in axes(G, 2))
        end
    end

    @testset "per-component accumulation beats the global norm(a)*b bound" begin
        s = gramcover(a, b, t, outidxs)
        @test all(s .<= norm(a) .* b .* GATE)
        # With more than one component no component carries every row, so each supported column
        # is strictly below the global bound that sums over all of them.
        @test all(s[j] < norm(a) * b[j] * (1 - 1e-12) for j in 1:t.ninputs if b[j] > 0)
    end

    @testset "invariant under the per-component gauge" begin
        sc = support_components(t, outidxs)
        s = gramcover(a, b, t, outidxs)
        for gam in ([2.0, 7.0], [0.25, 3.0])
            a2 = [a[k] * gam[rowcomponent(sc, k)] for k in 1:length(outidxs)]
            b2 = [b[j] / gam[colcomponent(sc, j)] for j in 1:t.ninputs]
            @test gramcover(a2, b2, t, outidxs) ≈ s
        end
    end

    @testset "off-diagonal W merges components" begin
        Wsep = Matrix(1.0I, 3, 3)
        Wmix = [1.0 0.0 1.0; 0.0 1.0 0.0; 1.0 0.0 1.0]       # couples output 1 to output 3
        ssep = gramcover(a, b, t, outidxs, Wsep)
        smix = gramcover(a, b, t, outidxs, Wmix)
        @test ssep ≈ gramcover(a, b, t, outidxs)             # identity W reproduces the J'J cover
        @test all(smix .>= ssep .* (1 - 1e-12))              # merging can only enlarge
        @test any(smix .> ssep)
        G = J' * Wmix * J
        @test all(abs(G[j, k]) <= smix[j] * smix[k] * GATE for j in axes(G, 1), k in axes(G, 2))
    end

    @testset "mutating form writes in place" begin
        s = fill(NaN, t.ninputs)
        @test gramcover!(s, a, b, t, outidxs) === s
        @test s == gramcover(a, b, t, outidxs)
    end

    @testset "a dead input gets a zero scale" begin
        # x[4] is untouched, so its Gram column vanishes and its scale is exactly zero.
        pd = xs -> [sin(xs[1]) * xs[2], xs[2] * exp(xs[1]), cos(xs[3])]
        td, od = tape_function(pd, x0)
        jcd = cover_jacobian(td, od)
        s = gramcover(jcd.a, jcd.b_max, td, od)
        @test s[4] == 0
        @test all(>(0), s[1:3])
    end
end
