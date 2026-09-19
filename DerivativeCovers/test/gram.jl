# Tape-based Gram covers must match MatrixCovers without materializing the surrogate. Formed
# J'WJ matrices provide ground truth.

@testset "gram.jl" begin

    # Outputs 1,2 depend on inputs 1,2; output 3 on inputs 3,4. Two support components.
    prog = xs -> [sin(xs[1]) * xs[2] + xs[1]^2, xs[2] * exp(xs[1]), xs[3] * xs[4] + cos(xs[4])]
    x0 = [0.3, 0.7, 1.1, -0.4]
    t, outidxs = tape_function(prog, x0)
    U = DerivativeCovers.jacobian_surrogate(t, outidxs)
    jc = cover_jacobian(WithDiagnostics(), t, outidxs)
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
        for (s, G) in (
                (gramcover(a, b, t, outidxs), J'J),
                (gramcover(a, b, t, outidxs, w), J' * Diagonal(w) * J),
                (gramcover(a, b, t, outidxs, W), J' * W * J),
            )
            @test all(>=(0), s)
            @test all(abs(G[j, k]) <= s[j] * s[k] * GATE for j in axes(G, 1), k in axes(G, 2))
        end
    end

    @testset "per-component accumulation beats the global norm(a)*b bound" begin
        s = gramcover(a, b, t, outidxs)
        @test all(s .<= norm(a) .* b .* GATE)
        # With more than one component no component carries every row, so each supported column
        # is strictly below the global bound that sums over all of them.
        @test all(s[j] < norm(a) * b[j] * (1 - 1.0e-12) for j in 1:t.ninputs if b[j] > 0)
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
        # Cross-component constraints cannot shrink a scale but need not enlarge one.
        @test all(smix .>= ssep .* (1 - 1.0e-12))
        G = J' * Wmix * J
        @test all(abs(G[j, k]) <= smix[j] * smix[k] * GATE for j in axes(G, 1), k in axes(G, 2))

        # Strong coupling can force a scale to grow.
        Wgrow = [1.0 0.0 100.0; 0.0 1.0 0.0; 100.0 0.0 1.0]
        Ggrow = J' * Wgrow * J
        @test any(abs(Ggrow[j, k]) > ssep[j] * ssep[k] for j in axes(Ggrow, 1), k in axes(Ggrow, 2))
        sgrow = gramcover(a, b, t, outidxs, Wgrow)
        @test any(sgrow .> ssep)
        @test all(
            abs(Ggrow[j, k]) <= sgrow[j] * sgrow[k] * GATE
                for j in axes(Ggrow, 1), k in axes(Ggrow, 2)
        )
    end

    @testset "the degenerate keyword reaches MatrixCovers" begin
        # A loopless bipartite support graph has no gauge-invariant cover.
        Wdeg = [0.0 0.0 1.0; 0.0 0.0 0.0; 1.0 0.0 0.0]
        @test_throws "no gauge-invariant cover exists" gramcover(a, b, t, outidxs, Wdeg)
        s = gramcover(a, b, t, outidxs, Wdeg; degenerate = :uniform)
        @test s == gramcover(a, b, U, Wdeg; degenerate = :uniform)
        sm = fill(NaN, t.ninputs)
        @test_throws "no gauge-invariant cover exists" gramcover!(sm, a, b, t, outidxs, Wdeg)
        @test gramcover!(sm, a, b, t, outidxs, Wdeg; degenerate = :uniform) === sm
        @test sm == s
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
        jcd = cover_jacobian(WithDiagnostics(), td, od)
        s = gramcover(jcd.a, jcd.b_max, td, od)
        @test s[4] == 0
        @test all(>(0), s[1:3])
    end
end
