# Tests for the tape route to a matrix's bipartite support components. The claim is that the O(L)
# sweep off the tape reproduces MatrixCovers' O(mn) reading of the materialized surrogate exactly,
# down to component ids, and that inputs and outputs no live path connects fall out as component 0.

@testset "support.jl" begin

    # Outputs 1,2 depend on inputs 1,2; output 3 on input 3. Input 4 is untouched.
    prog = xs -> [sin(xs[1]) * xs[2] + xs[1]^2, xs[2] * exp(xs[1]), cos(xs[3])]
    t, outidxs = tape_function(prog, [0.3, 0.7, 1.1, -0.4])
    U = DerivativeCovers.jacobian_surrogate(t, outidxs)

    @testset "matches the matrix reading exactly" begin
        sc = support_components(t, outidxs)
        scm = support_components(U)
        @test sc isa SupportComponents
        @test ncomponents(sc) == ncomponents(scm) == 2
        @test [rowcomponent(sc, k) for k in 1:length(outidxs)] ==
              [rowcomponent(scm, k) for k in 1:length(outidxs)]
        @test [colcomponent(sc, j) for j in 1:t.ninputs] ==
              [colcomponent(scm, j) for j in 1:t.ninputs]
    end

    @testset "position axes, one per output and input" begin
        sc = support_components(t, outidxs)
        @test sc.rowax == Base.OneTo(length(outidxs))
        @test sc.colax == Base.OneTo(t.ninputs)
    end

    @testset "an input no output reaches is component 0" begin
        sc = support_components(t, outidxs)
        @test colcomponent(sc, 4) == 0                       # x[4] is untouched
        @test all(!=(0), (colcomponent(sc, j) for j in 1:3))
    end

    @testset "a repeated output shares its row's component" begin
        sc = support_components(t, [outidxs; outidxs[1]])
        @test rowcomponent(sc, length(outidxs) + 1) == rowcomponent(sc, 1)
    end

    @testset "an edge whose partial vanishes carries no support" begin
        # Support is read off the recorded first partials, not off the tape's shape: at xs[1] = 0
        # the squaring node's partial is exactly zero, so output 1 depends on no input even though
        # a tape edge joins them. The matrix reading of the surrogate agrees.
        tz, oz = tape_function(xs -> [xs[1]^2, xs[2]], [0.0, 1.0])
        Uz = DerivativeCovers.jacobian_surrogate(tz, oz)
        @test Uz[1, :] == zeros(2)
        sc = support_components(tz, oz)
        scm = support_components(Uz)
        @test ncomponents(sc) == ncomponents(scm) == 1
        @test rowcomponent(sc, 1) == rowcomponent(scm, 1) == 0
        @test [rowcomponent(sc, k) for k in 1:2] == [rowcomponent(scm, k) for k in 1:2]
        @test [colcomponent(sc, j) for j in 1:2] == [colcomponent(scm, j) for j in 1:2]
    end

    @testset "out-of-range output fails fast" begin
        @test_throws BoundsError support_components(t, [nnodes(t) + 1])
    end
end
