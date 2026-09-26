# The tape Gram cover is `Utilde' * v` for the weight's seed vector `v`, computed by one reverse
# sweep. Formed `J' * W * J` matrices provide ground truth for the covering property.

@testset "gram.jl" begin

    # Outputs 1,2 depend on inputs 1,2; output 3 on inputs 3,4.
    prog = xs -> [sin(xs[1]) * xs[2] + xs[1]^2, xs[2] * exp(xs[1]), xs[3] * xs[4] + cos(xs[4])]
    x0 = [0.3, 0.7, 1.1, -0.4]
    t, outidxs = tape_function(prog, x0)
    U = cover_jacobian(WithDiagnostics(), t, outidxs).Utilde
    J = ForwardDiff.jacobian(prog, x0)
    m, n = size(J)
    rtol = 1.0e-13
    covers(G, s) = all(abs(G[i, j]) <= s[i] * s[j] * GATE for i in axes(G, 1), j in axes(G, 2))

    w = [0.5, -2.0, 0.0]                                     # mixed signs and a zero
    W = randn(MersenneTwister(3), m, m)                      # indefinite and unsymmetric
    vuser = fill(sqrt(maximum(abs, W)), m)

    @testset "equals Utilde' * v" begin
        @test isapprox(gramcover(t, outidxs), U' * ones(m); rtol)
        @test isapprox(gramcover(t, outidxs, w), U' * sqrt.(abs.(w)); rtol)
        @test gramcover(t, outidxs, Diagonal(w)) == gramcover(t, outidxs, w)
        vdef = symcover(max.(abs.(W), abs.(transpose(W))))
        @test isapprox(gramcover(t, outidxs, W), U' * vdef; rtol)
        @test isapprox(gramcover(t, outidxs, W; v = vuser), U' * vuser; rtol)
    end

    @testset "covers J' * W * J entrywise" begin
        for (s, G) in (
                (gramcover(t, outidxs), J'J),
                (gramcover(t, outidxs, w), J' * Diagonal(w) * J),
                (gramcover(t, outidxs, W), J' * W * J),
                (gramcover(t, outidxs, W; v = vuser), J' * W * J),
            )
            @test all(>=(0), s)
            @test covers(G, s)
        end
    end

    @testset "brackets the per-output quadrature h_gn" begin
        rng = MersenneTwister(5)
        mr, nr = 8, 4
        A = randn(rng, mr, nr)
        A[1:3, 4] .= 0                                       # uneven output multiplicities
        rmap(x) = [tanh(sum(A[k, j] * x[j] for j in 1:nr)) for k in 1:mr]
        xr = 0.5 .* randn(rng, nr)
        nls = cover_nls(WithDiagnostics(), rmap, xr)
        tr, or = tape_function(rmap, xr)
        c = gramcover(tr, or)
        nj = [count(>(0), view(nls.Utilde, :, j)) for j in 1:nr]
        @test all(nls.h_gn .<= c .* GATE)
        @test all(c .<= sqrt.(nj) .* nls.h_gn .* GATE)
        @test nj[4] == mr - 3
        Jr = ForwardDiff.jacobian(rmap, xr)
        @test covers(Jr'Jr, c)
    end

    @testset "input covariance" begin
        d = [2.0, 0.1, 5.0, 0.3]
        progd = y -> prog(y ./ d)                            # x = y ./ d, so J -> J * inv(D)
        td, od = tape_function(progd, d .* x0)
        @test isapprox(gramcover(td, od), gramcover(t, outidxs) ./ d; rtol)
        @test isapprox(gramcover(td, od, w), gramcover(t, outidxs, w) ./ d; rtol)
    end

    @testset "output covariance" begin
        C = [3.0, 0.02, 7.0]
        tc, oc = tape_function(xs -> C .* prog(xs), x0)      # J -> C * J
        wpos = [0.5, 2.0, 1.5]
        @test isapprox(gramcover(tc, oc, wpos ./ C .^ 2), gramcover(t, outidxs, wpos); rtol)
        @test isapprox(gramcover(tc, oc, 1 ./ C .^ 2), gramcover(t, outidxs); rtol)
    end

    @testset "a zero weight drops the output" begin
        @test gramcover(t, outidxs, [1.0, 0.0, 1.0]) == gramcover(t, outidxs[[1, 3]])
        @test gramcover(t, outidxs, [0.0, 0.0, 0.0]) == zeros(n)
    end

    @testset "a repeated output adds its seeds" begin
        @test gramcover(t, [outidxs[1], outidxs[1], outidxs[2]]) ==
            gramcover(t, outidxs[1:2], [4.0, 1.0])
    end

    @testset "outputs indexed by an offset axis" begin
        oo = OffsetArray(outidxs, 0:(m - 1))
        wo = OffsetArray(w, 0:(m - 1))
        @test gramcover(t, oo) == gramcover(t, outidxs)
        @test gramcover(t, oo, wo) == gramcover(t, outidxs, w)
        Wo = OffsetArray(W, 0:(m - 1), 0:(m - 1))
        @test gramcover(t, oo, Wo; v = OffsetArray(vuser, 0:(m - 1))) ==
            gramcover(t, outidxs, W; v = vuser)
        @test_throws "w has indices" gramcover(t, oo, w)
    end

    @testset "mutating form writes in place" begin
        for args in ((), (w,), (Diagonal(w),), (W,))
            s = fill(NaN, n)
            @test gramcover!(s, t, outidxs, args...) === s
            @test s == gramcover(t, outidxs, args...)
        end
        s = fill(NaN, n)
        @test gramcover!(s, t, outidxs, W; v = vuser) === s
        @test s == gramcover(t, outidxs, W; v = vuser)
    end

    @testset "argument checks" begin
        @test_throws "w has indices" gramcover(t, outidxs, [1.0, 2.0])
        @test_throws "weights must be finite" gramcover(t, outidxs, [1.0, NaN, 2.0])
        @test_throws "W has axes" gramcover(t, outidxs, ones(2, 2))
        @test_throws "W has axes" gramcover(t, outidxs, ones(3, 2); v = ones(3))
        @test_throws "v has indices" gramcover(t, outidxs, W; v = ones(2))
        @test_throws "finite and nonnegative" gramcover(t, outidxs, W; v = [1.0, -1.0, 1.0] .* vuser)
        @test_throws "finite and nonnegative" gramcover(t, outidxs, W; v = [Inf, 1.0, 1.0])
        @test_throws "v must satisfy" gramcover(t, outidxs, W; v = 0.5 .* vuser)
        @test_throws "s has indices" gramcover!(zeros(n - 1), t, outidxs)
        @test_throws "s has indices" gramcover!(OffsetArray(zeros(n), 0:(n - 1)), t, outidxs, w)
    end

    @testset "a dead input gets a zero scale" begin
        # x[4] is untouched, so its Gram column vanishes and its scale is exactly zero.
        pd = xs -> [sin(xs[1]) * xs[2], xs[2] * exp(xs[1]), cos(xs[3])]
        td, od = tape_function(pd, x0)
        s = gramcover(td, od)
        @test s[4] == 0
        @test all(>(0), s[1:3])
    end
end
