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

# `gramvariance` is `sqrt.(∑_k abs(w[k]) * U2[k, :])`, with `U2` the sum over paths of squared
# path products. The dense forward recurrence over the tape builds `U2` for comparison.
function squared_path_sums(t, outidxs)
    N, n = nnodes(t), t.ninputs
    U2 = zeros(N, n)
    for j in 1:n
        U2[j, j] = 1
    end
    for k in (n + 1):N, (i, p) in enumerate(t.parents[k])
        U2[k, :] .+= abs2(DerivativeCovers.magnitude(t.ds[k][i])) .* U2[p, :]
    end
    return U2[collect(outidxs), :]
end

# Exact `E[J.^2]` when each parent slot's local partial carries an independent sign, by
# enumerating all sign patterns.
function signed_second_moment(t, outidxs)
    N, n = nnodes(t), t.ninputs
    slots = [(k, i) for k in (n + 1):N for i in eachindex(t.parents[k])]
    S = length(slots)
    M = zeros(length(outidxs), n)
    for bits in 0:(2^S - 1)
        Jf = zeros(N, n)
        for j in 1:n
            Jf[j, j] = 1
        end
        for (s, (k, i)) in enumerate(slots)
            e = isodd(bits >> (s - 1)) ? -1 : 1
            Jf[k, :] .+= e * t.ds[k][i] .* Jf[t.parents[k][i], :]
        end
        M .+= Jf[collect(outidxs), :] .^ 2
    end
    return M ./ 2^S
end

@testset "gramvariance" begin
    gramvariance = DerivativeCovers.gramvariance
    # Outputs 1,2 depend on inputs 1,2 through reconvergent paths; output 3 on inputs 3,4.
    prog = xs -> [sin(xs[1]) * xs[2] + xs[1]^2, xs[2] * exp(xs[1]) * xs[2], xs[3] * xs[4] + cos(xs[4])]
    x0 = [0.3, 0.7, 1.1, -0.4]
    t, outidxs = tape_function(prog, x0)
    U = jacobian_surrogate(t, outidxs)
    U2 = squared_path_sums(t, outidxs)
    J = ForwardDiff.jacobian(prog, x0)
    m, n = size(J)
    rtol = 1.0e-13
    w = [0.5, -2.0, 0.0]

    @testset "equals the squared path sums" begin
        @test isapprox(gramvariance(t, outidxs), sqrt.(vec(sum(U2; dims = 1))); rtol)
        @test isapprox(gramvariance(t, outidxs, w), sqrt.(U2' * abs.(w)); rtol)
        @test gramvariance(t, outidxs, Diagonal(w)) == gramvariance(t, outidxs, w)
        @test gramvariance(t, outidxs, w) == gramvariance(t, outidxs, abs.(w))
    end

    @testset "is the expected Gram diagonal under random signs" begin
        rs = x -> [x[1] * x[1] + x[2], sin(x[1] * x[2]) * x[2]]
        ts, os = tape_function(rs, [0.7, -1.2])
        M = signed_second_moment(ts, os)
        @test isapprox(M, squared_path_sums(ts, os); rtol)
        ws = [2.0, 0.3]
        @test isapprox(gramvariance(ts, os, ws) .^ 2, M' * ws; rtol)
    end

    @testset "lies beneath h_gn and the Gram cover" begin
        @test all(U2 .<= U .^ 2 .* GATE)
        for ww in (ones(m), w)
            hv = gramvariance(t, outidxs, ww)
            hgn = sqrt.(U' .^ 2 * abs.(ww))
            @test all(hv .<= hgn .* GATE)
            @test all(hgn .<= gramcover(t, outidxs, ww) .* GATE)
        end
        # Reconvergent paths put it strictly below h_gn, and below the column norms of J.
        hv = gramvariance(t, outidxs)
        @test hv[1] < sqrt(sum(abs2, U[:, 1])) * (1 - 1.0e-3)
        tx, ox = tape_function(x -> [x[1] * x[1]], [3.0])
        @test gramvariance(tx, ox) ≈ [sqrt(18.0)]
        @test gramvariance(tx, ox)[1] < abs(ForwardDiff.jacobian(x -> [x[1] * x[1]], [3.0])[1])
    end

    @testset "single paths: equals h_gn and the column norms of J" begin
        r1 = x -> [x[1] * x[2], exp(x[1]) - x[3], x[2] + 2x[3], sin(x[3])]
        y0 = [0.5, 2.0, -1.0]
        t1, o1 = tape_function(r1, y0)
        J1 = ForwardDiff.jacobian(r1, y0)
        U1 = jacobian_surrogate(t1, o1)
        w1 = [1.0, 0.25, 3.0, 0.0]
        @test isapprox(gramvariance(t1, o1), sqrt.(vec(sum(abs2, U1; dims = 1))); rtol)
        @test isapprox(gramvariance(t1, o1), sqrt.(vec(sum(abs2, J1; dims = 1))); rtol)
        @test isapprox(gramvariance(t1, o1, w1), sqrt.(vec(sum(abs2, sqrt.(w1) .* J1; dims = 1))); rtol)
        nls = cover_nls(r1, y0)
        @test isapprox(nls.h_var_gn, nls.h_gn; rtol)
    end

    @testset "cover_nls carries it for the residuals" begin
        rn = x -> [x[1] * x[2] - 1.0, exp(x[1]) - 2.0, sin(x[1] * x[2]) * x[2], x[3] * x[3]]
        z0 = [0.5, 2.0, 0.3]
        tr, or = tape_function(rn, z0)
        nc = cover_nls(rn, z0)
        nd = cover_nls(WithDiagnostics(), rn, z0)
        @test isapprox(nc.h_var_gn, gramvariance(tr, or); rtol)
        @test nd.h_var_gn == nc.h_var_gn
        @test all(nc.h_var_gn .<= nc.h_gn .* GATE)
        @test nc.h_var_gn[3] < nc.h_gn[3] * (1 - 1.0e-3)
        # Coarser residual channels loosen h_gn but leave the scale alone.
        @test cover_nls(rn, z0; ngroups = 1).h_var_gn == nc.h_var_gn
    end

    @testset "input covariance" begin
        d = [2.0, 0.1, 5.0, 0.3]
        progd = y -> prog(y ./ d)
        td, od = tape_function(progd, d .* x0)
        @test isapprox(gramvariance(td, od), gramvariance(t, outidxs) ./ d; rtol)
        @test isapprox(gramvariance(td, od, w), gramvariance(t, outidxs, w) ./ d; rtol)
    end

    @testset "output covariance" begin
        C = [3.0, 0.02, 7.0]
        tc, oc = tape_function(xs -> C .* prog(xs), x0)
        wpos = [0.5, 2.0, 1.5]
        @test isapprox(gramvariance(tc, oc, wpos ./ C .^ 2), gramvariance(t, outidxs, wpos); rtol)
        @test isapprox(gramvariance(tc, oc, 1 ./ C .^ 2), gramvariance(t, outidxs); rtol)
    end

    @testset "a zero weight drops the output; a repeated output adds its weight" begin
        @test gramvariance(t, outidxs, [1.0, 0.0, 1.0]) == gramvariance(t, outidxs[[1, 3]])
        @test gramvariance(t, outidxs, [0.0, 0.0, 0.0]) == zeros(n)
        @test gramvariance(t, [outidxs[1], outidxs[1], outidxs[2]]) ==
            gramvariance(t, outidxs[1:2], [2.0, 1.0])
    end

    @testset "argument checks" begin
        @test_throws "w has indices" gramvariance(t, outidxs, [1.0, 2.0])
        @test_throws "weights must be finite" gramvariance(t, outidxs, [1.0, NaN, 2.0])
        @test_throws "diagonal weights only" gramvariance(t, outidxs, ones(m, m))
        @test_throws "diagonal weights only" gramvariance(t, outidxs, Matrix(Diagonal(w)))
    end
end
