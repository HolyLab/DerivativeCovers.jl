# The (max, ×) row statistic is pinned against a dense reference that forms every node's max-path
# vector and every atom's vector explicitly, at O(n L).

using DerivativeCovers: maxpath_forward, maxpath_rowcorrelation

function maxpath_reference(t, outidx, d)
    n, N = t.ninputs, DerivativeCovers.nnodes(t)
    U = [zeros(n) for _ in 1:N]
    for a in 1:n
        U[a][a] = 1
    end
    for k in (n + 1):N, (j, p) in enumerate(t.parents[k])
        U[k] .= max.(U[k], abs(t.ds[k][j]) .* U[p])
    end
    rhos, _ = local_covers(t, adjoints(t, outidx))
    m = zeros(n)
    for k in (n + 1):N
        isempty(rhos[k]) && continue
        ps = t.parents[k]
        for atom in DerivativeCovers.atom_slots(t.Cs[k], length(ps))
            s = zeros(n)
            for i in atom
                s .= max.(s, rhos[k][i] .* U[ps[i]])
            end
            for i in 1:n, l in 1:n
                (l == i || d[i] == 0 || d[l] == 0) && continue
                m[i] = max(m[i], s[i] * s[l] / (d[i] * d[l]))
            end
        end
    end
    return m
end

# A sum of squared linear forms, each input written once per form.
function squared_forms(W)
    return function (x)
        acc = zero(eltype(x))
        for j in axes(W, 2)
            form = W[1, j] * x[1]
            for l in 2:size(W, 1)
                form = form + W[l, j] * x[l]
            end
            acc = acc + form^2
        end
        return acc / 2
    end
end

@testset "maxpath.jl" begin
    @testset "matches the dense reference — $name" for (name, f, x0) in PROBLEMS
        t, outidx = tape_function(f, x0)
        h = cover_hessian(WithDiagnostics(), t, outidx).h_quad
        for d in (h, h .* exp.(range(-2, 2; length = length(h))))
            @test maxpath_rowcorrelation(t, outidx, d) ≈ maxpath_reference(t, outidx, d) rtol = 1.0e-12
        end
    end

    @testset "exact on a sum of squared forms" begin
        W = [
            1.0 0.0 -2.0 0.3
            0.5 1.0 0.0 0.0
            0.0 -3.0 0.1 1.0
            2.0 0.0 0.0 -0.7
        ]
        t, outidx = tape_function(squared_forms(W), [0.3, -1.0, 2.0, 0.5])
        h = cover_hessian(WithDiagnostics(), t, outidx).h_quad
        # Each form is one atom whose slot weight is 1 and whose max-path vector is abs(W[:, j]).
        @test h .^ 2 ≈ diag(W * W') rtol = 1.0e-12
        expected = [
            maximum(abs(W[i, j]) * maximum(abs(W[l, j]) / h[l] for l in axes(W, 1) if l != i)
                    for j in axes(W, 2)) / h[i] for i in axes(W, 1)
        ]
        @test maxpath_rowcorrelation(t, outidx, h) ≈ expected rtol = 1.0e-12
        # No larger than the correlation of the surrogate `abs.(W) * abs.(W)'`.
        A = abs.(W) * abs.(W)'
        rowmax = [maximum(abs(A[i, l]) / (h[i] * h[l]) for l in axes(A, 1) if l != i) for i in axes(A, 1)]
        @test all(maxpath_rowcorrelation(t, outidx, h) .<= rowmax .* (1 + 1.0e-12))
    end

    @testset "the forward sweep keeps the two largest distinct inputs" begin
        t, _ = tape_function(x -> x[1] * x[2] + 3 * x[3], [1.0, 2.0, 1.0])
        F = maxpath_forward(t, [1.0, 0.5, 4.0])
        k = DerivativeCovers.nnodes(t)
        # The output reaches x1 with weight 2 (scaled 2), x2 with weight 1 (scaled 2), x3 with 3/4.
        @test (F.v1[k], F.v2[k]) == (2.0, 2.0)
        @test Set((F.i1[k], F.i2[k])) == Set((1, 2))
    end

    @testset "invariance — $name" for (name, f, x0, D) in [
            ("rosenbrock n=2", rosenbrock, [-1.2, 1.0], [1.0e-6, 1.0e3]),
            (
                "logsumexp n=8", logsumexp,
                [-3.0, -1.0, -2.0, 0.0, 1.0, 2.0, -0.5, 3.0],
                [1.0e-3, 1.0e2, 1.0, 1.0e-1, 1.0e4, 1.0e-2, 1.0e1, 1.0e3],
            ),
            ("mlp 2-4-1", mlp_loss(MLP_XIN, MLP_YTAR), MLP_P, exp.(range(-3, 3; length = 17))),
        ]
        m = maxpath_rowcorrelation(tape_function(f, x0)..., cover_hessian(WithDiagnostics(), f, x0).h_quad)
        fD = xt -> 7.3 * f(xt ./ D)
        mD = maxpath_rowcorrelation(tape_function(fD, D .* x0)..., cover_hessian(WithDiagnostics(), fD, D .* x0).h_quad)
        @test mD ≈ m rtol = 1.0e-10
    end

    @testset "a zero scale is excluded" begin
        t, outidx = tape_function(x -> x[1]^2 * x[2] + x[3], [1.0, 2.0, 5.0])
        h = cover_hessian(WithDiagnostics(), t, outidx).h_quad
        @test h[3] == 0
        m = maxpath_rowcorrelation(t, outidx, h)
        @test m[3] == 0
        @test m ≈ maxpath_reference(t, outidx, h) rtol = 1.0e-12
    end

    @testset "argument checks" begin
        t, outidx = tape_function(rosenbrock, [-1.2, 1.0])
        @test_throws "expected 2 (one per input)" maxpath_rowcorrelation(t, outidx, [1.0])
        @test_throws "finite and nonnegative" maxpath_rowcorrelation(t, outidx, [1.0, -1.0])
        @test_throws "finite and nonnegative" maxpath_rowcorrelation(t, outidx, [1.0, Inf])
    end
end
