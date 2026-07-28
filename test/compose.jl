# Tests for compositional covers: the function-level composition rules and the declared-sub-map
# primitive. Ground truth throughout is ForwardDiff on the plain composite. The contracts pinned:
# first order through a declared call is exact (gradients, Jacobians), the covers built through
# it are valid on the true Hessian conditional on a declaration the tests audit directly, hvp
# fails fast, and the declared path is scale-covariant.

@testset "compositional covers" begin

    # A sub-map with nonlinear, transcendental, and exactly linear outputs (row 4 of S must be
    # exactly zero), and a scalar consumer of its outputs. Both run on TVar and plain vectors.
    Fsub = v -> [v[1] * v[2], sin(v[3]) + v[1], exp(0.3 * v[2]) * v[3], v[1] + v[2] + v[3]]
    gout = y -> y[1] * y[4] + exp(y[2]) + y[3]^2
    x0 = [0.7, -0.4, 1.1]

    @testset "compose_jacobian_cover" begin
        rng = MersenneTwister(30)
        J2, J1 = randn(rng, 4, 3), randn(rng, 3, 5)
        a2, b2 = ones(4), vec(maximum(abs.(J2); dims=1))
        a1, b1 = ones(3), vec(maximum(abs.(J1); dims=1))
        @test jacobian_ratio(J2, a2, b2) <= GATE     # the factors are covered...
        @test jacobian_ratio(J1, a1, b1) <= GATE
        pair = compose_jacobian_cover(a2, b2, a1, b1)
        @test jacobian_ratio(J2 * J1, pair.a, pair.b) <= GATE   # ...so the product is too

        # Exact on sign-definite rank-1 factors covered exactly.
        u2, w2 = [2.0, 0.25, 1.0], [4.0, 1.0]
        u1, w1 = [1.0, 2.0], [0.5, 3.0, 1.0]
        pr = compose_jacobian_cover(u2, w2, u1, w1)
        @test pr.a * pr.b' ≈ (u2 * w2') * (u1 * w1') rtol = 1e-14

        # A rescaling of the middle space (b2 → b2 ./ c alongside a1 → c .* a1) leaves the
        # composed pair bit-identical: the inner product is invariant.
        c = [4.0, 0.5]
        prc = compose_jacobian_cover(u2, w2 ./ c, c .* u1, w1)
        @test prc.a == pr.a && prc.b == pr.b

        @test_throws DimensionMismatch compose_jacobian_cover(u2, w2, [1.0, 2.0, 3.0], w1)
        @test_throws "nonnegative" compose_jacobian_cover(u2, w2, [-1.0, 2.0], w1)
    end

    @testset "declare_submap: exact first order, covered curvature" begin
        d = declare_submap(Fsub, x0)
        @test d.y ≈ Fsub(x0) rtol = 1e-14
        @test d.J ≈ ForwardDiff.jacobian(Fsub, x0) rtol = 1e-12
        for k in 1:4
            Hk = ForwardDiff.hessian(v -> Fsub(v)[k], x0)
            @test cover_ratio(Hk, d.S[k, :]) <= GATE
        end
        @test all(iszero, d.S[4, :])                 # the linear output declares no curvature

        # A scalar-valued sub-map is declared as one output.
        ds = declare_submap(v -> v[1] * exp(v[2]), x0[1:2])
        @test size(ds.J) == (1, 2) && size(ds.S) == (1, 2)
        @test vec(ds.J) ≈ ForwardDiff.gradient(v -> v[1] * exp(v[2]), x0[1:2]) rtol = 1e-12
    end

    @testset "compose_hessian_cover vs ForwardDiff" begin
        y0 = Fsub(x0)
        d = declare_submap(Fsub, x0)
        jc = cover_jacobian(Fsub, x0)
        hg = cover_hessian(WithDiagnostics(), gout, y0)
        gamma = abs.(ForwardDiff.gradient(gout, y0))
        res = compose_hessian_cover(hg.h_sum, gamma, jc.a, jc.b_max, d.S)
        H = ForwardDiff.hessian(x -> gout(Fsub(x)), x0)
        @test cover_ratio(H, res.h_sum) <= GATE
        @test cover_ratio(H, res.h_quad) <= GATE
        @test all(res.h_quad .<= res.h_sum .* GATE)

        # Linear F: S = nothing, and the composite Hessian is J'*∇²g*J alone.
        A = [1.0 2.0; -0.5 1.0; 0.25 -2.0]
        xl = [0.3, -0.9]
        gl = y -> exp(y[1]) + y[2] * y[3]
        dl = declare_submap(v -> A * v, xl)
        @test all(iszero, dl.S)
        jcl = cover_jacobian(v -> A * v, xl)
        hgl = cover_hessian(WithDiagnostics(), gl, A * xl)
        gammal = abs.(ForwardDiff.gradient(gl, A * xl))
        resl = compose_hessian_cover(hgl.h_sum, gammal, jcl.a, jcl.b_max, nothing)
        Hl = ForwardDiff.hessian(x -> gl(A * x), xl)
        @test cover_ratio(Hl, resl.h_sum) <= GATE

        @test_throws DimensionMismatch compose_hessian_cover(hg.h_sum[1:3], gamma, jc.a, jc.b_max)
        @test_throws DimensionMismatch compose_hessian_cover(hg.h_sum, gamma, jc.a, jc.b_max, d.S[:, 1:2])
        @test_throws "nonnegative" compose_hessian_cover(-hg.h_sum, gamma, jc.a, jc.b_max)
    end

    @testset "declared call on the tape — $name" for (name, fdecl, ftraced) in [
        ("inputs as arguments",
         xs -> gout(submap(Fsub, xs)),
         x -> gout(Fsub(x))),
        ("intermediates as arguments",
         xs -> gout(submap(Fsub, [xs[1] * xs[2], xs[3] + xs[1], xs[2]])),
         x -> gout(Fsub([x[1] * x[2], x[3] + x[1], x[2]]))),
    ]
        t, o = tape_function(fdecl, x0)
        res = cover_hessian(WithDiagnostics(), t, o)
        ttr, otr = tape_function(ftraced, x0)
        rtr = cover_hessian(WithDiagnostics(), ttr, otr)
        H = ForwardDiff.hessian(ftraced, x0)

        @test res.grad ≈ ForwardDiff.gradient(ftraced, x0) rtol = 1e-12
        @test res.grad ≈ rtr.grad rtol = 1e-12
        @test cover_ratio(H, res.h_sum) <= GATE
        @test cover_ratio(H, res.h_quad) <= GATE
        @test all(abs.(H) .<= res.B .* GATE .+ eps())
        @test nnodes(t) < nnodes(ttr)               # the call is recorded compressed

        # The variance channel runs through the declared node and shares the hard channel.
        vres = cover_variance(t, o)
        @test vres.h_sum ≈ res.h_sum rtol = 1e-12
        @test all(vres.severity .>= 1 - 1e-12)

        # Exact Hessian-vector products are impossible through a declared call.
        @test_throws "declared sub-map output" hvp(t, o, ones(length(x0)))
    end

    @testset "first-order consumers through a declared call" begin
        progd = xs -> begin
            ys = submap(Fsub, xs)
            [ys[1] * ys[2], ys[3] + xs[1]]
        end
        progt = x -> begin
            y = Fsub(x)
            [y[1] * y[2], y[3] + x[1]]
        end
        J = ForwardDiff.jacobian(progt, x0)
        t, outidxs = tape_function(progd, x0)
        @test adjoints(t, outidxs)[:, 1:t.ninputs] ≈ J rtol = 1e-12
        jc = cover_jacobian(t, outidxs)
        @test jacobian_ratio(J, jc.a, jc.b_max) <= GATE
        bal = balance_cover(t, outidxs)
        @test jacobian_ratio(J, bal.a, bal.b) <= GATE
    end

    @testset "nested declared calls" begin
        # declare_submap over a program that itself contains a declared call: the scratch-tape
        # sweeps handle CoveredPartials nodes, so declarations compose.
        G = xs -> begin
            ys = submap(Fsub, xs)
            [ys[1] + ys[2]^2, ys[3] * ys[4]]
        end
        Gtwin = x -> begin
            y = Fsub(x)
            [y[1] + y[2]^2, y[3] * y[4]]
        end
        d = declare_submap(G, x0)
        @test d.J ≈ ForwardDiff.jacobian(Gtwin, x0) rtol = 1e-12
        for k in 1:2
            Hk = ForwardDiff.hessian(x -> Gtwin(x)[k], x0)
            @test cover_ratio(Hk, d.S[k, :]) <= GATE
        end
    end

    @testset "submapnode error paths" begin
        t = Tape(x0)
        xs = [TVar(t, i) for i in 1:3]
        y, J, S = [1.0, 2.0], ones(2, 3), ones(2, 3)
        @test_throws "no arguments" submapnode(TVar[], y, ones(2, 0))
        @test_throws "no outputs" submapnode(xs, Float64[], ones(0, 3))
        @test_throws DimensionMismatch submapnode(xs, y, ones(2, 2))
        @test_throws DimensionMismatch submapnode(xs, y, J, ones(3, 3))
        @test_throws "nonnegative" submapnode(xs, y, J, -S)
        t2 = Tape(x0)
        @test_throws "different tapes" submapnode([xs[1], TVar(t2, 1), xs[3]], y, J)
    end

    @testset "covariance of the declared path" begin
        base = xs -> gout(submap(Fsub, xs))
        e = ones(3)
        te, oe = tape_function(xt -> base(xt ./ e), x0)
        re = cover_hessian(WithDiagnostics(), te, oe)

        # Conjugation by an input rescaling and an objective scale: h → sqrt(c) .* d .* h.
        # The wrapper is applied on both sides so the two tapes have identical shape.
        dvec = [2.0, 0.5, 4.0]
        c = 4.0
        tc, oc = tape_function(xt -> c * base(xt ./ (1 ./ dvec)), x0 ./ dvec)
        rc = cover_hessian(WithDiagnostics(), tc, oc)
        @test rc.grad ≈ c .* dvec .* re.grad rtol = 1e-12
        @test rc.h_sum ≈ sqrt(c) .* dvec .* re.h_sum rtol = 1e-12
        @test rc.h_quad ≈ sqrt(c) .* dvec .* re.h_quad rtol = 1e-12

        # Declaration-gauge invariance: rescaling the declared outputs y → C*y (with J → C*J,
        # S → sqrt.(C)*S, and the consumer reading ys ./ C) leaves the covers unchanged — the
        # gauge cancels between each output's adjoint and its declared curvature.
        dd = declare_submap(Fsub, x0)
        gauge = C -> tape_function(x0) do xs
            ys = submapnode(xs, C .* dd.y, C .* dd.J, sqrt.(C) .* dd.S)
            gout([ys[k] / C[k] for k in eachindex(C)])
        end
        t1, o1 = gauge(ones(4))
        tC, oC = gauge([4.0, 0.25, 16.0, 1.0])
        r1, rC = cover_hessian(WithDiagnostics(), t1, o1), cover_hessian(WithDiagnostics(), tC, oC)
        @test rC.h_sum == r1.h_sum
        @test rC.h_quad == r1.h_quad
        @test rC.grad == r1.grad
    end
end
