# Tests for the `mlp` generator in problems.jl. As in arrayops.jl, the core claim is parity
# between the array-notation program and its scalar-loop twin, plus validity of every cover
# against ForwardDiff's exact Hessian.

@testset "mlp.jl" begin

    sizes = (3, 4, 2)
    batch = 2
    p = mlp(sizes; batch)

    @testset "twin consistency" begin
        @test p.f(p.x0) ≈ p.twin(p.x0) rtol=1e-12
        @test ForwardDiff.gradient(p.f, p.x0) ≈ ForwardDiff.gradient(p.twin, p.x0)
    end

    @testset "parity between the array program and its scalar-loop twin" begin
        rp = cover_hessian(p.f, p.x0)
        rs = cover_hessian(p.twin, p.x0)
        @test rp.grad ≈ rs.grad rtol=1e-12
        @test rp.B ≈ rs.B rtol=1e-12
        @test rp.h_sum ≈ rs.h_sum rtol=1e-12
        @test rp.h_sum_rev ≈ rs.h_sum_rev rtol=1e-12
        @test rp.h_quad ≈ rs.h_quad rtol=1e-12

        vp = cover_variance(p.f, p.x0)
        vs = cover_variance(p.twin, p.x0)
        @test vp.h_sum ≈ vs.h_sum rtol=1e-12
        @test vp.h_var ≈ vs.h_var rtol=1e-12
        @test vp.severity ≈ vs.severity rtol=1e-12
    end

    @testset "validity vs ForwardDiff" begin
        H = ForwardDiff.hessian(p.f, p.x0)
        res = cover_hessian(p.f, p.x0)
        @test res.grad ≈ ForwardDiff.gradient(p.f, p.x0)
        @test bound_ratio(H, res.B) <= GATE
        @test cover_ratio(H, res.h_sum) <= GATE
        @test cover_ratio(H, res.h_quad) <= GATE
        @test all(res.h_quad .<= res.h_sum .* GATE)
    end

    @testset "tape economy" begin
        # sizes = (n0, n1, n2) = (3, 4, 2), so K = 2 layers, one hidden (l=1) and one linear
        # output (l=2). Per batch column:
        #   hidden layer:  n1 matvec rows (linearnode, TVar weights x Real data) +
        #                  n1 bias adds (binarynode) + n1 activations (unarynode) = 3*n1 nodes
        #   output layer:  n2 matvec rows (dotnode, TVar x TVar) +
        #                  n2 bias adds (binarynode) = 2*n2 nodes
        #   residual:      n2 subtracts + n2 abs2 (unarynode each) +
        #                  1 linear sum over the n2 terms = 2*n2 + 1 nodes
        # giving 3*n1 + 4*n2 + 1 nodes per column. Across `batch` columns, plus the parameters
        # as inputs, plus one linear node summing the batch, plus one unary node for the final
        # `/2`:
        n0, n1, n2 = sizes
        nparams = n1 * (n0 + 1) + n2 * (n1 + 1)
        per_column = 3 * n1 + 4 * n2 + 1
        expected = nparams + batch * per_column + 2

        t, _ = tape_function(p.f, p.x0)
        @test nnodes(t) == expected

        tw, _ = tape_function(p.twin, p.x0)
        @test nnodes(tw) > nnodes(t)
    end

    @testset "covariance" begin
        res = cover_hessian(p.f, p.x0)
        rng = MersenneTwister(2)
        d = exp.(4.0 .* randn(rng, length(p.x0)))
        @test maximum(d) / minimum(d) > 1e2

        resd = cover_hessian(xt -> p.f(xt ./ d), d .* p.x0)
        @test resd.h_sum ≈ res.h_sum ./ d rtol=1e-10
        @test resd.h_quad ≈ res.h_quad ./ d rtol=1e-10

        c = 7.3
        resc = cover_hessian(x -> c * p.f(x), p.x0)
        @test resc.h_sum ≈ sqrt(c) .* res.h_sum rtol=1e-12
        @test resc.h_quad ≈ sqrt(c) .* res.h_quad rtol=1e-12
    end

    @testset "alternate activation" begin
        ps = mlp(sizes; batch, activation=sin)
        H = ForwardDiff.hessian(ps.f, ps.x0)
        res = cover_hessian(ps.f, ps.x0)
        @test res.grad ≈ ForwardDiff.gradient(ps.f, ps.x0)
        @test cover_ratio(H, res.h_sum) <= GATE
        @test cover_ratio(H, res.h_quad) <= GATE
    end

    @testset "errors" begin
        @test_throws ArgumentError mlp((3,))
        @test_throws ArgumentError mlp((3, 0, 2))
        @test_throws DimensionMismatch p.f(rand(length(p.x0) + 1))
    end

end
