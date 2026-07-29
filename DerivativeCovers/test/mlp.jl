# The array and scalar-loop MLPs must agree, and both covers must bound ForwardDiff's Hessian.

@testset "mlp.jl" begin

    sizes = (3, 4, 2)
    batch = 2
    p = mlp(sizes; batch)

    @testset "twin consistency" begin
        @test p.f(p.x0) ≈ p.twin(p.x0) rtol = 1.0e-12
        @test ForwardDiff.gradient(p.f, p.x0) ≈ ForwardDiff.gradient(p.twin, p.x0)
    end

    @testset "parity between the array program and its scalar-loop twin" begin
        rp = cover_hessian(WithDiagnostics(), p.f, p.x0)
        rs = cover_hessian(WithDiagnostics(), p.twin, p.x0)
        @test rp.grad ≈ rs.grad rtol = 1.0e-12
        @test rp.B ≈ rs.B rtol = 1.0e-12
        @test rp.h_sum ≈ rs.h_sum rtol = 1.0e-12
        @test rp.h_sum_rev ≈ rs.h_sum_rev rtol = 1.0e-12
        @test rp.h_quad ≈ rs.h_quad rtol = 1.0e-12

        vp = cover_variance(p.f, p.x0)
        vs = cover_variance(p.twin, p.x0)
        @test vp.h_sum ≈ vs.h_sum rtol = 1.0e-12
        @test vp.h_var ≈ vs.h_var rtol = 1.0e-12
        @test vp.severity ≈ vs.severity rtol = 1.0e-12
    end

    @testset "validity vs ForwardDiff" begin
        H = ForwardDiff.hessian(p.f, p.x0)
        res = cover_hessian(WithDiagnostics(), p.f, p.x0)
        @test res.grad ≈ ForwardDiff.gradient(p.f, p.x0)
        @test bound_ratio(H, res.B) <= GATE
        @test cover_ratio(H, res.h_sum) <= GATE
        @test cover_ratio(H, res.h_quad) <= GATE
        @test all(res.h_quad .<= res.h_sum .* GATE)
    end

    @testset "tape economy" begin
        # Per batch column: 3n1 hidden-layer nodes, 2n2 output-layer nodes, and 2n2+1 loss nodes.
        # Add parameter inputs and two final reduction nodes.
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
        res = cover_hessian(WithDiagnostics(), p.f, p.x0)
        rng = MersenneTwister(2)
        d = exp.(4.0 .* randn(rng, length(p.x0)))
        @test maximum(d) / minimum(d) > 1.0e2

        resd = cover_hessian(WithDiagnostics(), xt -> p.f(xt ./ d), d .* p.x0)
        @test resd.h_sum ≈ res.h_sum ./ d rtol = 1.0e-10
        @test resd.h_quad ≈ res.h_quad ./ d rtol = 1.0e-10

        c = 7.3
        resc = cover_hessian(WithDiagnostics(), x -> c * p.f(x), p.x0)
        @test resc.h_sum ≈ sqrt(c) .* res.h_sum rtol = 1.0e-12
        @test resc.h_quad ≈ sqrt(c) .* res.h_quad rtol = 1.0e-12
    end

    @testset "alternate activation" begin
        ps = mlp(sizes; batch, activation = sin)
        H = ForwardDiff.hessian(ps.f, ps.x0)
        res = cover_hessian(WithDiagnostics(), ps.f, ps.x0)
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
