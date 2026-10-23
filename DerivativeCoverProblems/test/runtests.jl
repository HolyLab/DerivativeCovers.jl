using DerivativeCoverProblems
using ForwardDiff
using LinearAlgebra
using Random
using Test

# Validate the problem definitions independently of DerivativeCovers: twins, analytic derivatives,
# and ForwardDiff results must agree at each evaluation point.
@testset "DerivativeCoverProblems.jl" begin

    GENERATED = [
        (sigma, g(n))
            for sigma in (0.0, 2.0)
            for (_, g) in families(; sigma, seed = 7)
            for n in (3, 6)
    ]

    @testset "$(p.name) sigma=$sigma" for (sigma, p) in GENERATED
        @test nvars(p) == length(p.x0)
        @test isfinite(p.f(p.x0))
        @test all(isfinite, ForwardDiff.gradient(p.f, p.x0))
        # A problem with no curvature poses no covering problem at all.
        @test any(!iszero, ForwardDiff.hessian(p.f, p.x0))
        if has_twin(p)
            @test p.twin(p.x0) ≈ p.f(p.x0)
            @test ForwardDiff.gradient(p.twin, p.x0) ≈ ForwardDiff.gradient(p.f, p.x0)
        end
        if has_exact_hessian(p)
            @test exact_hessian(p) ≈ ForwardDiff.hessian(p.f, p.x0)
        end
    end

    @testset "mlp" begin
        p = mlp([3, 4, 2]; batch = 2, rng = MersenneTwister(5))
        @test isfinite(p.f(p.x0))
        @test has_twin(p)
        @test p.twin(p.x0) ≈ p.f(p.x0)
        @test ForwardDiff.gradient(p.twin, p.x0) ≈ ForwardDiff.gradient(p.f, p.x0)
    end

    @testset "cancellation pairs" begin
        # The pair shares values and derivative magnitudes but differs in curvature through signs.
        pc = cancellation(6; m = 3, delta = 1.0e-2, aligned = false, rng = MersenneTwister(7))
        pa = cancellation(6; m = 3, delta = 1.0e-2, aligned = true, rng = MersenneTwister(7))
        @test exact_hessian(pc) ≈ ForwardDiff.hessian(pc.f, pc.x0)
        @test exact_hessian(pa) ≈ ForwardDiff.hessian(pa.f, pa.x0)
        @test ForwardDiff.gradient(pc.f, pc.x0) ≈ ForwardDiff.gradient(pa.f, pa.x0)
        @test pc.f(pc.x0) ≈ pc.twin(pc.x0)
        @test pa.f(pa.x0) ≈ pa.twin(pa.x0)

        jc = jacobian_cancellation(6; nout = 3, rng = MersenneTwister(7))
        @test noutputs(jc) == 3
        @test length(jc.r(jc.x0)) == 3
        @test has_exact_jacobian(jc)
        @test exact_jacobian(jc) ≈ ForwardDiff.jacobian(jc.r, jc.x0)
    end

    @testset "stationary outputs" begin
        # The output is stationary in each u_j to second order, so its derivative there is
        # 1 - cos(u_j) while the two paths carrying it contribute 1 and abs(cos(u_j)).
        for n in (2, 3, 8), delta in (1.0e-1, 1.0e-2)
            p = adjoint_cancellation(n; delta)
            @test nvars(p) == n
            @test exact_hessian(p) ≈ ForwardDiff.hessian(p.f, p.x0)
            @test p.twin(p.x0) ≈ p.f(p.x0)
            @test ForwardDiff.gradient(p.twin, p.x0) ≈ ForwardDiff.gradient(p.f, p.x0)
            # Every unit sits at the same u_j, which is what makes the family size-independent.
            @test all(≈(delta), p.x0[j] * p.x0[j + 1] for j in 1:(n - 1))
        end

        # More Newton steps, a more nearly stationary output: the inner residual falls
        # quadratically, so the gradient of the inner objective at the final iterate does too.
        inner_residual(k) = (p = converged_inner_solve(; steps = k);
                             abs(ForwardDiff.gradient(p.f, p.x0)[1]))
        @test inner_residual(0) > inner_residual(1) > inner_residual(2) > inner_residual(3)
    end

    @testset "rescale is exactly the input-units transformation" begin
        p = quadratic(4; rng = MersenneTwister(5))
        d = logscales(MersenneTwister(6), 4; sigma = 4.0)
        q = rescale(p, d)
        @test q.x0 ≈ d .* p.x0
        @test q.f(q.x0) ≈ p.f(p.x0)
        @test exact_hessian(q) ≈ Diagonal(inv.(d)) * exact_hessian(p) * Diagonal(inv.(d))

        @test_throws DimensionMismatch rescale(p, [1.0, 2.0])
        @test_throws ArgumentError rescale(p, [1.0, 2.0, 0.0, 1.0])
    end

    @testset "planted deep composition" begin
        p = deep_composition(5; depth = 3, width = 5, planted = 0.5, rng = MersenneTwister(3))
        q = deep_composition(5; depth = 3, width = 5, rng = MersenneTwister(3))
        @test p.x0 == q.x0
        @test p.twin(p.x0) ≈ p.f(p.x0)
        @test has_xstar(p) && !has_xstar(q)
        # With width == n the planted point is the unique zero of the first layer, W*x + b.
        W, b = p.twin.W, p.twin.b
        @test p.xstar ≈ -(W \ b)
        @test p.f(p.xstar) <= 1e-24
        @test p.twin(p.xstar) <= 1e-24
        @test 0 < norm(p.xstar - p.x0) < 10
        d = [1.0, 2.0, 0.5, 3.0, 0.25]
        @test rescale(p, d).xstar == d .* p.xstar
        @test_throws ArgumentError deep_composition(4; planted = -1.0)
    end

    @testset "Lotka-Volterra parameter estimation" begin
        p = lotka_volterra(3; delta = 0.2, nsteps = 20, every = 4, rng = MersenneTwister(2))
        @test nvars(p) == 3 + 9
        @test has_xstar(p)
        @test p.f(p.xstar) == 0
        @test p.twin(p.xstar) <= 1e-20
        @test isfinite(p.f(p.x0)) && p.f(p.x0) > 0
        @test p.twin(p.x0) ≈ p.f(p.x0)
        @test ForwardDiff.gradient(p.twin, p.x0) ≈ ForwardDiff.gradient(p.f, p.x0)
        @test ForwardDiff.hessian(p.twin, p.x0) ≈ ForwardDiff.hessian(p.f, p.x0)
        # The planted equilibrium r = -A*ybar is stable: the symmetric part of A*Diagonal(ybar) is negative definite.
        A = reshape(p.xstar[4:end], 3, 3)
        ybar = -(A \ p.xstar[1:3])
        @test all(>(0), ybar)
        @test maximum(eigvals(Symmetric(A * Diagonal(ybar) + (A * Diagonal(ybar))'))) < 0
        @test p.x0 == lotka_volterra(3; delta = 0.2, nsteps = 20, every = 4, rng = MersenneTwister(2)).x0
        @test_throws ArgumentError lotka_volterra(3; every = 0)
        @test_throws ArgumentError lotka_volterra(3; delta = -0.1)
        @test_throws DimensionMismatch p.f(zeros(5))
    end

    @testset "generator argument checks" begin
        @test_throws ArgumentError quadratic(4; rank = 5)
        @test_throws ArgumentError quadratic(4; form = :bogus)
        @test_throws ArgumentError deep_composition(4; width = 0)
        @test_throws ArgumentError random_dag(4; sharing = 1.5)
        @test_throws ArgumentError cancellation(1)
        @test_throws "n must be at least 2" jacobian_cancellation(1)
        @test_throws "n must be at least 2" adjoint_cancellation(1)
        @test_throws "delta must be positive" adjoint_cancellation(4; delta = 0.0)
        @test_throws "steps must be nonnegative" converged_inner_solve(; steps = -1)
    end

end
