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

    @testset "generator argument checks" begin
        @test_throws ArgumentError quadratic(4; rank = 5)
        @test_throws ArgumentError quadratic(4; form = :bogus)
        @test_throws ArgumentError deep_composition(4; width = 0)
        @test_throws ArgumentError random_dag(4; sharing = 1.5)
        @test_throws ArgumentError cancellation(1)
        @test_throws "n must be at least 2" jacobian_cancellation(1)
    end

end
