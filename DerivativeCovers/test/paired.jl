# The fused signed/absolute sweep must match both ForwardDiff and the constructive cover path.

@testset "paired.jl" begin

    @testset "matches constructive path — $name" for (name, f, x0) in PROBLEMS
        t, outidx = tape_function(f, x0)
        r = cover_hessian(WithDiagnostics(), t, outidx)
        pr = cover_hessian(CoverOnly(), t, outidx)
        @test pr.grad ≈ r.grad rtol = 1.0e-12
        @test pr.h_sum ≈ r.h_sum rtol = 1.0e-12
        @test pr.grad ≈ ForwardDiff.gradient(f, x0) rtol = 1.0e-8

        H = ForwardDiff.hessian(f, x0)
        @test all(abs.(H) .<= pr.h_sum * pr.h_sum' .* GATE)

        # `CoverOnly()` is what an unadorned call returns, from either entry point.
        for c in (cover_hessian(t, outidx), cover_hessian(f, x0))
            @test c isa HessianCover
            @test c.grad == pr.grad
            @test c.h_sum == pr.h_sum
        end
    end

    # Given finished adjoints, the sweep returns the same channels without forming them again.
    @testset "from adjoints — $name" for (name, f, x0) in PROBLEMS
        t, outidx = tape_function(f, x0)
        vbar = DerivativeCovers.adjoints(t, outidx)
        for check in (true, false), (scales, ratio) in ((nothing, nothing), (2 .* abs.(x0) .+ 1, 1.0e-3))
            pr = cover_hessian(CoverOnly(), t, outidx; check, scales, ratio)
            c = cover_hessian(CoverOnly(), t, vbar; check, scales, ratio)
            @test c isa HessianCover{eltype(x0)}
            @test c.grad == pr.grad
            @test c.h_sum == pr.h_sum
            @test c.h_var == pr.h_var
        end
    end

    @testset "from adjoints: argument checks" begin
        t, outidx = tape_function(x -> x[1] * x[2]^2, [0.5, 2.0])
        vbar = DerivativeCovers.adjoints(t, outidx)
        @test_throws "one adjoint per tape node" cover_hessian(CoverOnly(), t, vbar[1:(end - 1)])
        @test_throws "CoverOnly cannot merge over SignedTangents" cover_hessian(
            CoverOnly(), t, vbar; channel = SignedTangents()
        )
    end

    @testset "primitive tapes — $name" for (name, fp, _, x0) in PRIMITIVE_PARITY
        t, outidx = tape_function(fp, x0)
        r = cover_hessian(WithDiagnostics(), t, outidx)
        pr = cover_hessian(CoverOnly(), t, outidx)
        @test pr.grad ≈ r.grad rtol = 1.0e-12
        @test pr.h_sum ≈ r.h_sum rtol = 1.0e-12
    end

    @testset "declared sub-map tape" begin
        inner = v -> [v[1] * v[2], exp(v[1])]
        f = function (x)
            y = submap(inner, [x[1], x[2]])
            return y[1] + tanh(y[2]) * x[3]
        end
        x0 = [0.8, -1.1, 0.6]
        t, outidx = tape_function(f, x0)
        r = cover_hessian(WithDiagnostics(), t, outidx)
        pr = cover_hessian(CoverOnly(), t, outidx)
        @test pr.grad ≈ r.grad rtol = 1.0e-12
        @test pr.h_sum ≈ r.h_sum rtol = 1.0e-12
    end
end
