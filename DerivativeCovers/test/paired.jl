# The cover sweeps, which carry the adjoint beside one cover channel, must match both ForwardDiff
# and the constructive cover path.

@testset "paired.jl" begin

    @testset "matches constructive path — $name" for (name, f, x0) in PROBLEMS
        t, outidx = tape_function(f, x0)
        r = cover_hessian(WithDiagnostics(), t, outidx)
        pr = cover_hessian(CoverOnly(), t, outidx)
        @test pr.grad ≈ r.grad rtol = 1.0e-12
        @test pr.h ≈ r.h_sum rtol = 1.0e-12
        @test pr.grad ≈ ForwardDiff.gradient(f, x0) rtol = 1.0e-8

        H = ForwardDiff.hessian(f, x0)
        @test all(abs.(H) .<= pr.h * pr.h' .* GATE)

        # `CoverOnly()` is what an unadorned call returns, from either entry point.
        for c in (cover_hessian(t, outidx), cover_hessian(f, x0))
            @test c isa HessianCover{Float64, Float64, SumMerge}
            @test c.grad == pr.grad
            @test c.h == pr.h
        end
    end

    # Given finished adjoints, the sweep returns the same channels without forming them again.
    @testset "from adjoints — $name" for (name, f, x0) in PROBLEMS
        t, outidx = tape_function(f, x0)
        vbar = DerivativeCovers.adjoints(t, outidx)
        for check in (true, false), (scales, ratio) in ((nothing, nothing), (2 .* abs.(x0) .+ 1, 1.0e-3))
            for merge in (SumMerge(), SignAveragedMerge())
                pr = cover_hessian(CoverOnly(), t, outidx; merge, check, scales, ratio)
                c = cover_hessian(CoverOnly(), t, vbar; merge, check, scales, ratio)
                @test c isa HessianCover{eltype(x0), <:Any, typeof(merge)}
                @test c.grad == pr.grad
                @test c.h == pr.h
            end
        end
    end

    @testset "from adjoints: argument checks" begin
        t, outidx = tape_function(x -> x[1] * x[2]^2, [0.5, 2.0])
        vbar = DerivativeCovers.adjoints(t, outidx)
        @test_throws "one adjoint per tape node" cover_hessian(CoverOnly(), t, vbar[1:(end - 1)])
    end

    @testset "the merge keyword selects one cover" begin
        f, x0 = x -> x[1] * x[2]^2 + x[1] * x[1], [0.5, 2.0]
        t, outidx = tape_function(f, x0)
        vbar = DerivativeCovers.adjoints(t, outidx)
        for merge in (SumMerge(), SignAveragedMerge())
            @test @inferred(cover_hessian(CoverOnly(), t, outidx; merge)) isa
                HessianCover{Float64, Float64, typeof(merge)}
            @test @inferred(cover_hessian(CoverOnly(), t, vbar; merge)) isa
                HessianCover{Float64, Float64, typeof(merge)}
            @test cover_hessian(f, x0; merge) isa HessianCover{Float64, Float64, typeof(merge)}
        end
        s = cover_hessian(f, x0)
        v = cover_hessian(f, x0; merge = SignAveragedMerge())
        @test v.grad == s.grad
        @test all(v.h .<= s.h)
        @test any(v.h .< s.h)       # `x[1] * x[1]` seeds two slots
        for merge in (QuadMerge(), MaxMerge())
            @test_throws "cannot be computed by a reverse sweep" cover_hessian(f, x0; merge)
            @test_throws "cannot be computed by a reverse sweep" cover_hessian(CoverOnly(), t, vbar; merge)
        end
        # The diagnostic entry point returns the sum cover only.
        @test cover_hessian(WithDiagnostics(), f, x0; merge = SumMerge()).h_sum ≈ s.h rtol = 1.0e-12
        @test_throws "available only with CoverOnly()" cover_hessian(
            WithDiagnostics(), f, x0; merge = SignAveragedMerge()
        )
        @test sprint(show, MIME"text/plain"(), v) == "HessianCover (sign-averaged): 2 inputs; grad, h, nguarded"
    end

    @testset "primitive tapes — $name" for (name, fp, _, x0) in PRIMITIVE_PARITY
        t, outidx = tape_function(fp, x0)
        r = cover_hessian(WithDiagnostics(), t, outidx)
        pr = cover_hessian(CoverOnly(), t, outidx)
        @test pr.grad ≈ r.grad rtol = 1.0e-12
        @test pr.h ≈ r.h_sum rtol = 1.0e-12
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
        @test pr.h ≈ r.h_sum rtol = 1.0e-12
    end
end
