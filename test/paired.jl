# The paired sweep must reproduce, in one reverse pass, both outputs of the constructive
# production path: the exact gradient (signed channel, vs `adjoints` and ForwardDiff) and the
# sum-merge cover (absolute channel, vs `local_covers` + `abs_reverse`).

@testset "paired.jl" begin

    @testset "matches constructive path — $name" for (name, f, x0) in PROBLEMS
        t, outidx = tape_function(f, x0)
        r = cover_hessian(WithDiagnostics(), t, outidx)
        pr = cover_hessian(CoverOnly(), t, outidx)
        @test pr.grad ≈ r.grad rtol = 1e-12
        @test pr.h ≈ r.h_sum rtol = 1e-12
        @test pr.grad ≈ ForwardDiff.gradient(f, x0) rtol = 1e-8

        H = ForwardDiff.hessian(f, x0)
        @test all(abs.(H) .<= pr.h * pr.h' .* GATE)

        # `CoverOnly()` is what an unadorned call returns, from either entry point.
        @test cover_hessian(t, outidx) == pr
        @test cover_hessian(f, x0) == pr
    end

    @testset "primitive tapes — $name" for (name, fp, _, x0) in PRIMITIVE_PARITY
        t, outidx = tape_function(fp, x0)
        r = cover_hessian(WithDiagnostics(), t, outidx)
        pr = cover_hessian(CoverOnly(), t, outidx)
        @test pr.grad ≈ r.grad rtol = 1e-12
        @test pr.h ≈ r.h_sum rtol = 1e-12
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
        @test pr.grad ≈ r.grad rtol = 1e-12
        @test pr.h ≈ r.h_sum rtol = 1e-12
    end
end
