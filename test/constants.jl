# Numeric literals obtained from the element type. `one(eltype(x))`/`zero(eltype(x))` are the
# generic-code idiom for a typed literal, so a program that uses them must tape and cover.

@testset "constants.jl" begin

    @testset "literals are tape-free constants" begin
        @test one(TVar) === 1.0
        @test zero(TVar) === 0.0
        @test oneunit(TVar) === 1.0
        t = Tape([2.5])
        x = TVar(t, 1)
        @test one(x) === 1.0
        @test zero(x) === 0.0
        @test oneunit(x) === 1.0
        # No node was recorded for any of them.
        @test nnodes(t) == 1
    end

    @testset "objective seeded with one(eltype(x))" begin
        f = function (x)
            s = one(eltype(x))
            z = zero(eltype(x))
            return (x[1] + 2s)^2 + (x[2] - s)^4 + z * x[1]
        end
        x0 = [3.0, 4.0]
        r = cover_hessian(f, x0)
        @test r.grad ≈ ForwardDiff.gradient(f, x0) rtol = 1e-8
        H = ForwardDiff.hessian(f, x0)
        for h in (r.h_sum, r.h_quad)
            @test all(abs.(H) .<= h * h' .* GATE)
        end
    end
end
