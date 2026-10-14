# Upward-rounded arithmetic on magnitudes.

using DerivativeCovers: Upward, boundtype

@testset "Upward" begin
    exact(x::Upward) = Rational{BigInt}(x.x)
    exact(x::AbstractFloat) = Rational{BigInt}(x)

    @testset "each operation bounds the exact result and steps at most once" begin
        rng = Xoshiro(3)
        ok = true
        for _ in 1:2000
            a, b = exp2(rand(rng, -40:40)) * (rand(rng) + 0.5), exp2(rand(rng, -40:40)) * (rand(rng) + 0.5)
            A, B = Upward(a), Upward(b)
            for (r, ex, near) in (
                    (A + B, exact(a) + exact(b), a + b),
                    (A * B, exact(a) * exact(b), a * b),
                    (A / B, exact(a) / exact(b), a / b),
                )
                ok &= r isa Upward{Float64} && exact(r) >= ex && r.x in (near, nextfloat(near))
            end
            s = sqrt(A)
            ok &= exact(s)^2 >= exact(a) && s.x in (sqrt(a), nextfloat(sqrt(a)))
        end
        @test ok
        # A product that underflows still bounds the exact one.
        p = Upward(floatmin()) * Upward(0.3)
        @test exact(p) >= exact(floatmin()) * exact(0.3)
        @test exact(Upward(1.0e-300) * Upward(1.0e-300)) > 0
    end

    @testset "exact operations do not step" begin
        x = Upward(0.1)
        @test x + Upward(0.0) === x
        @test x * Upward(1.0) === x
        @test x * Upward(0.0) === Upward(0.0)
        @test x / Upward(1.0) === x
        @test Upward(0.0) / x === Upward(0.0)
        @test sqrt(Upward(0.0)) === Upward(0.0)
        @test sqrt(Upward(1.0)) === Upward(1.0)
        @test x^2 === x * x
    end

    @testset "conversion and promotion" begin
        @test Upward(1.5) + 2 isa Upward{Float64}
        @test 2.0 * Upward(1.5) isa Upward{Float64}
        @test convert(Float64, Upward(1.5)) === 1.5
        @test exact(Upward{Float64}(1 // 3)) >= 1 // 3
        @test Upward{Float64}(1 // 3).x in (Float64(1 // 3), nextfloat(Float64(1 // 3)))
        @test Upward{Float64}(2) === Upward(2.0)
        @test zero(Upward{Float64}) === Upward(0.0)
        @test one(Upward{Float64}) === Upward(1.0)
        @test max(Upward(1.0), Upward(2.0)) === Upward(2.0)
        @test Upward(1.0) < 2 && Upward(2.0) == 2
        @test sprint(show, Upward(1.5)) == "1.5"
    end

    @testset "operations that do not preserve an upper bound are not defined" begin
        @test_throws "subtraction is decreasing" Upward(2.0) - Upward(1.0)
        @test_throws "subtraction is decreasing" Upward(2.0) - 1.0
        @test_throws MethodError log(Upward(2.0))
        @test_throws MethodError exp(Upward(2.0))
    end

    @testset "boundtype" begin
        @test boundtype(Float64) === Float64
        @test boundtype(Float32) === Float32
        @test boundtype(Complex{Float64}) === Float64
    end

    @testset "composition in upward arithmetic" begin
        a2, b2, a1, b1 = [0.3, 0.7], [0.1, 1.1], [0.6, 0.2], [1.3, 0.15, 0.9]
        c = compose_jacobian_cover(Upward.(a2), Upward.(b2), Upward.(a1), Upward.(b1))
        @test eltype(c.a) === eltype(c.b) === Upward{Float64}
        cn = compose_jacobian_cover(a2, b2, a1, b1)
        @test all(c.a .>= cn.a) && all(c.b .>= cn.b)
        hg, gamma, S = [0.4, 1.7], [0.3, 0.6], [0.1 0.2 0.3; 0.4 0.5 0.6]
        h = compose_hessian_cover(Upward.(hg), Upward.(gamma), Upward.(a2), Upward.(b1), Upward.(S))
        hn = compose_hessian_cover(hg, gamma, a2, b1, S)
        @test eltype(h.h_sum) === eltype(h.h_quad) === Upward{Float64}
        @test all(h.h_sum .>= hn.h_sum) && all(h.h_quad .>= hn.h_quad)
    end
end
