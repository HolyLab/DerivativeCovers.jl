# A float with a separate exponent, and the declared scales it carries.

using DerivativeCovers: ExpScaled, split_block, SplitScale

@testset "ExpScaled" begin
    @testset "rounds as $F does in $F's normal range" for F in (Float64, Float32)
        rng = Xoshiro(5)
        lo, hi = exponent(floatmin(F)) ÷ 2, exponent(floatmax(F)) ÷ 2
        rnd() = (x = ldexp(F(0.5) + rand(rng, F), rand(rng, lo:hi)); rand(rng, Bool) ? -x : x)
        isnormalresult(r) = isfinite(r) && (iszero(r) || abs(r) >= floatmin(F))
        mismatches = 0
        for _ in 1:20_000
            a, b = rnd(), rnd()
            for op in (+, -, *, /)
                r = op(a, b)
                isnormalresult(r) || continue
                mismatches += F(op(ExpScaled(a), ExpScaled(b))) !== r
            end
            mismatches += F(sqrt(ExpScaled(abs(a)))) !== sqrt(abs(a))
            # Near-cancellation, where the exponent alignment matters most.
            c = nextfloat(a, rand(rng, -3:3))
            mismatches += F(ExpScaled(a) - ExpScaled(c)) !== a - c
            # Mixed with a plain number, through promotion.
            r = a * b
            isnormalresult(r) && (mismatches += F(a * ExpScaled(b)) !== r)
        end
        @test mismatches == 0
    end

    @testset "products and sums stay off zero and infinity" begin
        x = ExpScaled(1.0e-300)^2
        @test x > 0
        @test Float64(x) == 0
        @test Float64(sqrt(x)) == 1.0e-300
        @test Float64(x * 1.0e300 * 1.0e300) ≈ 1.0e-0
        y = ExpScaled(1.0e300)^2
        @test isfinite(y)
        @test Float64(y) == Inf
        @test Float64(x * y) ≈ 1
        # A sum whose terms are far apart in exponent returns the larger, as the plain sum would.
        @test x + 1.0 == 1.0
        @test Float64(x + x) == 0 && (x + x) / x == 2
        # Subnormal inputs are held at full precision.
        @test ExpScaled(nextfloat(0.0)) == ExpScaled(0.5, -1073)
        @test Float64(ExpScaled(nextfloat(0.0))) === nextfloat(0.0)
    end

    @testset "zero, sign and non-finite values" begin
        z = zero(ExpScaled{Float64})
        @test iszero(z) && z.exp == 0
        @test z + ExpScaled(1.0e-300)^2 == ExpScaled(1.0e-300)^2
        @test signbit(ExpScaled(-0.0) + ExpScaled(-0.0))
        @test !signbit(ExpScaled(-0.0) + ExpScaled(0.0))
        @test ExpScaled(-0.0) == ExpScaled(0.0)
        @test ExpScaled(Inf) + ExpScaled(1.0e-300)^2 == ExpScaled(Inf)
        @test isnan(ExpScaled(Inf) - ExpScaled(Inf))
        @test isnan(ExpScaled(Inf) * z)
        @test ExpScaled(NaN) != ExpScaled(NaN)
        tiny, huge = ExpScaled(1.0e-300)^2, ExpScaled(1.0e300)^2
        @test -huge < -tiny < z < tiny < huge < ExpScaled(Inf)
        @test !(ExpScaled(NaN) < tiny) && !(tiny < ExpScaled(NaN))
        @test abs(-tiny) == tiny
        @test sort([huge, -tiny, z, tiny, -huge]) == [-huge, -tiny, z, tiny, huge]
    end

    @testset "conversion, promotion and hashing" begin
        @test ExpScaled(1.5) == 1.5
        @test isequal(ExpScaled(1.5), 1.5) && hash(ExpScaled(1.5)) == hash(1.5)
        tiny = ExpScaled(1.0e-300)^2
        @test hash(tiny) == hash(ExpScaled{Float64}(tiny.sig, tiny.exp))
        @test hash(ExpScaled{Float32}(0.5f0, -2000)) == hash(ExpScaled{Float64}(0.5, -2000))
        @test ExpScaled(2.0f0) + 1 isa ExpScaled{Float32}
        @test ExpScaled(2.0f0) + 1.0 isa ExpScaled{Float64}
        @test float(tiny) === tiny
        @test AbstractFloat(ExpScaled(2.0f0)) === 2.0f0
        @test ExpScaled{Float32}(tiny) == ExpScaled{Float32}(Float32(tiny.sig), tiny.exp)
        @test ldexp(ExpScaled(0.75), -5000) == ExpScaled(0.75, -5000)
        @test frexp(ExpScaled(0.75, -5000)) == (0.75, -5000)
        @test eval(Meta.parse(repr(tiny))) == tiny
    end

    @testset "promotes with BigFloat" begin
        @test promote_type(BigFloat, ExpScaled{Float64}) === ExpScaled{BigFloat}
        @test promote_type(ExpScaled{BigFloat}, BigFloat) === ExpScaled{BigFloat}
        @test big(2.0) * ExpScaled(0.5, -3000) == ExpScaled(big(0.5), -2999)
    end

    @testset "an exponent beyond Int is an error" begin
        top = ExpScaled(0.5, typemax(Int))
        @test_throws OverflowError top * top
        @test_throws OverflowError ExpScaled(0.5, typemin(Int)) / top
        @test_throws OverflowError ldexp(top, 2)
    end
end

@testset "a declared scale below the floating-point range" begin
    # Each block is off-diagonal-active and its value pair carries no ratio, so the split falls
    # back on the declared pair, whose scales lie below the smallest `Float64` but are still the
    # scales of live slots.
    tiny(e) = ExpScaled(0.5, e)
    for (c11, c12, c22, s1, s2) in (
            (0.0, 1.0, 0.0, tiny(-1300), ExpScaled(1.0)),
            (0.0, 0.998, 0.0, tiny(-1063), tiny(-2000)),
            (0.0, 0.998, 0.0, tiny(-2000), ExpScaled(2.1e-307)),
        )
        eta1, eta2 = split_block(1.0e-8, c11, c12, c22, SplitScale(0.0, s1), SplitScale(0.1, s2))
        @test isfinite(eta1) && eta1 > 0 && isfinite(eta2) && eta2 > 0
        @test eta1 * eta2 ≈ c12
        # The split equalizes the declared-scale-weighted entries, which the separate exponents
        # resolve although the product of the scales does not exist in `Float64`.
        @test eta1 * s1 ≈ eta2 * s2 rtol = 1.0e-12
    end
    # A pair whose equalizing point, `2^2500`, is outside the range takes the nearest pair inside
    # the window of `clamp_representable`.
    @test split_block(
        1.0e-8, 0.0, 1.0, 0.0, SplitScale(0.0, tiny(-5000)), SplitScale(0.0, ExpScaled(1.0))
    ) === (2.0^511, 2.0^-511)

    # Programs whose declared scales underflow or overflow `Float64` at a live slot.
    d3 = [1, 1, 4]
    tointgss(x) = ((10 // 1002) + x[3]^2) * (2 - exp(-(x[1] - x[2])^2 / (1 // 10 + x[3]^2)))
    programs = (
        ("underflowed factor", x -> (x[1] * 1.0e-200 * 1.0e-200) * x[2], [0.0, 1.0]),
        ("two underflowed factors", x -> (x[1] * 1.0e-200 * 1.0e-120) * (x[2] * 1.0e-200 * 1.0e-200), [0.0, 1.0]),
        ("two underflowed factors, swapped", x -> (x[1] * 1.0e-200 * 1.0e-120) * (x[2] * 1.0e-200 * 1.0e-200), [1.0, 0.0]),
        ("tointgss term at subnormal coordinates", y -> tointgss(y ./ d3), [1.0e-170, 0.0, 4 * 5.0e-324]),
    )
    @testset "$name" for (name, f, x0) in programs
        H = ForwardDiff.hessian(f, x0)
        hc = cover_hessian(f, x0)
        fc = forward_cover(f, x0)
        @test max_coverage_ratio(hc.h_sum, H) <= GATE
        @test max_coverage_ratio(fc.h_quad, H) <= GATE
        # Covariant under a dyadic rescaling of the inputs, with the declared scales alongside.
        D = [2.0^-3, 2.0^5, 2.0^7][eachindex(x0)]
        fD = y -> f(y .* D)
        @test cover_hessian(fD, x0 ./ D; scales = 1 ./ D).h_sum == hc.h_sum .* D
        @test forward_cover(fD, x0 ./ D; scales = 1 ./ D).h_quad == fc.h_quad .* D
    end

    # The declared scale of `u = x[1]*1e400` exceeds `floatmax`; the split still finds a finite
    # cover, and `h1*h2` reaches the off-diagonal entry `1e400`, which is `Inf` in `Float64`.
    fover = x -> (x[1] * 1.0e200 * 1.0e200) * x[2]
    hc = cover_hessian(fover, [0.0, 1.0])
    @test all(isfinite, hc.h_sum)
    @test big(hc.h_sum[1]) * big(hc.h_sum[2]) >= big(1.0e200)^2 * (1 - 1.0e-12)
    @test forward_cover(fover, [0.0, 1.0]).h_quad isa Vector{Float64}

    # A long product whose values underflow: the late multiplication blocks aim at an `eta` far
    # beyond `floatmax`, and several of them seed the same chain of the reverse sweep. In
    # `BigFloat` every split is representable, and rounding that cover gives the same result.
    fprod = x -> sum(x)^2 + (prod(x) - 1)^2
    x0 = [fill(1.0e-20, 59); 3.0]
    hc = cover_hessian(fprod, x0)
    fc = forward_cover(fprod, x0)
    @test hc.nguarded > 0 && fc.nguarded > 0
    @test hc.h_sum == Float64.(cover_hessian(fprod, big.(x0)).h_sum)
    @test fc.h_quad == Float64.(forward_cover(fprod, big.(x0)).h_quad)
    H = ForwardDiff.hessian(fprod, x0)
    @test max_coverage_ratio(hc.h_sum, H) <= GATE
    @test max_coverage_ratio(fc.h_quad, H) <= GATE
end
