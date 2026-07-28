# Tests for the first-order matrix-free products (`jvp`, `vjp`) and the two-sided equilibration
# built on them. Ground truth for the products is `ForwardDiff.jacobian`; the properties worth
# pinning for the equilibration are that a valid cover maps to entries bounded by 1, that the
# matrix-free singular value approaches the dense one strictly *from below* (so an unguarded shrink
# would be invalid), and that the shrink inherits validity.

@testset "equilibration" begin

    W = [0.5 -1.0 2.0; 1.5 0.0 -0.5; -1.0 1.0 1.0; 0.25 0.5 -2.0]
    programs = [
        ("scalarized",
         xs -> [xs[1] * xs[2], sin(xs[3]), xs[1] + xs[3], exp(xs[2])],
         [0.7, -0.4, 1.1]),
        ("array-native matvec", xs -> W * xs, [0.7, -0.4, 1.1]),
        ("nonlinear rows", xs -> [tanh(y) for y in W * xs], [0.3, -0.2, 0.5]),
        ("duplicated and input outputs",
         xs -> [xs[1], xs[1], xs[2] * xs[3], xs[3]],
         [0.7, -0.4, 1.1]),
    ]

    @testset "jvp and vjp match ForwardDiff — $name" for (name, f, x0) in programs
        t, outidxs = tape_function(f, x0)
        J = ForwardDiff.jacobian(f, x0)
        rng = MersenneTwister(11)
        v = randn(rng, length(x0))
        u = randn(rng, length(outidxs))

        @test jvp(t, outidxs, v) ≈ J * v
        @test vjp(t, outidxs, u) ≈ J' * u
        # The two are transposes of one another, whatever the tape does internally.
        @test dot(u, jvp(t, outidxs, v)) ≈ dot(vjp(t, outidxs, u), v)

        # The function form retapes and must agree with the tape form.
        @test jvp(f, x0, v) ≈ J * v
        @test vjp(f, x0, u) ≈ J' * u

        # The in-place forms write the caller's destination and return it. `Jv` is indexed by
        # the outputs, `Jtu` by the inputs, so the two destinations differ in length whenever
        # the program is not square.
        Jv, Jtu = fill(NaN, length(outidxs)), fill(NaN, length(x0))
        @test jvp!(Jv, t, outidxs, v) === Jv
        @test vjp!(Jtu, t, outidxs, u) === Jtu
        @test Jv == jvp(t, outidxs, v)
        @test Jtu == vjp(t, outidxs, u)
    end

    @testset "duplicated outputs accumulate on vjp" begin
        # Two output slots naming one node: jvp repeats the entry, so vjp must sum the seeds.
        f = xs -> (y = xs[1] * xs[2]; [y, y, xs[3]])
        x0 = [2.0, -3.0, 0.5]
        t, outidxs = tape_function(f, x0)
        @test outidxs[1] == outidxs[2]
        @test jvp(t, outidxs, [1.0, 0.0, 0.0])[1] == jvp(t, outidxs, [1.0, 0.0, 0.0])[2]
        @test vjp(t, outidxs, [1.0, 1.0, 0.0]) ≈ 2 .* vjp(t, outidxs, [1.0, 0.0, 0.0])
    end

    @testset "product argument checks" begin
        t, outidxs = tape_function(xs -> [xs[1] * xs[2], xs[2]], [1.0, 2.0])
        @test_throws "expected 2 (one per input)" jvp(t, outidxs, [1.0, 2.0, 3.0])
        @test_throws "expected 2 (one per output)" vjp(t, outidxs, [1.0])
        # A destination of the wrong shape is caught before the sweep runs.
        @test_throws DimensionMismatch jvp!(zeros(3), t, outidxs, [1.0, 2.0])
        @test_throws DimensionMismatch vjp!(zeros(3), t, outidxs, [1.0, 2.0])
    end

    @testset "equilibrate normalizes a valid cover" begin
        f, x0 = programs[1][2], programs[1][3]
        t, outidxs = tape_function(f, x0)
        J = ForwardDiff.jacobian(f, x0)
        c = cover_jacobian(t, outidxs)

        for b in (c.b_sum, c.b_quad, c.b_max)
            Jhat = equilibrate(J, c.a, b)
            @test size(Jhat) == size(J)
            @test maximum(abs, Jhat) <= GATE            # validity, restated on the scaled matrix
            @test Jhat .* (c.a * b') ≈ J                # the scaling is exactly undone
        end
        # b_max is the minimal b for this a, so some entry of each column saturates the bound.
        Jmax = equilibrate(J, c.a, c.b_max)
        @test all(j -> maximum(abs, Jmax[:, j]) ≈ 1, axes(Jmax, 2))
    end

    @testset "equilibrate zero-weight handling" begin
        # A weight may vanish only where the row or column it scales does.
        J = [1.0 0.0; 0.0 0.0]
        @test equilibrate(J, [1.0, 0.0], [1.0, 0.0]) == [1.0 0.0; 0.0 0.0]
        @test scaled_singular_value(J, [1.0, 0.0], [1.0, 0.0]) ≈ 1
        @test_throws "a[1] is zero but row 1" equilibrate(J, [0.0, 1.0], [1.0, 1.0])
        @test_throws "b[1] is zero but column 1" equilibrate(J, [1.0, 1.0], [0.0, 1.0])
        @test scaled_singular_value(zeros(2, 2), zeros(2), zeros(2)) == 0.0
        @test_throws DimensionMismatch equilibrate(J, [1.0, 1.0, 1.0], [1.0, 1.0])
    end

    @testset "matrix-free singular value converges from below — $name" for (name, f, x0) in programs
        t, outidxs = tape_function(f, x0)
        J = ForwardDiff.jacobian(f, x0)
        c = cover_jacobian(t, outidxs)
        b = max.(c.b_sum, eps())            # the power iteration needs strict positivity

        exact = scaled_singular_value(J, c.a, b)
        est = scaled_singular_value(t, outidxs, c.a, b; rtol=1e-10, maxiter=2000)
        # Every iterate is norm(Jhat*v) at a unit v, hence a lower bound. This direction is what
        # forces `safety > 1` in `jacobian_shrink`; an estimate from above would need no guard.
        @test est <= exact * GATE
        @test est ≈ exact rtol = 1e-6
        # The spectral norm dominates every entry, which is the whole basis of the shrink.
        @test exact >= maximum(abs, equilibrate(J, c.a, b)) / GATE
    end

    @testset "matrix-free singular value argument checks" begin
        f, x0 = programs[1][2], programs[1][3]
        t, outidxs = tape_function(f, x0)
        c = cover_jacobian(t, outidxs)
        @test_throws "strictly positive" scaled_singular_value(t, outidxs, c.a, zeros(3))
        @test_throws "expected 3 (one per input)" scaled_singular_value(t, outidxs, c.a, ones(2))
        @test_throws "expected 4 (one per output)" scaled_singular_value(t, outidxs, ones(2), c.b_max)
        # Non-convergence is an error, never a silently returned partial estimate.
        @test_throws "did not converge" scaled_singular_value(t, outidxs, c.a, c.b_max;
                                                              rtol=1e-15, maxiter=2)
    end

    @testset "jacobian_shrink" begin
        f = xs -> [tanh(y) for y in W * xs]
        x0 = [0.3, -0.2, 0.5]
        t, outidxs = tape_function(f, x0)
        J = ForwardDiff.jacobian(f, x0)
        c = cover_jacobian(t, outidxs)
        a, b = c.a, c.b_max

        # An entrywise-minimal cover saturates an entry, so sigma >= 1 and the clamp makes the
        # shrink a no-op rather than inflating a cover that has no spectral slack.
        sigma = scaled_singular_value(J, a, b)
        @test sigma >= 1
        @test jacobian_shrink(a, b, sigma) == (a, b)

        # Uniform inflation is pure spectral slack, and the shrink removes it: one pass lands on
        # sigma == 1 exactly, from either starting inflation, and a second pass changes nothing.
        for factor in (2.0, 5.0)
            ai, bi = a .* factor, b .* factor
            si = scaled_singular_value(J, ai, bi)
            @test si ≈ sigma / factor^2
            as, bs = jacobian_shrink(ai, bi, si)
            @test all(abs.(J) .<= (as * bs') .* GATE)         # validity is inherited
            @test all(as .<= ai) && all(bs .<= bi)            # and it is a strict improvement
            @test scaled_singular_value(J, as, bs) ≈ 1
            @test jacobian_shrink(as, bs, scaled_singular_value(J, as, bs)) == (as, bs)
        end

        # The slack is split evenly, so the product a*b' carries exactly one factor of it.
        as, bs = jacobian_shrink(a, b, 0.25)
        @test as * bs' ≈ 0.25 .* (a * b')

        @test_throws "sigma must be nonnegative" jacobian_shrink(a, b, -1.0)
        @test_throws "safety must be positive" jacobian_shrink(a, b, 0.5; safety=0.0)
    end

end
