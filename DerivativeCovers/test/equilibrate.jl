# Compare matrix-free Jacobian products with ForwardDiff. Equilibration tests verify entrywise
# bounds, convergence of the singular-value estimate from below, and validity after rescaling by it.

@testset "equilibration" begin

    W = [0.5 -1.0 2.0; 1.5 0.0 -0.5; -1.0 1.0 1.0; 0.25 0.5 -2.0]
    programs = [
        (
            "scalarized",
            xs -> [xs[1] * xs[2], sin(xs[3]), xs[1] + xs[3], exp(xs[2])],
            [0.7, -0.4, 1.1],
        ),
        ("array-native matvec", xs -> W * xs, [0.7, -0.4, 1.1]),
        ("nonlinear rows", xs -> [tanh(y) for y in W * xs], [0.3, -0.2, 0.5]),
        (
            "duplicated and input outputs",
            xs -> [xs[1], xs[1], xs[2] * xs[3], xs[3]],
            [0.7, -0.4, 1.1],
        ),
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
        @test_throws "u and outidxs must share indices" vjp(t, outidxs, [1.0])
        # A destination of the wrong shape is caught before the sweep runs.
        @test_throws DimensionMismatch jvp!(zeros(3), t, outidxs, [1.0, 2.0])
        @test_throws DimensionMismatch vjp!(zeros(3), t, outidxs, [1.0, 2.0])
    end

    @testset "equilibrate normalizes a valid cover" begin
        f, x0 = programs[1][2], programs[1][3]
        t, outidxs = tape_function(f, x0)
        J = ForwardDiff.jacobian(f, x0)
        c = cover_jacobian(WithDiagnostics(), t, outidxs)

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

        # The empty-support branch carries the element type of the computation, as the
        # normal path does.
        J32, a32, b32 = Float32[1 2; 3 4], ones(Float32, 2), ones(Float32, 2)
        @test scaled_singular_value(J32, a32, b32) isa Float32
        @test scaled_singular_value(zeros(Float32, 2, 2), zeros(Float32, 2), b32) isa Float32
        H32, h32 = Float32[1 0; 0 1], ones(Float32, 2)
        @test scaled_spectral_radius(H32, h32) isa Float32
        @test scaled_spectral_radius(zeros(Float32, 2, 2), zeros(Float32, 2)) isa Float32

        # The matrix-free path returns early when the Hessian vanishes on the whole support.
        tlin, olin = tape_function(xs -> sum(xs), [1.0, 2.0, 3.0])
        @test scaled_spectral_radius(tlin, olin, ones(3)) === 0.0
    end

    @testset "matrix-free singular value converges from below — $name" for (name, f, x0) in programs
        t, outidxs = tape_function(f, x0)
        J = ForwardDiff.jacobian(f, x0)
        c = cover_jacobian(WithDiagnostics(), t, outidxs)
        b = max.(c.b_sum, eps())            # the power iteration needs strict positivity

        exact = scaled_singular_value(J, c.a, b)
        est = scaled_singular_value(t, outidxs, c.a, b; rtol = 1.0e-10, maxiter = 2000)
        # Every iterate is norm(Jhat*v) at a unit v, hence a lower bound, so rescaling a cover by
        # the estimate needs a safety factor above 1.
        @test est <= exact * GATE
        @test est ≈ exact rtol = 1.0e-6
        # The spectral norm dominates every entry, which is what makes rescaling by it valid.
        @test exact >= maximum(abs, equilibrate(J, c.a, b)) / GATE
    end

    @testset "matrix-free singular value argument checks" begin
        f, x0 = programs[1][2], programs[1][3]
        t, outidxs = tape_function(f, x0)
        c = cover_jacobian(WithDiagnostics(), t, outidxs)
        @test_throws "strictly positive" scaled_singular_value(t, outidxs, c.a, zeros(3))
        @test_throws "expected 3 (one per input)" scaled_singular_value(t, outidxs, c.a, ones(2))
        @test_throws "a and outidxs must share indices" scaled_singular_value(
            t, outidxs, ones(2),
            c.b_max
        )
        # Non-convergence is an error, never a silently returned partial estimate.
        @test_throws "did not converge" scaled_singular_value(
            t, outidxs, c.a, c.b_max;
            rtol = 1.0e-15, maxiter = 2
        )
    end

    @testset "rescaling a cover by its singular value" begin
        f = xs -> [tanh(y) for y in W * xs]
        x0 = [0.3, -0.2, 0.5]
        t, outidxs = tape_function(f, x0)
        J = ForwardDiff.jacobian(f, x0)
        c = cover_jacobian(WithDiagnostics(), t, outidxs)
        a, b = c.a, c.b_max

        # An entrywise-minimal cover saturates an entry, so it has no spectral slack.
        sigma = scaled_singular_value(J, a, b)
        @test sigma >= 1

        # Uniform inflation is pure spectral slack, and rescaling both factors by `sqrt(sigma)`
        # removes it: one pass lands on sigma == 1 from either starting inflation.
        for factor in (2.0, 5.0)
            ai, bi = a .* factor, b .* factor
            si = scaled_singular_value(J, ai, bi)
            @test si ≈ sigma / factor^2
            as, bs = sqrt(si) .* ai, sqrt(si) .* bi
            @test all(abs.(J) .<= (as * bs') .* GATE)
            @test scaled_singular_value(J, as, bs) ≈ 1
        end
    end

end
