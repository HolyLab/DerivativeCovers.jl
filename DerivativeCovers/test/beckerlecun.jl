# The Becker–LeCun estimate must agree with the diagonal of the Hessian wherever its certificate
# claims exactness, and the certificate must flag the programs whose dropped terms are nonzero.

using IntervalArithmetic: Interval, interval, in_interval

# A sum of window terms `s * (s * (s^2 - 20) - 1/10)` in which the window sum `s` is recorded
# three separate times, so each variable reaches the term along three paths that meet in products.
function curly_like(x; k = 2)
    n = length(x)
    acc = zero(eltype(x))
    for i in eachindex(x)
        w = i:min(i + k, n)
        s1, s2, s3 = sum(x[w]), sum(x[w]), sum(x[w])
        acc += s1 * (s2 * (s3^2 - 20) - 1 // 10)
    end
    return acc
end

# A sum of squared residuals in which every variable enters several residuals.
function shared_least_squares(x)
    acc = zero(eltype(x))
    for i in firstindex(x):(lastindex(x) - 1)
        acc += (x[i] * x[i + 1] - 1)^2 + (exp(x[i]) - x[i + 1])^2 + (sin(x[i]) + 2x[i + 1]^2)^2
    end
    return acc
end

@testset "beckerlecun.jl" begin

    @testset "examples with known diagonals" begin
        x, y = 0.7, -1.3
        r = becker_lecun(v -> (v[1] - 1)^2 + (v[1] * v[2])^2, [x, y]; exactness = true)
        @test r.diag ≈ [2 + 2y^2, 2x^2] rtol = 1.0e-14
        @test r.exact == [true, true]

        # The two children of `x`, `x` itself and `x^2`, meet before the square.
        r = becker_lecun(v -> (v[1] + v[1]^2)^2, [0.4]; exactness = true)
        @test r.diag ≈ [5.52] rtol = 1.0e-14
        @test ForwardDiff.hessian(v -> (v[1] + v[1]^2)^2, [0.4]) ≈ [8.72;;] rtol = 1.0e-14
        @test r.exact == [false]

        # A parent in both slots of a product: the slots' partials add before squaring.
        r = becker_lecun(v -> (v[1] * v[1])^2, [0.9]; exactness = true)
        @test r.diag ≈ [12 * 0.9^2] rtol = 1.0e-14
        @test r.exact == [true]

        # At the origin, `x + y` has a zero adjoint and still passes on the curvature of its square.
        f0(v) = (v[1] + v[2])^2 / 2 + (v[2]^2 - 1)^2 / 4
        r = becker_lecun(f0, [0.0, 0.0]; exactness = true)
        @test r.diag == [1.0, 0.0]
        @test r.exact == [true, true]
    end

    @testset "exactness is reported per coordinate" begin
        f(v) = (v[1] - exp(v[1]))^2 + (v[2] - v[1])^2
        x0 = [0.3, 0.8]
        r = becker_lecun(f, x0; exactness = true)
        H = ForwardDiff.hessian(f, x0)
        @test r.exact == [false, true]
        @test r.diag[2] ≈ H[2, 2] rtol = 1.0e-14
        @test !isapprox(r.diag[1], H[1, 1]; rtol = 1.0e-3)
    end

    @testset "reconvergent window sums are flagged" begin
        x0 = [0.1 * cos(i) for i in 1:8]
        r = becker_lecun(curly_like, x0; exactness = true)
        H = ForwardDiff.hessian(curly_like, x0)
        @test !any(r.exact)
        @test maximum(abs, r.diag - diag(H)) > 1.0e-3 * maximum(abs, diag(H))
    end

    @testset "least squares with shared variables is certified" begin
        x0 = [0.2 * i - 0.7 for i in 1:6]
        r = becker_lecun(shared_least_squares, x0; exactness = true)
        @test all(r.exact)
        @test r.diag ≈ diag(ForwardDiff.hessian(shared_least_squares, x0)) rtol = 1.0e-12
    end

    @testset "certified coordinates are exact — $name" for (name, f, x0) in PROBLEMS
        t, outidx = tape_function(f, x0)
        r = becker_lecun(t, outidx; exactness = true)
        H = ForwardDiff.hessian(f, x0)
        @test r.grad == adjoints(t, outidx)[1:t.ninputs]
        for i in eachindex(x0)
            r.exact[i] || continue
            @test r.diag[i] ≈ H[i, i] atol = 1.0e-12 * maximum(abs, H[i, :]) rtol = 1.0e-12
        end
        rr = becker_lecun(t, outidx)
        @test rr.exact === nothing
        @test rr.diag == r.diag
    end

    @testset "tree-structured programs are certified throughout" begin
        r = becker_lecun(tree_program, TREE_X; exactness = true)
        @test all(r.exact)
        @test r.diag ≈ diag(ForwardDiff.hessian(tree_program, TREE_X)) rtol = 1.0e-12
    end

    @testset "inner products, including a parent in paired slots" begin
        A = [2.0 0.5 0.0; 0.5 1.0 -0.3; 0.0 -0.3 3.0]
        for f in (v -> dot(v, v)^2, v -> dot(v, A * v), v -> exp(dot(v, v)))
            x0 = [0.4, -0.2, 0.9]
            r = becker_lecun(f, x0; exactness = true)
            H = ForwardDiff.hessian(f, x0)
            for i in eachindex(x0)
                r.exact[i] && @test r.diag[i] ≈ H[i, i] rtol = 1.0e-12
            end
        end
    end

    @testset "over a box, certified entries enclose the diagonal" begin
        boxes = [
            (v -> (v[1] - 1)^2 + (v[1] * v[2])^2, [0.5, -1.5], [0.9, -1.1]),
            (v -> (v[1] - exp(v[1]))^2 + (v[2] - v[1])^2, [0.2, 0.6], [0.4, 0.9]),
            (shared_least_squares, collect(range(-0.8, 0.3, 5)), collect(range(-0.7, 0.4, 5))),
        ]
        rng = Xoshiro(20261114)
        for (f, lo, hi) in boxes
            rI = becker_lecun(f, interval.(lo, hi); exactness = true)
            @test eltype(rI.diag) <: Interval
            for _ in 1:50
                x = lo .+ rand(rng, length(lo)) .* (hi .- lo)
                r = becker_lecun(f, x; exactness = true)
                H = ForwardDiff.hessian(f, x)
                @test all(in_interval.(r.diag, rI.diag))
                @test r.exact == rI.exact
                @test all(in_interval(H[i, i], rI.diag[i]) for i in eachindex(x) if rI.exact[i])
            end
        end
    end

    @testset "a declared sub-map is refused" begin
        g(v) = sum(submap(w -> [w[1] * exp(w[2])], v)) + v[1]^2
        @test_throws "records no signed second partials" becker_lecun(g, [0.5, 0.2])
    end

    @testset "show" begin
        r = becker_lecun(v -> (v[1] + v[1]^2)^2 + v[2]^2, [0.4, 1.0]; exactness = true)
        @test sprint(show, r) == "BeckerLeCunEstimate(2 inputs)"
        @test sprint(show, MIME("text/plain"), r) ==
            "BeckerLeCunEstimate: 2 inputs; grad, diag, exact (1 of 2 certified)"
        rr = becker_lecun(v -> v[1]^2, [1.0])
        @test sprint(show, MIME("text/plain"), rr) ==
            "BeckerLeCunEstimate: 1 inputs; grad, diag, exact (not checked)"
    end
end
