# Tests for vector-output recording: `tape_function` on programs returning a vector of TVars, and
# the vector-mode signed `adjoints` that reads the Jacobian off such a tape. Ground truth is
# `ForwardDiff.jacobian`. The two things worth pinning are that the Jacobian matches on both
# scalarized and array-native (dot/linear-node) programs, and that repeated or input-node outputs
# are handled as distinct rows rather than collapsed.

@testset "vector output" begin

    # Jacobian read off the tape against ForwardDiff, over programs that exercise scalar elementals,
    # array-native matvec rows (dot nodes), constant-matrix rows (linear nodes), and outputs that
    # are inputs or duplicates.
    W = [0.5 -1.0 2.0; 1.5 0.0 -0.5; -1.0 1.0 1.0; 0.25 0.5 -2.0]
    @testset "adjoints Jacobian matches ForwardDiff — $name" for (name, f, x0) in [
        ("scalarized",
         xs -> [xs[1] * xs[2], sin(xs[3]), xs[1] + xs[3], exp(xs[2])],
         [0.7, -0.4, 1.1]),
        ("array-native matvec", xs -> W * xs, [0.7, -0.4, 1.1]),
        ("nonlinear rows", xs -> [tanh(y) for y in W * xs], [0.3, -0.2, 0.5]),
        ("identity", xs -> xs, [0.7, -0.4, 1.1]),
        ("duplicated and input outputs",
         xs -> [xs[1], xs[1], xs[2] * xs[3], xs[3]],
         [0.7, -0.4, 1.1]),
    ]
        t, outidxs = tape_function(f, x0)
        @test outidxs isa Vector{Int}
        m = length(f([1.0, 1.0, 1.0] .* x0))   # number of outputs, independent of the tape
        @test length(outidxs) == m

        Vbar = adjoints(t, outidxs)
        @test size(Vbar) == (m, nnodes(t))
        J = Vbar[:, 1:t.ninputs]
        @test J ≈ ForwardDiff.jacobian(f, x0)
    end

    @testset "duplicated outputs are distinct rows" begin
        f = xs -> (y = xs[1] * xs[2]; [y, y, xs[3]])
        x0 = [2.0, -3.0, 0.5]
        t, outidxs = tape_function(f, x0)
        @test outidxs[1] == outidxs[2]            # one node placed at two output slots
        J = adjoints(t, outidxs)[:, 1:t.ninputs]
        @test J[1, :] == J[2, :]                  # identical rows, not merged away
        @test J ≈ ForwardDiff.jacobian(f, x0)
    end

    @testset "input-node output gives a unit row" begin
        f = xs -> [xs[2], sin(xs[1])]
        x0 = [0.6, 1.3]
        t, outidxs = tape_function(f, x0)
        @test outidxs[1] == 2                     # the output is the second input node itself
        J = adjoints(t, outidxs)[:, 1:t.ninputs]
        @test J[1, :] == [0.0, 1.0]
    end

    # The production merges seed `abs_reverse` at the output nodes rather than building the signed
    # Jacobian. The m-channel absolute sweep yields the surrogate Ũ that bounds |J| entrywise, the
    # first-order object the cover merges propagate.
    @testset "output-seeded abs_reverse bounds the Jacobian" begin
        f = xs -> [sin(xs[1] * xs[2]), xs[1] + 2 * xs[3], exp(xs[2]) - xs[3]]
        x0 = [0.5, -0.7, 0.9]
        t, outidxs = tape_function(f, x0)
        m, n, N = length(outidxs), t.ninputs, nnodes(t)

        Theta = zeros(m, N)
        for k in 1:m
            Theta[k, outidxs[k]] += 1.0
        end
        Utilde = abs_reverse(t, Theta)[:, 1:n]
        J = ForwardDiff.jacobian(f, x0)
        @test all(Utilde .>= abs.(J) .- 1e-12)
        # No path cancellation in these residuals, so the absolute sweep is tight to |J|.
        @test Utilde ≈ abs.(J)
    end

    @testset "scalar output path is unchanged" begin
        f = xs -> xs[1] * xs[2] + exp(xs[3])
        x0 = [2.0, -3.0, 0.5]
        t, outidx = tape_function(f, x0)
        @test outidx isa Int
        @test outidx == nnodes(t)
    end

    # A constant output is a fixed offset of the program, legitimate in a least-squares model: it
    # gets a node of its own, and hence a zero Jacobian row, which the asymmetric cover already
    # handles exactly (`a_k = 0`).
    @testset "constant outputs get a node and a zero row" begin
        f = xs -> [one(eltype(xs)), xs[1] * xs[2], eltype(xs)(2.5), sin(xs[3])]
        x0 = [0.7, -0.4, 1.1]
        t, outidxs = tape_function(f, x0)
        @test length(outidxs) == 4
        @test t.vals[outidxs[1]] == 1.0 && t.vals[outidxs[3]] == 2.5
        @test isempty(t.parents[outidxs[1]]) && isempty(t.parents[outidxs[3]])

        J = adjoints(t, outidxs)[:, 1:t.ninputs]
        @test J ≈ ForwardDiff.jacobian(y -> [1.0, y[1] * y[2], 2.5, sin(y[3])], x0)
        @test all(iszero, J[1, :]) && all(iszero, J[3, :])
        # The constant rows carry no absolute sensitivity either, so they cost the cover nothing.
        Utilde = jacobian_surrogate(t, outidxs)
        @test all(iszero, Utilde[1, :]) && all(iszero, Utilde[3, :])

        bal = balance_cover(Utilde; arule=:geomean)
        @test bal.a[1] == 0 && bal.a[3] == 0
        @test jacobian_ratio(Utilde, bal.a, bal.b) <= GATE
    end

    @testset "malformed vector outputs fail fast" begin
        x0 = [1.0, 2.0]
        @test_throws "returned no outputs" tape_function(xs -> TVar[], x0)
        # Depending on no input at all is a bug, not an offset.
        @test_throws "every program output is a constant" tape_function(xs -> [one(eltype(xs))], x0)
        @test_throws "only TVars" tape_function(xs -> [xs[1], 1.0], x0)
        @test_throws "not a TVar or a vector of TVars" tape_function(xs -> (xs[1], xs[2]), x0)
        @test_throws "different tape" tape_function(x0) do xs
            t2, o2 = tape_function(xs -> xs[1] * xs[2], x0)
            [xs[1], TVar(t2, o2)]
        end
    end

end
