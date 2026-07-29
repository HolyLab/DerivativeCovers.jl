# Values of the element type built from numbers. `one(eltype(x))`, `eltype(x)(c)` and the
# containers they populate are the generic-code idioms for typed constants, so a program using
# them must tape and cover; a constant TVar is what makes that possible.

@testset "constants.jl" begin

    @testset "typed literals and conversions are constants" begin
        for c in (one(TVar), zero(TVar), oneunit(TVar), TVar(2.5), TVar(3), convert(TVar, 2.5))
            @test c isa TVar
            @test isconstant(c)
        end
        @test value(one(TVar)) == 1
        @test value(zero(TVar)) == 0
        @test value(TVar(2.5)) == 2.5

        # `TVar` on a TVar is the identity in both states, so generic code that asks for the
        # element type of something already of that type neither copies nor loses the tape.
        @test TVar(TVar(2.5)) === TVar(2.5)

        t = Tape([2.5])
        x = TVar(t, 1)
        @test TVar(x) === x
        @test !isconstant(x)
        @test value(x) == 2.5
        @test isconstant(zero(x)) && value(zero(x)) == 0
        @test isconstant(one(x)) && value(one(x)) == 1
        @test isconstant(oneunit(x)) && value(oneunit(x)) == 1
        # No node was recorded for any constant.
        @test nnodes(t) == 1
    end

    @testset "constants are closed under the elementals" begin
        a, b = TVar(3.0), TVar(0.5)
        for (op, expect) in ((+, 3.5), (-, 2.5), (*, 1.5), (/, 6.0), (^, sqrt(3.0)))
            r = op(a, b)
            @test isconstant(r)
            @test value(r) ≈ expect
        end
        for op in (exp, log, sin, cos, tan, atan, tanh, sqrt, inv, abs2, -)
            @test isconstant(op(a)) && value(op(a)) ≈ op(3.0)
        end
        @test isconstant(a + 1) && value(a + 1) == 4
        @test isconstant(2 * a) && value(2 * a) == 6
        @test isconstant(2^b) && value(2^b) ≈ sqrt(2)
        # Nothing above needs a tape, and none of it can be a program output.
        @test_throws "does not depend on the inputs" tape_function(x -> one(eltype(x)), [1.0])
    end

    @testset "a constant enters a node as a coefficient, not a parent" begin
        # Recording against a constant gives the same tape as against the plain number: the
        # constant folds into a unary node rather than becoming a second parent.
        withconst = x -> TVar(3.0) * x[1] + x[2]^2
        withnumber = x -> 3.0 * x[1] + x[2]^2
        x0 = [1.5, -2.0]
        tc, _ = tape_function(withconst, x0)
        tn, _ = tape_function(withnumber, x0)
        @test nnodes(tc) == nnodes(tn)
        @test cover_hessian(WithDiagnostics(), withconst, x0).h_sum ==
              cover_hessian(WithDiagnostics(), withnumber, x0).h_sum
        # A constant reaching a node builder directly is a caller bug, and says so.
        @test_throws "no node on any tape" DerivativeCovers.noderef(TVar(1.0))
    end

    @testset "objective seeded with one(eltype(x))" begin
        f = function (x)
            s = one(eltype(x))
            z = zero(eltype(x))
            return (x[1] + 2s)^2 + (x[2] - s)^4 + z * x[1]
        end
        x0 = [3.0, 4.0]
        r = cover_hessian(WithDiagnostics(), f, x0)
        @test r.grad ≈ ForwardDiff.gradient(f, x0) rtol = 1e-8
        H = ForwardDiff.hessian(f, x0)
        for h in (r.h_sum, r.h_quad)
            @test all(abs.(H) .<= h * h' .* GATE)
        end
    end

    @testset "objective converting a constant to the element type" begin
        # The `eltype(x)(c)` idiom, as `biggs6` and `clplatea` write it.
        f = x -> sum(exp(-x[1] * i / 10) for i = 1:4) - eltype(x)(2.75) * x[2]^2
        x0 = [0.4, 1.25]
        r = cover_hessian(WithDiagnostics(), f, x0)
        @test r.grad ≈ ForwardDiff.gradient(f, x0) rtol = 1e-8
        H = ForwardDiff.hessian(f, x0)
        @test all(abs.(H) .<= r.h_sum * r.h_sum' .* GATE)
    end

    @testset "objective building a container of the element type" begin
        # The `eltype(x)[...]` idiom, as `penalty2` writes it: a vector of constants indexed
        # against the variables.
        f = function (x)
            y = eltype(x)[exp(i // 10) for i = 1:3]
            return sum((exp(x[i] / 10) - y[i])^2 for i = 1:3)
        end
        x0 = [0.3, -0.2, 0.7]
        r = cover_hessian(WithDiagnostics(), f, x0)
        @test r.grad ≈ ForwardDiff.gradient(f, x0) rtol = 1e-8
        H = ForwardDiff.hessian(f, x0)
        for h in (r.h_sum, r.h_quad)
            @test all(abs.(H) .<= h * h' .* GATE)
        end
    end

    @testset "an all-constant operand records as a linear node" begin
        # A constant-filled Vector{TVar} is a coefficient vector: the inner product carries no
        # second-order term, exactly as it would for a Vector{Float64}.
        wc = x -> dot(eltype(x)[2.0, -3.0, 0.5], x)
        wn = x -> dot([2.0, -3.0, 0.5], x)
        x0 = [1.0, 2.0, 3.0]
        tc, oc = tape_function(wc, x0)
        @test nnodes(tc) == nnodes(first(tape_function(wn, x0)))
        @test isempty(tc.Cs[oc])
    end

    @testset "an operand mixing the two states is split by pair" begin
        # Only the pairs recording on both sides reach an inner-product node; a constant side is a
        # coefficient of a linear node. Parity with the scalarized product is in primitives.jl.
        x0 = [1.0, 2.0, 3.0]
        mixed = x -> dot([TVar(2.0), x[2], TVar(0.5)], x)
        t, o = tape_function(mixed, x0)
        @test t.vals[o] ≈ 2.0 * 1.0 + 2.0 * 2.0 + 0.5 * 3.0
        @test count(C -> C isa DerivativeCovers.PairedPartials, t.Cs) == 1

        # A pair constant on both sides is an additive offset.
        withoffset = x -> dot([TVar(2.0), x[1]], [TVar(3.0), x[2]])
        t2, o2 = tape_function(withoffset, x0)
        @test t2.vals[o2] ≈ 6.0 + 1.0 * 2.0

        # An all-constant product has nothing to record and is itself a constant.
        s = dot(TVar[TVar(2.0), TVar(3.0)], TVar[TVar(4.0), TVar(5.0)])
        @test isconstant(s) && value(s) == 23
    end

    @testset "a sum over constants and variables keeps the offset" begin
        f = x -> sum([x[1]^2, TVar(4.0), x[2]^2, TVar(-1.5)])
        x0 = [2.0, 3.0]
        t, outidx = tape_function(f, x0)
        @test t.vals[outidx] ≈ 2.0^2 + 3.0^2 + 2.5
        r = cover_hessian(WithDiagnostics(), f, x0)
        @test r.grad ≈ [4.0, 6.0]
        H = ForwardDiff.hessian(y -> y[1]^2 + y[2]^2 + 2.5, x0)
        @test all(abs.(H) .<= r.h_sum * r.h_sum' .* GATE)
        # All-constant: nothing to record, so the sum is itself a constant.
        s = sum([TVar(1.0), TVar(2.0)])
        @test isconstant(s) && value(s) == 3
    end
end
