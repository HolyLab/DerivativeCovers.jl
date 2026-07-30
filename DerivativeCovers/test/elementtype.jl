# The tape's element type is the working precision of every construction built on it: a Float32
# starting point yields Float32 covers, and a Float64 literal inside the program or a Float64 weight
# passed to a cover function is converted into the tape's type rather than widening the result.

# Every field of a result carries the tape's element type.
fieldeltypes(r) = unique(eltype(getfield(r, f)) for f in fieldnames(typeof(r)) if getfield(r, f) isa AbstractArray)

@testset "elementtype.jl" begin

    f32(x) = exp(x[1]) + x[1] * x[2]^2
    x032 = Float32[0.5, 2.0]
    x064 = Float64.(x032)

    @testset "the starting point sets the element type" begin
        @test eltype(Tape(x032)) === Float32
        @test eltype(Tape(x064)) === Float64
        # An integer starting point has no precision of its own and floats.
        @test eltype(Tape([1, 2])) === Float64
        # An explicit parameter overrides the starting point's own type.
        @test eltype(Tape{Float32}(x064)) === Float32
        @test eltype(Tape{Float64}(x032)) === Float64

        t = Tape(x032)
        @test t isa Tape{Float32}
        @test eltype(t.vals) === Float32
        @test eltype(eltype(t.ds)) === Float32
        x = TVar(t, 1)
        @test x isa TVar{Float32}
        @test value(x) === 0.5f0
    end

    @testset "recorded nodes stay in the tape's type" begin
        t, outidx = tape_function(f32, x032)
        @test t isa Tape{Float32}
        @test eltype(t.vals) === Float32
        @test all(d -> eltype(d) === Float32, t.ds)
        # The curvature forms the sweeps resolve statically hold the same element type; without
        # this the dispatch ladder falls through to its open branch on every node.
        @test any(C -> C isa DerivativeCovers.UnaryCurvature{Float32}, t.Cs)
        @test any(C -> C isa DerivativeCovers.BinaryCurvature{Float32}, t.Cs)
        @test all(t.Cs) do C
            C isa DerivativeCovers.NoCurvature ||
                C isa DerivativeCovers.UnaryCurvature{Float32} ||
                C isa DerivativeCovers.BinaryCurvature{Float32}
        end

        # A Float64 literal in the program is a coefficient, converted on the way onto the tape.
        g(x) = 2.5 * exp(x[1]) - 1.0 + x[1] * x[2]^2 + 3
        tg, _ = tape_function(g, x032)
        @test tg isa Tape{Float32}
        @test eltype(tg.vals) === Float32
        @test all(d -> eltype(d) === Float32, tg.ds)
    end

    @testset "generic typed-literal idioms follow the element type" begin
        # The shapes generic numerical code uses to obtain values of the element type.
        t = Tape(x032)
        x = [TVar(t, i) for i in 1:t.ninputs]
        @test x isa Vector{TVar{Float32}}
        @test zero(eltype(x)) isa TVar{Float32}
        @test one(eltype(x)) isa TVar{Float32}
        @test oneunit(eltype(x)) isa TVar{Float32}
        @test zero(x[1]) isa TVar{Float32}
        @test eltype(x)(2.75) isa TVar{Float32}
        @test value(eltype(x)(2.75)) === 2.75f0
        @test convert(eltype(x), 3) isa TVar{Float32}
        @test eltype(x)[1.5, -2.5] isa Vector{TVar{Float32}}
        # An accumulator seeded that way records on the tape it is combined with.
        acc = zero(eltype(x))
        for xi in x
            acc += xi^2
        end
        @test acc isa TVar{Float32}
        @test !isconstant(acc)
    end

    @testset "every cover result carries the element type" begin
        t, outidx = tape_function(f32, x032)

        c = cover_hessian(t, outidx)
        @test c isa HessianCover{Float32}
        @test fieldeltypes(c) == [Float32]

        d = cover_hessian(WithDiagnostics(), t, outidx)
        @test d isa HessianDiagnostics{Float32}
        @test fieldeltypes(d) == [Float32]

        vc = cover_variance(t, outidx)
        @test vc isa VarianceCover{Float32}
        @test fieldeltypes(vc) == [Float32]

        @test cover_hybrid(t, outidx, 3) isa Vector{Float32}
        @test eltype(adjoints(t, outidx)) === Float32
        @test eltype(sensitivities(t)[1]) === Float32
        @test eltype(abs_forward(t, zeros(Float32, nnodes(t)))) === Float32
        @test eltype(abs_reverse(t, zeros(Float32, nnodes(t)))) === Float32
        rhos, theta = local_covers(t, adjoints(t, outidx))
        @test eltype(theta) === Float32
        @test all(r -> eltype(r) === Float32, rhos)

        # Both split rules, so the dead-slot branch of the local cover is typed too.
        @test cover_hessian(t, outidx; rule = SensitivitySplit()).h_sum isa Vector{Float32}
        @test cover_hessian(t, outidx; rule = SensitivitySplit(ones(Float32, 2))).h_sum isa
            Vector{Float32}
        @test eltype(DerivativeCovers.split_scales(ValueSplit(), t)) === Float32
        @test eltype(DerivativeCovers.split_scales(SensitivitySplit(), t)) === Float32

        # Products mix the tape's type with the vector's.
        @test hvp(t, outidx, Float32[1, 0]) isa Vector{Float32}
        @test eltype(hvp(t, outidx, Float64[1, 0])) === Float64
    end

    @testset "vector-output results carry the element type" begin
        r32(x) = [x[1] * x[2], exp(x[1]) - x[3], x[2] + 2x[3]]
        y032 = Float32[0.5, 2.0, -1.0]
        t, outidxs = tape_function(r32, y032)
        @test t isa Tape{Float32}

        jc = cover_jacobian(t, outidxs)
        @test jc isa JacobianCover{Float32}
        @test fieldeltypes(jc) == [Float32]
        @test eltype(jacobian_surrogate(t, outidxs)) === Float32
        @test cover_jacobian_hybrid(t, outidxs, 2) isa Vector{Float32}
        @test eltype(marginal_weights(t, outidxs, ones(Float32, 3))) === Float32

        bc = balance_cover(t, outidxs)
        @test eltype(bc.a) === Float32 && eltype(bc.b) === Float32

        @test jvp(t, outidxs, Float32[1, 0, 1]) isa Vector{Float32}
        @test vjp(t, outidxs, Float32[1, 1, 1]) isa Vector{Float32}

        nls32(x) = [x[1] * x[2] - 1, exp(x[1]) - 2]
        nc = cover_nls(nls32, Float32[0.5, 2.0])
        @test nc isa NLSCover{Float32}
        @test fieldeltypes(nc) == [Float32]

        nd = cover_nls(WithDiagnostics(), nls32, Float32[0.5, 2.0])
        @test nd isa NLSDiagnostics{Float32}
        @test fieldeltypes(nd) == [Float32]
        a, b = gn_ab(nd)
        @test eltype(a) === Float32 && eltype(b) === Float32

        # A declared sub-map distills its declaration in the scratch tape's type, and the
        # `DeclaredCurvature` it records is what the cover sweep resolves statically.
        dec = declare_submap(x -> [x[1] * x[2], x[1] + x[2]], Float32[0.7, -1.3])
        @test eltype(dec.y) === Float32
        @test eltype(dec.J) === Float32
        @test eltype(dec.S) === Float32
        ts = Tape(Float32[0.7, -1.3])
        xs = [TVar(ts, i) for i in 1:2]
        ys = submap(x -> [x[1] * x[2], x[1] + x[2]], xs)
        @test ys isa Vector{TVar{Float32}}
        @test any(C -> C isa DerivativeCovers.DeclaredCurvature{Float32}, ts.Cs)
        @test cover_hessian(ts, ys[1].idx).h_sum isa Vector{Float32}
    end

    @testset "an inner-product node records in the element type" begin
        # PairedCurvature carries no element type of its own, so its cover comes from the tape's.
        t, outidx = tape_function(v -> dot(v[1:3], v[4:6]), Float32[0.9, -0.4, 1.3, 0.4, -1.1, 0.7])
        @test t isa Tape{Float32}
        @test any(C -> C isa DerivativeCovers.PairedCurvature, t.Cs)
        @test cover_hessian(t, outidx).h_sum isa Vector{Float32}
        @test cover_hybrid(t, outidx, 2) isa Vector{Float32}
        W32 = Float32[1 2 0.5; 0.5 1 2; 1 0 1]
        tm, om = tape_function(x -> sum(v^2 for v in W32 * x), Float32[0.7, -1.3, 0.4])
        @test tm isa Tape{Float32}
        @test cover_hessian(tm, om).h_sum isa Vector{Float32}
    end

    @testset "caller-supplied weights join the tape's type" begin
        r32(x) = [x[1] * x[2], exp(x[1]) - x[3], x[2] + 2x[3]]
        t, outidxs = tape_function(r32, Float32[0.5, 2.0, -1.0])
        # Float64 weights do not widen a Float32 cover: the tape's type is the cover's.
        jc = cover_jacobian(t, outidxs; a = [1.0, 2.0, 0.5])
        @test jc isa JacobianCover{Float32}
        @test cover_jacobian_hybrid(t, outidxs, 2; a = [1.0, 2.0, 0.5]) isa Vector{Float32}
        @test eltype(balance_cover(t, outidxs; a0 = [1.0, 2.0, 0.5]).a) === Float32
        # The weights are copied on the way in, so the caller's vector is never rescaled.
        a0 = [1.0, 2.0, 0.5]
        balance_cover(t, outidxs; a0)
        @test a0 == [1.0, 2.0, 0.5]
    end

    @testset "Float32 covers are valid and agree with Float64" begin
        t32, o32 = tape_function(f32, x032)
        t64, o64 = tape_function(f32, x064)
        c32 = cover_hessian(WithDiagnostics(), t32, o32)
        c64 = cover_hessian(WithDiagnostics(), t64, o64)
        @test c32.grad ≈ c64.grad rtol = sqrt(eps(Float32))
        @test c32.h_sum ≈ c64.h_sum rtol = sqrt(eps(Float32))
        @test c32.h_quad ≈ c64.h_quad rtol = sqrt(eps(Float32))

        # The covering chain holds in Float32, up to Float32 roundoff.
        H = ForwardDiff.hessian(f32, x064)
        gate32 = 1 + sqrt(eps(Float32))
        @test cover_ratio(H, Float64.(c32.h_sum)) <= gate32
        @test cover_ratio(H, Float64.(c32.h_quad)) <= gate32
        @test bound_ratio(H, Float64.(c32.B)) <= gate32
        @test all(abs.(H) .<= Float64.(c32.h_sum) * Float64.(c32.h_sum)' * gate32)
    end

    @testset "the same program in three precisions" begin
        # Every element type runs the same code path; only the reported precision changes.
        for T in (Float16, Float32, Float64, BigFloat)
            t, outidx = tape_function(f32, T[0.5, 2.0])
            @test eltype(t) === T
            c = cover_hessian(t, outidx)
            @test c isa HessianCover{T}
            @test eltype(c.h_sum) === T
            @test all(>=(0), c.h_sum)
        end
    end

end
