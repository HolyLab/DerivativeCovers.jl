# Run array-facing APIs on vectors, views, and OffsetArrays. The expected behavior depends on the
# role of each axis:
#
#   - Data axes propagate to index-matched results but do not affect values.
#   - Tape enumeration axes are one-based; offset inputs must fail explicitly.
#   - outidxs may have arbitrary axes. Only jvp propagates them to its result.

@testset "genericaxes.jl" begin

    offsetcopy(A) = OffsetArray(A, ntuple(d -> -2, ndims(A)))
    fullview(A) = view(A, ntuple(_ -> Colon(), ndims(A))...)

    H = [1.0 0.3 -0.2; 0.3 2.0 0.5; -0.2 0.5 1.4]
    B = abs.(H) .+ 0.1
    h = [1.2, 1.5, 0.9]
    href = [1.0, 2.0, 0.9]
    J = [1.0 0.3 0.5; 0.2 2.0 0.1]
    av = [1.5, 2.0]
    bv = [1.1, 1.4, 0.9]

    # Reductions: only the values are defined, and they must not depend on how the data is
    # labelled. Each case names the arrays it consumes so the wrappers can be applied uniformly.
    @testset "value invariance — $name" for (name, f, args) in (
            ("cover_ratio", cover_ratio, (H, h)),
            ("bound_ratio", bound_ratio, (H, B)),
            ("rank1_ratio", rank1_ratio, (B, h)),
            ("tightness", tightness, (h, href)),
            ("jacobian_ratio", jacobian_ratio, (J, av, bv)),
            ("scaled_spectral_radius", scaled_spectral_radius, (H, h)),
            ("scaled_singular_value", scaled_singular_value, (J, av, bv)),
            ("anisotropy", anisotropy, (h,)),
        )
        ref = f(args...)
        @test f(map(offsetcopy, args)...) == ref
        @test f(map(fullview, args)...) == ref
    end

    # Index-matched results: the values are likewise invariant, but the result must additionally
    # come back on the axes of the argument it is matched to. `which` picks that argument.
    @testset "axes propagation — $name" for (name, f, args, which) in (
            ("spectral_shrink", h -> spectral_shrink(h, 0.5), (h,), 1),
            ("equilibrate", equilibrate, (J, av, bv), 1),
            ("dependency_severity", dependency_severity, (h, href), 1),
            ("jacobian_shrink", (a, b) -> jacobian_shrink(a, b, 0.5)[1], (av, bv), 1),
            (
                "compose_jacobian_cover", (a2, b2, a1, b1) ->
                compose_jacobian_cover(a2, b2, a1, b1).a, (av, bv, bv, av), 1,
            ),
            (
                "compose_hessian_cover", (hg, g, a, b) ->
                compose_hessian_cover(hg, g, a, b).h_sum, (av, av, av, bv), 4,
            ),
            ("inflate", (h, f) -> inflate(h, f), (h, fill(2.0, 3)), 1),
            ("inflation_factors", h -> inflation_factors(RampInflation(), h, 4.0), (h,), 1),
            ("inflation_shape", h -> inflation_shape(LognormalInflation(), h), (h,), 1),
        )
        ref = f(args...)
        offargs = map(offsetcopy, args)
        got = f(offargs...)
        # Approximate on the values: the composition rules reach `dot`, whose summation order —
        # and so whose last bit — depends on whether the operand is a `Vector` BLAS can take.
        # Wrapping must not move the result further than that.
        @test collect(got) ≈ collect(ref)
        @test axes(got) == axes(offargs[which])
        @test collect(f(map(fullview, args)...)) ≈ collect(ref)
    end

    # A spike names a coordinate, so it must be given in the coordinates of the cover it applies
    # to rather than as a 1-based position.
    @testset "SpikeInflation indexes h's own axes" begin
        ho = offsetcopy(h)
        @test inflation_shape(SpikeInflation(-1), ho)[-1] == 1
        @test sum(inflation_shape(SpikeInflation(-1), ho)) == 1
        @test_throws BoundsError inflation_shape(SpikeInflation(3), ho)
    end

    # The tape numbers its inputs `1:n`, so the programs and the vectors laid out over the inputs
    # are 1-based. The contract is declared, and an offset argument must say so rather than fail
    # somewhere downstream.
    @testset "one-based declarations" begin
        f(x) = sum(x[i] * x[i + 1] for i in 1:(length(x) - 1)) + 0.5 * sum(abs2, x)
        x0 = [0.7, -1.3, 0.4]
        v = [1.0, 0.5, -2.0]
        t, outidx = tape_function(f, x0)

        @test_throws "offset arrays are not supported" Tape(offsetcopy(x0))
        @test_throws "offset arrays are not supported" hvp(f, offsetcopy(x0), v)
        @test_throws "offset arrays are not supported" hvp(t, outidx, offsetcopy(v))
        @test_throws "offset arrays are not supported" cover_hessian(f, offsetcopy(x0))
        @test_throws "offset arrays are not supported" scaled_spectral_radius(
            t, outidx,
            offsetcopy(h)
        )
        @test_throws "offset arrays are not supported" jacobian_minimax(offsetcopy(B))
        @test_throws "offset arrays are not supported" jacobian_representability(offsetcopy(B))

        # The declaration is the only thing offset about them: the same data 1-based goes through.
        @test hvp(f, x0, v) == hvp(t, outidx, v)
        @test scaled_spectral_radius(t, outidx, h) isa Real
    end

    # `outidxs` is a container of node numbers. Its axes are the caller's business, and the
    # results must not change when they change.
    @testset "outidxs carries its own axes" begin
        r(x) = [x[1] * x[2], x[2] * x[3] + x[1], x[3]^2]
        x0 = [0.7, -1.3, 0.4]
        v = [1.0, 0.5, -2.0]
        u = [1.0, -0.5, 2.0]
        t, outidxs = tape_function(r, x0)
        oidx = offsetcopy(collect(outidxs))

        # jvp is index-matched to outidxs on both sides of the call.
        @test collect(jvp(t, oidx, v)) == collect(jvp(t, outidxs, v))
        @test axes(jvp(t, oidx, v)) == axes(oidx)

        # vjp's seed is the output-side vector, so it shares outidxs' axes; its result is laid
        # out over the inputs and is 1-based whatever outidxs does.
        @test vjp(t, oidx, offsetcopy(u)) == vjp(t, outidxs, u)
        @test axes(vjp(t, oidx, offsetcopy(u))) == (Base.OneTo(t.ninputs),)
        @test_throws DimensionMismatch vjp(t, oidx, u)

        # Everything else numbers its outputs by position, so offset outidxs is accepted and
        # leaves no trace in the result.
        @test cover_jacobian(t, oidx).Utilde == cover_jacobian(t, outidxs).Utilde
        @test cover_jacobian(t, oidx).b_max == cover_jacobian(t, outidxs).b_max
        @test cover_jacobian_hybrid(t, oidx, 2) == cover_jacobian_hybrid(t, outidxs, 2)
        @test jacobian_surrogate(t, oidx) == jacobian_surrogate(t, outidxs)
        @test adjoints(t, oidx) == adjoints(t, outidxs)

        # The marginal alternation pairs `outidxs` against a 1-based iterate, so its weights must
        # come back on output positions. It needs a pattern with total support to converge.
        rfull(x) = [x[1] * x[2] + x[3], x[2] * x[3] + x[1], x[1] * x[3] + x[2]]
        tf, oif = tape_function(rfull, x0)
        oidxf = offsetcopy(collect(oif))
        bpos = [1.0, 0.5, 2.0]
        @test marginal_weights(tf, oidxf, bpos) == marginal_weights(tf, oif, bpos)
        @test axes(marginal_weights(tf, oidxf, bpos)) == (Base.OneTo(3),)
        @test balance_cover(tf, oidxf) == balance_cover(tf, oif)

        sc, scref = support_components(t, oidx), support_components(t, outidxs)
        @test ncomponents(sc) == ncomponents(scref)
        @test [rowcomponent(sc, k) for k in 1:3] == [rowcomponent(scref, k) for k in 1:3]

        @test scaled_singular_value(t, oidx, offsetcopy(ones(3)), ones(3)) ==
            scaled_singular_value(t, outidxs, ones(3), ones(3))

        # The weights are indexed by output position, not by outidxs, so an offset one is refused.
        @test_throws "offset arrays are not supported" cover_jacobian(
            t, outidxs;
            a = offsetcopy(ones(3))
        )
    end

    # Arrays built inside a taped program carry their own axes through the recording rules: a
    # matrix-vector product is index-matched to the matrix's row axis, and every node form that
    # consumes the result pairs coefficients with terms by index.
    @testset "recording rules follow the operand axes" begin
        W = [1.0 2.0 0.5; 0.5 1.0 2.0; 1.0 0.0 1.0]
        x0 = [0.7, -1.3, 0.4]
        v = [1.0, 0.5, -2.0]
        Wrows = OffsetArray(W, -2, 0)          # rows offset; columns index the tape's inputs

        t = Tape(x0)
        xs = [TVar(t, i) for i in 1:3]
        @test axes(Wrows * xs, 1) == axes(Wrows, 1)

        @test hvp(x -> sum(Wrows * x), x0, v) == hvp(x -> sum(W * x), x0, v)
        @test hvp(x -> dot(Wrows * x, Wrows * x), x0, v) == hvp(x -> dot(W * x, W * x), x0, v)
        @test cover_hessian(x -> sum(Wrows * x), x0).h_sum ==
            cover_hessian(x -> sum(W * x), x0).h_sum

        # The column axis indexes the tape's inputs, which are 1-based, so a mismatch there is a
        # dimension error rather than something to carry through.
        @test_throws DimensionMismatch offsetcopy(W) * xs

        # `mul!` is index-matched to its destination, and the matrix-matrix rule carries the row
        # axis of one operand and the column axis of the other.
        yo = OffsetArray(Vector{TVar}(undef, 3), -2)
        @test axes(mul!(yo, Wrows, xs), 1) == axes(Wrows, 1)
        @test value.(collect(yo)) == value.(collect(W * xs))
        @test_throws "mul! destination has axes" mul!(Vector{TVar}(undef, 3), Wrows, xs)

        Xcols = OffsetArray(reshape(xs, 3, 1), 0, -2)
        Yo = OffsetArray(Matrix{TVar}(undef, 3, 1), -2, -2)
        @test axes(mul!(Yo, Wrows, Xcols)) == (axes(Wrows, 1), axes(Xcols, 2))
        @test value.(vec(collect(Yo))) == value.(collect(W * xs))

        # Out of place, LinearAlgebra sizes the destination one-based, so an offset operand is a
        # mismatch there rather than a result relabelled onto axes it was not computed over.
        @test_throws "mul! destination has axes" Wrows * Xcols
    end

end
