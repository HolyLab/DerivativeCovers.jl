# Inner-product and linear nodes must match hand-scalarized graphs in every cover quantity.

"f = sum_j w[j]*x[j], written as scalar multiply-adds in evaluation order."
function scalar_dot(w, x)
    acc = 0.0 * x[1]
    first = true
    for j in eachindex(w, x)
        t = w[j] * x[j]
        acc = first ? t : acc + t
        first = false
    end
    return acc
end

"f = sum(tanh.(W*x)), W and x both variable, written with the matvec primitive."
function matvec_prim(z)
    W = reshape(z[1:6], 2, 3)
    x = z[7:9]
    return sum(tanh.(W * x))
end

"Same function as [`matvec_prim`](@ref), scalarized row by row."
function matvec_scalar(z)
    W = reshape(z[1:6], 2, 3)
    x = z[7:9]
    acc = 0.0 * z[1]
    first = true
    for m in 1:2
        t = tanh(scalar_dot(view(W, m, :), x))
        acc = first ? t : acc + t
        first = false
    end
    return acc
end

"f = exp(dot(z[1:3], z[4:6])), the primitive form of a single inner product."
dot_prim(z) = exp(dot(z[1:3], z[4:6]))
dot_scalar(z) = exp(scalar_dot(z[1:3], z[4:6]))

"f = 0.5*dot(w0, z)^2 with constant `w0`, the primitive form of a constant-coefficient dot."
dot_const_prim(w0) = z -> 0.5 * dot(w0, z)^2
dot_const_scalar(w0) = z -> 0.5 * scalar_dot(w0, z)^2

const W0_CONST = [1.0e-2, 0.1, 1.0, 10.0]

"Coefficient vector mixing typed constants with variables, for the split inner product."
mixed_w(z) = (T = eltype(z); T[T(2.0), z[5], T(-1.5), z[6]])

"f = exp(dot(w, z[1:4])/4) with `w` constant in two of its four slots."
dot_mixed_prim(z) = exp(dot(mixed_w(z), z[1:4]) / 4)
dot_mixed_scalar(z) = exp(scalar_dot(mixed_w(z), z[1:4]) / 4)

"""
Operand pair exercising all four pair states, in slot order: constant×variable,
variable×constant, variable×variable, constant×constant.
"""
function both_mixed(z)
    T = eltype(z)
    return T[T(2.0), z[4], z[5], T(-1.5)], T[z[1], T(0.5), z[2], T(3.0)]
end

"f = exp((dot(w, x) + z[3])/4) with both operands mixed."
function dot_bothmixed_prim(z)
    w, x = both_mixed(z)
    return exp((dot(w, x) + z[3]) / 4)
end

"Same function as [`dot_bothmixed_prim`](@ref), scalarized."
function dot_bothmixed_scalar(z)
    w, x = both_mixed(z)
    return exp((scalar_dot(w, x) + z[3]) / 4)
end

"Name, primitive program, scalarized twin, and evaluation point for each parity case."
const PRIMITIVE_PARITY = [
    (
        "matvec var x var", matvec_prim, matvec_scalar,
        [0.3, -0.7, 1.1, -0.2, 0.5, 0.9, -1.3, 2.0, 0.4],
    ),
    ("single dot", dot_prim, dot_scalar, [0.3, -1.1, 0.7, 1.2, -0.4, 0.9]),
    (
        "constant-coefficient dot", dot_const_prim(W0_CONST), dot_const_scalar(W0_CONST),
        [0.3, -1.2, 2.0, -0.5],
    ),
    (
        "mixed coefficient operand", dot_mixed_prim, dot_mixed_scalar,
        [0.3, -1.1, 0.7, 1.2, -0.4, 0.9],
    ),
    (
        "both operands mixed", dot_bothmixed_prim, dot_bothmixed_scalar,
        [0.6, -0.9, 0.4, 1.1, -0.3],
    ),
]

"Every primitive program above, paired with a plain-Julia twin usable by ForwardDiff."
const PRIMITIVE_VALIDITY = [(name, f, x0) for (name, f, _, x0) in PRIMITIVE_PARITY]

@testset "primitives.jl" begin

    @testset "parity with scalarized twin — $name" for (name, fp, fs, x0) in PRIMITIVE_PARITY
        rp = cover_hessian(WithDiagnostics(), fp, x0)
        rs = cover_hessian(WithDiagnostics(), fs, x0)
        @test rp.grad ≈ rs.grad rtol = 1.0e-12
        @test rp.B ≈ rs.B rtol = 1.0e-12
        @test rp.h_sum ≈ rs.h_sum rtol = 1.0e-12
        @test rp.h_sum_rev ≈ rs.h_sum_rev rtol = 1.0e-12
        @test rp.h_quad ≈ rs.h_quad rtol = 1.0e-12

        # The primitive form's inner-product nodes carry `PairedCurvature`, the scalarized twin's
        # multiply nodes a dense 2x2, so this pins the two curvature branches of `hvp!` together.
        v = Float64[cos(3i) for i in eachindex(x0)]
        @test hvp(fp, x0, v) ≈ hvp(fs, x0, v) rtol = 1.0e-12

        vp = cover_variance(fp, x0)
        vs = cover_variance(fs, x0)
        @test vp.h_sum ≈ vs.h_sum rtol = 1.0e-12
        @test vp.h_var ≈ vs.h_var rtol = 1.0e-12
        @test vp.severity ≈ vs.severity rtol = 1.0e-12
    end

    @testset "validity vs ForwardDiff — $name" for (name, f, x0) in PRIMITIVE_VALIDITY
        H = ForwardDiff.hessian(f, x0)
        res = cover_hessian(WithDiagnostics(), f, x0)
        @test bound_ratio(H, res.B) <= GATE
        @test cover_ratio(H, res.h_sum) <= GATE
        @test cover_ratio(H, res.h_quad) <= GATE
        @test all(res.h_quad .<= res.h_sum .* GATE)
        @test res.h_sum_rev ≈ res.h_sum rtol = 1.0e-12
    end

    @testset "exact cover of the factored quadratic with constant weights" begin
        f = dot_const_prim(W0_CONST)
        x0 = [0.3, -1.2, 2.0, -0.5]
        res = cover_hessian(WithDiagnostics(), f, x0)
        @test res.h_sum ≈ abs.(W0_CONST) rtol = 1.0e-12
        @test res.h_quad ≈ abs.(W0_CONST) rtol = 1.0e-12

        # The entrywise expansion of the same function loses the factored structure and is
        # strictly looser, as METHODS.md's tightness study predicts.
        n = 4
        f_entrywise = z -> begin
            acc = 0.0 * z[1]
            first = true
            for i in 1:n, j in 1:n
                t = W0_CONST[i] * W0_CONST[j] * z[i] * z[j]
                acc = first ? t : acc + t
                first = false
            end
            return acc / 2
        end
        res_e = cover_hessian(WithDiagnostics(), f_entrywise, x0)
        @test all(res_e.h_sum .>= abs.(W0_CONST))
        @test maximum(res_e.h_sum ./ abs.(W0_CONST)) > 2
    end

    @testset "exact cover of least squares in the weights" begin
        rng = MersenneTwister(31)
        p, q = 3, 2
        x0v = randn(rng, q)
        y0v = randn(rng, p)
        f = z -> begin
            W = reshape(z, p, q)
            r = W * x0v .- y0v
            0.5 * sum(abs2.(r))
        end
        zW = randn(rng, p * q)
        res = cover_hessian(WithDiagnostics(), f, zW)
        pattern = vec(repeat(abs.(x0v)', p, 1))    # each row's weights get |x0v|
        @test res.h_sum ≈ pattern rtol = 1.0e-12
        @test res.h_quad ≈ pattern rtol = 1.0e-12

        H = ForwardDiff.hessian(f, zW)
        @test bound_ratio(H, res.B) <= GATE
        @test cover_ratio(H, res.h_sum) <= GATE
    end

    @testset "repeated parents" begin
        f = z -> log(1 + dot(z, z))
        x0 = [0.5, -1.2, 0.3]
        H = ForwardDiff.hessian(f, x0)
        res = cover_hessian(WithDiagnostics(), f, x0)
        @test res.grad ≈ ForwardDiff.gradient(f, x0)
        @test bound_ratio(H, res.B) <= GATE
        @test cover_ratio(H, res.h_sum) <= GATE
        @test cover_ratio(H, res.h_quad) <= GATE
        @test all(res.h_quad .<= res.h_sum .* GATE)
    end

    # The split is over parent slots, so a vector against itself splits pairwise as well: the
    # constant slot pairs with itself into the offset, the recording slots into the atom set.
    @testset "repeated parents in a mixed operand" begin
        f = z -> (T = eltype(z); log(1 + dot(T[z[1], T(2.0), z[2]], T[z[1], T(2.0), z[2]])))
        x0 = [0.5, -1.2]
        H = ForwardDiff.hessian(f, x0)
        res = cover_hessian(WithDiagnostics(), f, x0)
        @test res.grad ≈ ForwardDiff.gradient(f, x0)
        @test bound_ratio(H, res.B) <= GATE
        @test cover_ratio(H, res.h_sum) <= GATE
        @test cover_ratio(H, res.h_quad) <= GATE
    end

    # However long the operand, the split records the product in a bounded number of nodes; the
    # scalarized twin grows as 2q.
    @testset "a mixed operand costs O(1) nodes" begin
        function mixed_extra(q)
            x0 = [0.1 * i for i in 1:(q + q ÷ 2)]
            f = z -> dot(TVar[isodd(j) ? TVar(float(j)) : z[q + (j ÷ 2)] for j in 1:q], z[1:q])
            t, _ = tape_function(f, x0)
            return nnodes(t) - length(x0)
        end
        @test mixed_extra(4) == mixed_extra(16) == mixed_extra(64)
    end

    @testset "constant-x and constant-W matvecs" begin
        xconst = [0.5, -0.3]
        f_constx = z -> sum(abs2.(reshape(z, 2, 2) * xconst))
        x0 = [0.1, 0.2, 0.3, 0.4]
        H = ForwardDiff.hessian(f_constx, x0)
        res = cover_hessian(WithDiagnostics(), f_constx, x0)
        @test res.grad ≈ ForwardDiff.gradient(f_constx, x0)
        @test bound_ratio(H, res.B) <= GATE
        @test cover_ratio(H, res.h_sum) <= GATE
        @test cover_ratio(H, res.h_quad) <= GATE

        Aconst = [1.0 0.5; -0.3 2.0]
        f_constW = z -> sum(abs2.(Aconst * z)) / 2
        z0 = [0.5, -0.2]
        Hw = ForwardDiff.hessian(f_constW, z0)
        resw = cover_hessian(WithDiagnostics(), f_constW, z0)
        @test resw.grad ≈ ForwardDiff.gradient(f_constW, z0)
        @test cover_ratio(Hw, resw.h_sum) <= GATE
        @test cover_ratio(Hw, resw.h_quad) <= GATE
        # A constant-W matvec records only linear nodes: no PairedCurvature atom on the tape.
        t, _ = tape_function(f_constW, z0)
        @test !any(C -> C isa DerivativeCovers.PairedCurvature, t.Cs)
        # H = A'A exactly (each residual is linear in z), so the minimal cover is the row norms.
        @test Hw ≈ Aconst' * Aconst
    end

    @testset "merges and variance compose on the matvec program" begin
        f, x0 = matvec_prim, [0.3, -0.7, 1.1, -0.2, 0.5, 0.9, -1.3, 2.0, 0.4]
        H = ForwardDiff.hessian(f, x0)
        res = cover_hessian(WithDiagnostics(), f, x0)
        t, outidx = tape_function(f, x0)
        N = nnodes(t)

        @test cover_hybrid(t, outidx, 1) ≈ res.h_sum rtol = 1.0e-12

        h_nodewise = cover_hybrid(t, outidx, collect(1:N))
        @test all(>=(0), h_nodewise)
        @test cover_ratio(H, h_nodewise) <= GATE
        @test all(res.h_quad .<= h_nodewise .* GATE)      # atoms travel together within a node

        rng = MersenneTwister(41)
        groups = rand(rng, 1:5, N)
        h = cover_hybrid(t, outidx, groups)
        @test all(>=(0), h)
        @test cover_ratio(H, h) <= GATE
    end

    @testset "covariance — dot and matvec" begin
        cases = [
            (
                "single dot", dot_prim, [0.3, -1.1, 0.7, 1.2, -0.4, 0.9],
                [1.0e-6, 1.0e3, 1.0, 1.0e-2, 1.0e4, 1.0],
            ),
            (
                "matvec", matvec_prim, [0.3, -0.7, 1.1, -0.2, 0.5, 0.9, -1.3, 2.0, 0.4],
                [1.0e-3, 1.0e2, 1.0, 1.0e-1, 1.0e4, 1.0e-2, 1.0e1, 1.0e3, 1.0],
            ),
        ]
        @testset "$name" for (name, f, x0, d) in cases
            res = cover_hessian(WithDiagnostics(), f, x0)

            resd = cover_hessian(WithDiagnostics(), xt -> f(xt ./ d), d .* x0)
            @test resd.h_sum ≈ res.h_sum ./ d rtol = 1.0e-12
            @test resd.h_quad ≈ res.h_quad ./ d rtol = 1.0e-12

            c = 7.3
            resc = cover_hessian(WithDiagnostics(), x -> c * f(x), x0)
            @test resc.h_sum ≈ sqrt(c) .* res.h_sum rtol = 1.0e-12
            @test resc.h_quad ≈ sqrt(c) .* res.h_quad rtol = 1.0e-12
        end
    end

    @testset "errors" begin
        t, _ = tape_function(x -> x[1] + x[2] + x[3], [1.0, 2.0, 3.0])
        xs = [TVar(t, 1), TVar(t, 2), TVar(t, 3)]

        @test_throws DimensionMismatch dot(xs, xs[1:2])
        @test_throws DimensionMismatch dot(xs[1:2], [1.0, 2.0, 3.0])
        # The two builders name what they were asked to record: constant coefficients reach
        # linearnode, recording operands on both sides reach dotnode.
        @test_throws "cannot record an empty inner product" dot(TVar[], TVar[])
        @test_throws "cannot record an empty linear combination" dot(Float64[], TVar[])

        t2, _ = tape_function(x -> x[1] + x[2], [1.0, 2.0])
        ys = [TVar(t2, 1), TVar(t2, 2), TVar(t2, 1)]
        mixed = [xs[1], xs[2], TVar(t2, 1)]
        @test_throws "different tapes" dot(xs, ys)
        @test_throws "different tapes" dot([1.0, 2.0, 3.0], mixed)

        W = reshape([xs[1], xs[2], xs[3], xs[1], xs[2], xs[3]], 2, 3)
        @test_throws DimensionMismatch W * xs[1:2]
        @test_throws DimensionMismatch W * [1.0, 2.0]
        Wreal = [1.0 2.0; 3.0 4.0]
        @test_throws DimensionMismatch Wreal * xs

        # The diagonal product broadcasts, which would silently expand a mismatched operand.
        D, Dreal = Diagonal(xs), Diagonal([10.0, 20.0, 30.0])
        @test length(D * xs) == length(Dreal * xs) == 3
        @test_throws DimensionMismatch D * xs[1:2]
        @test_throws DimensionMismatch D * [1.0, 2.0]
        @test_throws DimensionMismatch Dreal * xs[1:2]
        @test_throws DimensionMismatch Diagonal(xs[1:1]) * xs
        @test_throws DimensionMismatch Diagonal([1.0]) * xs

        # A node with three or more parents carries second partials only in the forms the cover
        # sweep knows how to split; a general dense C is not one, and says so rather than
        # reading the 2x2 corner of a larger block. The bound and the product still read it.
        t3 = Tape([1.0, 2.0, 3.0])
        DerivativeCovers.pushnode!(
            t3, 6.0, [1, 2, 3], ones(3), DerivativeCovers.DenseCurvature(ones(3, 3))
        )
        vbar = [0.0, 0.0, 0.0, 1.0]
        @test_throws "the rank-1 cover is defined for the unary and binary forms" (
            local_covers(t3, vbar)
        )
        @test DerivativeCovers.hessian_bound(t3, vbar, sensitivities(t3)) == ones(3, 3)
        @test hvp(t3, 4, [1.0, 0.0, 0.0]) == ones(3)
    end

    @testset "inline unary and binary curvature" begin
        # The elementals record their blocks inline; `getindex` and `size` report them as the
        # matrices they stand for.
        t, _ = tape_function(x -> exp(x[1]) * x[2], [0.5, 2.0])
        Cu = t.Cs[findfirst(C -> C isa DerivativeCovers.UnaryCurvature, t.Cs)]
        @test size(Cu) == (1, 1)
        @test Cu[1, 1] == exp(0.5)
        @test_throws BoundsError Cu[1, 2]
        Cb = t.Cs[findfirst(C -> C isa DerivativeCovers.BinaryCurvature, t.Cs)]
        @test size(Cb) == (2, 2)
        @test Cb[1, 2] == Cb[2, 1] == 1.0        # d²(u*v)/du dv
        @test Cb[1, 1] == Cb[2, 2] == 0.0
        @test_throws BoundsError Cb[3, 1]
        # No elemental records the general dense form.
        @test !any(C -> C isa DerivativeCovers.DenseCurvature, t.Cs)
    end

    @testset "PairedCurvature entries" begin
        # An inner-product node's second partials are the cross terms alone: slots are ordered
        # (w_1..w_q, x_1..x_q), so the only nonzeros are the pairs q apart.
        t, _ = tape_function(v -> dot(v[1:3], v[4:6]), [0.9, -0.4, 1.3, 0.4, -1.1, 0.7])
        k = findfirst(C -> C isa DerivativeCovers.PairedCurvature, t.Cs)
        C = t.Cs[k]
        @test size(C) == (6, 6)
        @test all(C[i, j] == Float64(abs(i - j) == 3) for i in 1:6, j in 1:6)
        @test all(C[i, j] == C[j, i] for i in 1:6, j in 1:6)
    end

end
