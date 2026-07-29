# Array operations must match their scalarized equivalents and cover ForwardDiff's Hessian.

"f = sum(abs2, z ./ (1 + dot(z, z))), exercising a scalar TVar as a broadcast divisor."
scaled_sumabs2(z) = sum(abs2, z ./ (1 + dot(z, z)))

"f = (sum z)²/2, the primitive form of a sum reduction."
sum_prim(z) = abs2(sum(z)) / 2

"Same function as [`sum_prim`](@ref), with the sum written as an explicit chain of binary `+`."
function sum_scalar(z)
    acc = z[1]
    for i in 2:length(z)
        acc = acc + z[i]
    end
    return abs2(acc) / 2
end

"f = sum(abs2, z), the primitive form of `sum(f, xs)`."
sumf_prim(z) = sum(abs2, z)

"Same function as [`sumf_prim`](@ref), with the sum written as an explicit chain of binary `+`."
function sumf_scalar(z)
    acc = abs2(z[1])
    for i in 2:length(z)
        acc = acc + abs2(z[i])
    end
    return acc
end

"Head `dot(c0, softmax(z))`, the primitive form."
softmax_prim(c0) = z -> dot(c0, softmax(z))

"""
Same function as [`softmax_prim`](@ref), scalarized: the max is detached via `value.(z)` (a plain
number when `z` is a `Vector{TVar}`), the exponentials and their sum are explicit chains, and the
head is an explicit multiply-add chain.
"""
function softmax_scalar(c0)
    return z -> begin
        m = maximum(value.(z))
        n = length(z)
        es = [exp(z[i] - m) for i in 1:n]
        s = es[1]
        for i in 2:n
            s = s + es[i]
        end
        ps = [es[i] / s for i in 1:n]
        acc = c0[1] * ps[1]
        for i in 2:n
            acc = acc + c0[i] * ps[i]
        end
        return acc
    end
end

"""
Softmax with the max subtracted as a `TVar` (`z[argmax(value.(z))]`) instead of detached: still a
valid cover of the same Hessian, but strictly looser, since the absolute-value sweep cannot see
the exact cancellation of the max entry's derivative contributions between numerator and
denominator.
"""
function softmax_taped(x::AbstractVector)
    isempty(x) && throw(ArgumentError("softmax of an empty vector is undefined"))
    m = x[argmax(value.(x))]
    e = exp.(x .- m)
    return e ./ sum(e)
end
softmax_taped_prim(c0) = z -> dot(c0, softmax_taped(z))

"Head `dot(c0, tanh.(layernorm(z)))`, the primitive form."
layernorm_prim(c0) = z -> dot(c0, tanh.(layernorm(z)))

"Same function as [`layernorm_prim`](@ref), scalarized with explicit `+` chains for both sums."
function layernorm_scalar(c0)
    return z -> begin
        n = length(z)
        s = z[1]
        for i in 2:n
            s = s + z[i]
        end
        mu = s / n
        d = [z[i] - mu for i in 1:n]
        sq = [abs2(d[i]) for i in 1:n]
        ssum = sq[1]
        for i in 2:n
            ssum = ssum + sq[i]
        end
        s2 = ssum / n
        denom = sqrt(s2)
        y = [tanh(d[i] / denom) for i in 1:n]
        acc = c0[1] * y[1]
        for i in 2:n
            acc = acc + c0[i] * y[i]
        end
        return acc
    end
end

const C0_5 = [0.7, -1.3, 0.4, 2.1, -0.6]

@testset "arrayops.jl" begin

    @testset "broadcast with a TVar scalar" begin
        x0 = [0.3, -0.9, 1.1]
        H = ForwardDiff.hessian(scaled_sumabs2, x0)
        res = cover_hessian(WithDiagnostics(), scaled_sumabs2, x0)
        @test res.grad ≈ ForwardDiff.gradient(scaled_sumabs2, x0)
        @test bound_ratio(H, res.B) <= GATE
        @test cover_ratio(H, res.h_sum) <= GATE
        @test cover_ratio(H, res.h_quad) <= GATE
        @test all(res.h_quad .<= res.h_sum .* GATE)
    end

    @testset "sum parity and economy" begin
        x0 = [0.3, -1.1, 0.7, 1.2]
        rp = cover_hessian(WithDiagnostics(), sum_prim, x0)
        rs = cover_hessian(WithDiagnostics(), sum_scalar, x0)
        @test rp.grad ≈ rs.grad rtol = 1.0e-12
        @test rp.B ≈ rs.B rtol = 1.0e-12
        @test rp.h_sum ≈ rs.h_sum rtol = 1.0e-12
        @test rp.h_sum_rev ≈ rs.h_sum_rev rtol = 1.0e-12
        @test rp.h_quad ≈ rs.h_quad rtol = 1.0e-12

        vp = cover_variance(sum_prim, x0)
        vs = cover_variance(sum_scalar, x0)
        @test vp.h_sum ≈ vs.h_sum rtol = 1.0e-12
        @test vp.h_var ≈ vs.h_var rtol = 1.0e-12
        @test vp.severity ≈ vs.severity rtol = 1.0e-12

        H = ForwardDiff.hessian(sum_prim, x0)
        @test bound_ratio(H, rp.B) <= GATE
        @test cover_ratio(H, rp.h_sum) <= GATE
        @test cover_ratio(H, rp.h_quad) <= GATE

        # A sum of n TVars records as one linear node: n inputs plus that one node.
        t, _ = tape_function(z -> sum(z), x0)
        @test nnodes(t) == length(x0) + 1

        # sum(f, xs) parity: sum(abs2, z) vs the chained abs2 twin.
        rpf = cover_hessian(WithDiagnostics(), sumf_prim, x0)
        rsf = cover_hessian(WithDiagnostics(), sumf_scalar, x0)
        @test rpf.grad ≈ rsf.grad rtol = 1.0e-12
        @test rpf.B ≈ rsf.B rtol = 1.0e-12
        @test rpf.h_sum ≈ rsf.h_sum rtol = 1.0e-12
        @test rpf.h_quad ≈ rsf.h_quad rtol = 1.0e-12

        @test_throws "cannot record an empty sum" sum(TVar[])
    end

    @testset "softmax" begin
        x0 = [0.3, -1.7, 2.1, -0.4, 1.0]
        f = softmax_prim(C0_5)

        @testset "validity vs ForwardDiff" begin
            H = ForwardDiff.hessian(f, x0)
            res = cover_hessian(WithDiagnostics(), f, x0)
            @test res.grad ≈ ForwardDiff.gradient(f, x0)
            @test bound_ratio(H, res.B) <= GATE
            @test cover_ratio(H, res.h_sum) <= GATE
            @test cover_ratio(H, res.h_quad) <= GATE
        end

        @testset "parity with scalarized twin" begin
            fs = softmax_scalar(C0_5)
            rp = cover_hessian(WithDiagnostics(), f, x0)
            rs = cover_hessian(WithDiagnostics(), fs, x0)
            @test rp.grad ≈ rs.grad rtol = 1.0e-12
            @test rp.B ≈ rs.B rtol = 1.0e-12
            @test rp.h_sum ≈ rs.h_sum rtol = 1.0e-12
            @test rp.h_sum_rev ≈ rs.h_sum_rev rtol = 1.0e-12
            @test rp.h_quad ≈ rs.h_quad rtol = 1.0e-12

            vp = cover_variance(f, x0)
            vs = cover_variance(fs, x0)
            @test vp.h_sum ≈ vs.h_sum rtol = 1.0e-12
            @test vp.h_var ≈ vs.h_var rtol = 1.0e-12
            @test vp.severity ≈ vs.severity rtol = 1.0e-12
        end

        @testset "shift transparency of the detached max" begin
            res = cover_hessian(WithDiagnostics(), f, x0)
            fshift = z -> dot(C0_5, softmax(z .+ 10.0))
            res_shift = cover_hessian(WithDiagnostics(), fshift, x0 .- 10.0)
            @test res_shift.h_sum ≈ res.h_sum rtol = 1.0e-12
            @test res_shift.h_quad ≈ res.h_quad rtol = 1.0e-12
        end

        @testset "detached vs taped max" begin
            x0u = [0.2, -0.5, 3.0, -1.1, 0.6]     # a clear unique max at index 3
            H = ForwardDiff.hessian(f, x0u)
            res_d = cover_hessian(WithDiagnostics(), f, x0u)
            res_t = cover_hessian(WithDiagnostics(), softmax_taped_prim(C0_5), x0u)

            @test cover_ratio(H, res_t.h_sum) <= GATE
            @test cover_ratio(H, res_t.h_quad) <= GATE
            @test all(res_d.h_sum .<= res_t.h_sum .* (1 + 1.0e-9))
            @test all(res_d.h_quad .<= res_t.h_quad .* (1 + 1.0e-9))
            ratio = maximum(res_t.h_sum ./ res_d.h_sum)
            @test ratio > 1.1
        end

        @testset "plain-vector behavior" begin
            xv = [0.5, -1.2, 2.0, 0.1]
            sv = softmax(xv)
            @test sum(sv) ≈ 1
            @test sv ≈ exp.(xv) ./ sum(exp.(xv))

            @test_throws "softmax of an empty vector" softmax(Float64[])

            @test all(isfinite, softmax([1000.0, 0.0]))
        end
    end

    @testset "layernorm" begin
        x0 = [0.3, -1.7, 2.1, -0.4, 1.0]
        f = layernorm_prim(C0_5)

        @testset "validity vs ForwardDiff" begin
            H = ForwardDiff.hessian(f, x0)
            res = cover_hessian(WithDiagnostics(), f, x0)
            @test res.grad ≈ ForwardDiff.gradient(f, x0)
            @test bound_ratio(H, res.B) <= GATE
            @test cover_ratio(H, res.h_sum) <= GATE
            @test cover_ratio(H, res.h_quad) <= GATE
        end

        @testset "parity with scalarized twin" begin
            fs = layernorm_scalar(C0_5)
            rp = cover_hessian(WithDiagnostics(), f, x0)
            rs = cover_hessian(WithDiagnostics(), fs, x0)
            @test rp.grad ≈ rs.grad rtol = 1.0e-12
            @test rp.B ≈ rs.B rtol = 1.0e-12
            @test rp.h_sum ≈ rs.h_sum rtol = 1.0e-12
            @test rp.h_sum_rev ≈ rs.h_sum_rev rtol = 1.0e-12
            @test rp.h_quad ≈ rs.h_quad rtol = 1.0e-12

            vp = cover_variance(f, x0)
            vs = cover_variance(fs, x0)
            @test vp.h_sum ≈ vs.h_sum rtol = 1.0e-12
            @test vp.h_var ≈ vs.h_var rtol = 1.0e-12
            @test vp.severity ≈ vs.severity rtol = 1.0e-12
        end

        @testset "plain-vector sanity" begin
            n = 6
            xv = [0.4, -1.1, 2.3, -0.7, 1.5, 0.2]
            yv = layernorm(xv)
            @test sum(yv) ≈ 0 atol = 1.0e-12
            @test sum(abs2, yv) / n ≈ 1 atol = 1.0e-12
        end

        @testset "epsilon keyword" begin
            fe = z -> dot(C0_5, tanh.(layernorm(z; epsilon = 1.0e-5)))
            H = ForwardDiff.hessian(fe, x0)
            res = cover_hessian(WithDiagnostics(), fe, x0)
            @test res.grad ≈ ForwardDiff.gradient(fe, x0)
            @test cover_ratio(H, res.h_sum) <= GATE
            @test cover_ratio(H, res.h_quad) <= GATE
        end

        @testset "errors" begin
            @test_throws "layernorm needs at least 2 entries" layernorm([2.0])
        end
    end

    @testset "structured matrix-vector products" begin
        # LinearAlgebra's wrapper types are `AbstractMatrix` and so tie with the generic
        # matrix-vector methods. Each disambiguating method must record the same tape its
        # dense twin does, whichever side carries the variables.
        M = [2.0 -1.0 0.5; 0.3 1.4 -0.8; -0.6 0.2 1.1]
        c = [0.9, -0.4, 1.3]
        x0 = [0.4, -1.1, 0.7]
        v0 = vec(M)

        @testset "$wrap" for wrap in (
                UpperTriangular, LowerTriangular,
                UnitUpperTriangular, UnitLowerTriangular,
            )
            cases = [
                (
                    "constant matrix", z -> sum(abs2, wrap(M) * z),
                    z -> sum(abs2, Matrix(wrap(M)) * z), x0,
                ),
                (
                    "variable matrix", v -> sum(abs2, wrap(reshape(v, 3, 3)) * v[1:3]),
                    v -> sum(abs2, Matrix(wrap(reshape(v, 3, 3))) * v[1:3]), v0,
                ),
                (
                    "variable matrix, constant vector",
                    v -> sum(abs2, wrap(reshape(v, 3, 3)) * x0),
                    v -> sum(abs2, Matrix(wrap(reshape(v, 3, 3))) * x0), v0,
                ),
            ]
            @testset "$name" for (name, f, fd, z0) in cases
                res = cover_hessian(WithDiagnostics(), f, z0)
                @test res.grad ≈ ForwardDiff.gradient(fd, z0)
                @test cover_ratio(ForwardDiff.hessian(fd, z0), res.h_sum) <= GATE
                @test res.h_sum == cover_hessian(WithDiagnostics(), fd, z0).h_sum
            end
        end

        @testset "$name" for (name, f, fd, z0) in
            [
                (
                    "Diagonal, constant matrix", z -> sum(abs2, Diagonal(c) * z),
                    z -> sum(abs2, Matrix(Diagonal(c)) * z), x0,
                ),
                (
                    "Diagonal, variable matrix", v -> sum(abs2, Diagonal(v[1:3]) * v[4:6]),
                    v -> sum(abs2, Matrix(Diagonal(v[1:3])) * v[4:6]), [c; x0],
                ),
                ("adjoint row vector, constant row", z -> abs2(c' * z), z -> abs2(sum(c .* z)), x0),
                (
                    "adjoint row vector, variable row", v -> abs2(v[1:3]' * v[4:6]),
                    v -> abs2(sum(v[1:3] .* v[4:6])), [c; x0],
                ),
                (
                    "adjoint row vector, constant vector", v -> abs2(v[1:3]' * c),
                    v -> abs2(sum(v[1:3] .* c)), [c; x0],
                ),
                (
                    "transpose row vector, variable row", v -> abs2(transpose(v[1:3]) * v[4:6]),
                    v -> abs2(sum(v[1:3] .* v[4:6])), [c; x0],
                ),
                (
                    "transpose row vector, constant vector", v -> abs2(transpose(v[1:3]) * c),
                    v -> abs2(sum(v[1:3] .* c)), [c; x0],
                ),
                # `transpose` and `adjoint` are defined on TVar itself, so a wrapper around variable
                # data has element type TVar: the row-vector methods dispatch on it, and an entry that
                # is itself transposed on access resolves rather than erroring.
                (
                    "transposed variable matrix", v -> sum(abs2, transpose(reshape(v, 3, 3)) * v[1:3]),
                    v -> sum(abs2, Matrix(transpose(reshape(v, 3, 3))) * v[1:3]), v0,
                ),
                (
                    "transposed variable matrix, constant vector",
                    v -> sum(abs2, transpose(reshape(v, 3, 3)) * x0),
                    v -> sum(abs2, Matrix(transpose(reshape(v, 3, 3))) * x0), v0,
                ),
                (
                    "adjointed variable matrix", v -> sum(abs2, reshape(v, 3, 3)' * v[1:3]),
                    v -> sum(abs2, Matrix(reshape(v, 3, 3)') * v[1:3]), v0,
                ),
            ]
            res = cover_hessian(WithDiagnostics(), f, z0)
            @test res.grad ≈ ForwardDiff.gradient(fd, z0)
            @test cover_ratio(ForwardDiff.hessian(fd, z0), res.h_sum) <= GATE
            @test res.h_sum == cover_hessian(WithDiagnostics(), fd, z0).h_sum
        end
    end

    @testset "covariance — softmax and layernorm" begin
        cases = [
            (
                "softmax", softmax_prim(C0_5), [0.3, -1.7, 2.1, -0.4, 1.0],
                [1.0e-3, 1.0e2, 1.0, 1.0e-1, 1.0e4],
            ),
            (
                "layernorm", layernorm_prim(C0_5), [0.3, -1.7, 2.1, -0.4, 1.0],
                [1.0e-2, 1.0e3, 1.0, 1.0e-1, 1.0e2],
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

end
