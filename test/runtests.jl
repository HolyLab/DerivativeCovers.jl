using DerivativeCovers
using DerivativeCovers.Problems
using DerivativeCovers.Diagnostics
# The tests exercise the sweeps and their intermediates, most of which are deliberately not in the
# top namespace. Naming them here keeps the call sites readable while leaving the export list —
# pinned in the "public API boundary" testset — as the only thing `using DerivativeCovers` grants.
using DerivativeCovers: submapnode, value, isconstant, nnodes, cover_ratio, bound_ratio,
                        rank1_ratio, tightness, jacobian_ratio, jacobian_representability,
                        jacobian_minimax, dependency_severity, scaled_spectral_radius,
                        scaled_singular_value, spectral_shrink, jacobian_shrink, equilibrate,
                        compose_hessian_cover, compose_jacobian_cover, gn_ab, nls_loss_tape,
                        declare_submap, submap, cyclic_partition, level_partition, node_levels,
                        nls_partition, adjoints, sensitivities, abs_reverse, abs_forward,
                        local_covers, balanced_split, variance_seeds, dependency_sweep,
                        jacobian_surrogate, marginal_weights, balance_cover
# This package adds `Tape` methods to these MatrixCovers functions but does not re-export them.
using MatrixCovers: support_components, gramcover, gramcover!, SupportComponents, ncomponents,
                    rowcomponent, colcomponent
using Aqua
using ExplicitImports
using ForwardDiff
using LinearAlgebra
using Random
using Test

include("programs.jl")

# The covering chain abs.(H) <= B <= sum_k s_k s_k' <= h*h' is exact in exact arithmetic;
# roundoff can push a measured ratio a hair above 1.
const GATE = 1 + 1e-10

include("constants.jl")
include("primitives.jl")
include("composites.jl")
include("arrayops.jl")
include("mlp.jl")
include("cancellation.jl")
include("nls.jl")
include("vectoroutput.jl")
include("support.jl")
include("jacobian.jl")
include("gram.jl")
include("jacobianconverse.jl")
include("equilibrate.jl")
include("compose.jl")
include("paired.jl")

@testset "DerivativeCovers.jl" begin

    @testset "Aqua" begin
        Aqua.test_all(DerivativeCovers)
    end

    @testset "ExplicitImports" begin
        # `Base.broadcastable` is the documented way to opt a type into broadcasting, but Base
        # does not mark it public. The two public-ness checks read `Base.ispublic` only from
        # Julia 1.11; before that they fall back to `isexported` and flag public-but-unexported
        # names, so they run only where they are accurate.
        public_ok = VERSION >= v"1.11"
        test_explicit_imports(DerivativeCovers;
                              all_explicit_imports_are_public=public_ok,
                              all_qualified_accesses_are_public=public_ok ?
                                  (; ignore=(:broadcastable,)) : false)
    end

    @testset "public API boundary" begin
        # TVar records operations rather than standing in for a number: it must not be a Real, so
        # that generic `::Real` methods reject it instead of silently taping something the package
        # has no rule for.
        @test !(TVar <: Real)
        @test supertype(TVar) === Any

        # What `using DerivativeCovers` grants, in full. Pinned exactly, so that a name cannot
        # drift into the top namespace as a side effect of adding one somewhere in `src/`.
        # `names` reports `public` declarations alongside exports from 1.11 on, and includes the
        # module's own name, so neither can be read off it directly.
        exported = filter(s -> Base.isexported(DerivativeCovers, s), names(DerivativeCovers))
        @test Set(setdiff(exported, [:DerivativeCovers])) == Set([
            :Tape, :TVar, :tape_function,
            :cover_hessian, :cover_jacobian, :cover_nls, :cover_variance,
            :cover_hybrid, :cover_jacobian_hybrid,
            :CoverOnly, :WithDiagnostics,
            :HessianCover, :HessianDiagnostics, :JacobianCover, :NLSCover, :VarianceCover,
            :SplitRule, :ValueSplit, :SensitivitySplit,
            :hvp, :jvp, :vjp,
        ])

        # Addressable by qualification and covered by the compatibility promise, but unexported:
        # the contracts a new elemental implements, the quantities defined on a cover from any
        # source, and the adapters between the result types and the bare-vector functions.
        internals = (:pushnode!, :unarynode, :binarynode, :dotnode, :linearnode, :noderef,
                     :constantnode, :atom_slots, :certifies_dead, :check_finite_partials,
                     :submapnode, :split_scales, :value, :isconstant, :nnodes,
                     :cover_ratio, :bound_ratio, :rank1_ratio, :tightness, :jacobian_ratio,
                     :jacobian_representability, :jacobian_minimax, :dependency_severity,
                     :scaled_spectral_radius, :scaled_singular_value, :spectral_shrink,
                     :jacobian_shrink, :equilibrate,
                     :compose_hessian_cover, :compose_jacobian_cover, :gn_ab, :nls_loss_tape,
                     :declare_submap, :submap)
        for sym in internals
            @test isdefined(DerivativeCovers, sym)
            @test !Base.isexported(DerivativeCovers, sym)
            VERSION >= v"1.11" && @test Base.ispublic(DerivativeCovers, sym)
        end

        # The sweep machinery and the partition policies behind `cover_hybrid`'s integer argument
        # carry no promise: they are what a reimplementation of the sweeps is free to restructure.
        private = (:cyclic_partition, :level_partition, :node_levels, :nls_partition,
                   :group_seeds, :adjoints, :sensitivities, :abs_reverse, :abs_forward,
                   :hessian_bound, :local_covers, :local_cover_slots, :balanced_split,
                   :variance_seeds, :dependency_sweep, :jacobian_surrogate, :marginal_weights,
                   :balance_cover)
        for sym in private
            @test isdefined(DerivativeCovers, sym)
            VERSION >= v"1.11" && @test !Base.ispublic(DerivativeCovers, sym)
        end

        # MatrixCovers' functions, which this package extends with `Tape` methods rather than
        # owning. Reaching them takes `using MatrixCovers`.
        for sym in (:support_components, :gramcover, :gramcover!, :SupportComponents,
                    :ncomponents, :rowcomponent, :colcomponent)
            @test !Base.isexported(DerivativeCovers, sym)
        end

        # The study instruments live in their own namespace, reached by
        # `using DerivativeCovers.Diagnostics`, and are not re-exported by the parent.
        diagnostics = (:InflationPattern, :UniformInflation, :LognormalInflation,
                       :SpikeInflation, :RampInflation, :inflation_shape, :inflation_factors,
                       :inflate, :anisotropy, :cover_norm, :stopping_index, :stopping_shift,
                       :regularized_step, :regularization_bias, :psd_shift, :psd_excess,
                       :cover_cg, :step_degradation)
        for sym in diagnostics
            @test Base.isexported(DerivativeCovers.Diagnostics, sym)
            @test !isdefined(DerivativeCovers, sym)
        end
    end

    @testset "tape mechanics" begin
        f = x -> x[1] * x[2] + exp(x[3])
        x0 = [2.0, -3.0, 0.5]
        t, outidx = tape_function(f, x0)
        @test t.ninputs == 3
        @test t.vals[outidx] ≈ f(x0)
        @test outidx == nnodes(t)
        # Inputs come first and carry no parents; every parent precedes its child.
        @test all(isempty, t.parents[1:t.ninputs])
        @test all(k -> all(<(k), t.parents[k]), 1:nnodes(t))

        @test_throws "program output is not a TVar" tape_function(x -> 1.0, x0)
        @test_throws "different tapes" tape_function(x0) do xs
            t2, o2 = tape_function(f, x0)
            xs[1] * TVar(t2, o2)
        end
        # Elementals outside the supported set fail fast rather than silently linearizing.
        @test_throws MethodError tape_function(x -> abs(x[1]), x0)
    end

    # A program that is one elemental applied straight to the inputs has unit absolute
    # sensitivities, so its elementwise bound `B` is exactly `abs.(H)`. Both partials of each
    # elemental are therefore pinned against ForwardDiff: the first by `grad`, the second by `B`.
    @testset "elemental partials — $name" for (name, f, x0) in [
        ("exp", x -> exp(x[1]), [0.7]),
        ("log", x -> log(x[1]), [1.7]),
        ("sin", x -> sin(x[1]), [0.6]),
        ("cos", x -> cos(x[1]), [0.6]),
        ("tan", x -> tan(x[1]), [0.6]),
        ("atan", x -> atan(x[1]), [0.7]),
        ("tanh", x -> tanh(x[1]), [0.4]),
        ("sqrt", x -> sqrt(x[1]), [2.3]),
        ("inv", x -> inv(x[1]), [2.0]),
        ("abs2", x -> abs2(x[1]), [-1.3]),
        ("reciprocal", x -> 3.0 / x[1], [2.0]),
        ("constant power", x -> x[1]^2.5, [1.4]),
        ("variable base", x -> x[1]^x[2], [1.7, 2.3]),
        ("variable exponent", x -> 2.0^x[1], [1.3]),
        ("rational base", x -> (3 // 2)^x[1], [1.3]),
        ("product", x -> x[1] * x[2], [2.0, -3.0]),
        ("quotient", x -> x[1] / x[2], [2.0, -3.0]),
    ]
        res = cover_hessian(WithDiagnostics(), f, x0)
        @test res.grad ≈ ForwardDiff.gradient(f, x0)
        @test res.B ≈ abs.(ForwardDiff.hessian(f, x0))
    end

    @testset "identity elementals leave the tape unchanged" begin
        x0 = [2.0, -3.0]
        res = cover_hessian(WithDiagnostics(), x -> x[1] * x[2], x0)
        for f in (x -> (+x[1]) * x[2], x -> x[1]' * x[2])
            t, _ = tape_function(f, x0)
            @test nnodes(t) == 3                       # two inputs, one product node
            @test cover_hessian(WithDiagnostics(), f, x0).h_sum == res.h_sum
        end
    end

    @testset "a TVar scales an array of constants" begin
        c = [2.0, -0.5, 3.0]
        x0 = [1.3]
        for f in (x -> sum(x[1] * c), x -> sum(c * x[1]))
            @test cover_hessian(WithDiagnostics(), f, x0).grad ≈ ForwardDiff.gradient(f, x0)
        end
    end

    @testset "a variable exponent needs a positive base" begin
        @test_throws "requires a positive base" cover_hessian(x -> x[1]^x[2], [-1.7, 2.0])
        @test_throws "requires a positive base" cover_hessian(x -> (-2.0)^x[1], [1.3])
    end

    @testset "constant powers at the origin" begin
        # `x^0` is constant and `x^1` linear, so the derivatives their vanishing leading factors
        # multiply are zero at the origin, where the power those factors multiply is singular.
        x0 = [0.0, 1.3]
        for p in (0, 1, 2, 3)
            f = x -> x[1]^p + x[1] * x[2]
            res = cover_hessian(WithDiagnostics(), f, x0)
            @test res.grad ≈ ForwardDiff.gradient(f, x0)
            @test res.B ≈ abs.(ForwardDiff.hessian(f, x0))
        end
        # A nonzero factor leaves a genuine singularity singular.
        @test !all(isfinite,
                   cover_hessian(WithDiagnostics(), x -> x[1]^0.5 + x[1] * x[2], x0;
                                 check=false).h_sum)
    end

    @testset "gradient matches ForwardDiff — $name" for (name, f, x0) in PROBLEMS
        res = cover_hessian(WithDiagnostics(), f, x0)
        @test res.grad ≈ ForwardDiff.gradient(f, x0)
    end

    @testset "covering chain — $name" for (name, f, x0) in PROBLEMS
        H = ForwardDiff.hessian(f, x0)
        res = cover_hessian(WithDiagnostics(), f, x0)
        @test bound_ratio(H, res.B) <= GATE                  # abs.(H) <= B
        for h in (res.h_sum, res.h_quad)
            @test all(>=(0), h)
            @test rank1_ratio(res.B, h) <= GATE              # B <= h*h'
            @test cover_ratio(H, h) <= GATE                  # abs.(H) <= h*h'
        end
        # The quadrature merge is never looser than the sum merge, coordinatewise.
        @test all(res.h_quad .<= res.h_sum .* GATE)
        # The absolute-value reverse sweep reproduces the sum merge.
        @test res.h_sum_rev ≈ res.h_sum rtol=1e-12
        # For H ⪰ 0 the minimal cover is sqrt.(diag(H)); no valid cover can go below it.
        if isposdef(H)
            @test all(res.h_quad .>= sqrt.(diag(H)) .* (1 - 1e-10))
        end
    end

    @testset "exact cover of the factored rank-1 quadratic" begin
        # f = (w'x)²/2 has H = w*w', whose minimal cover is abs.(w). A single nonlinear node
        # carries the whole Hessian, so both merges attain it.
        res = cover_hessian(WithDiagnostics(), quad_factored(WQ), XQ)
        @test res.h_sum ≈ abs.(WQ)
        @test res.h_quad ≈ abs.(WQ)
        H = ForwardDiff.hessian(quad_factored(WQ), XQ)
        @test cover_ratio(H, res.h_sum) ≈ 1
    end

    @testset "elementwise bound on the cancellation pair" begin
        x0 = CANCEL_X
        n = length(x0)

        # Entrywise form: each Hessian entry is produced by exactly one tape node, so the
        # absolute-value bound has nothing to cancel and attains abs.(H).
        He = ForwardDiff.hessian(cancellation, x0)
        Be = cover_hessian(WithDiagnostics(), cancellation, x0).B
        @test Be ≈ abs.(He)

        # Summed form: the same Hessian I - e*e'/n, but its diagonal now arises by cancellation
        # between sum(x²)/2 and (sum x)²/(2n). The bound sums magnitudes instead, giving
        # B = I + e*e'/n and a diagonal excess of (n+1)/(n-1).
        Hs = ForwardDiff.hessian(cancellation_summed, x0)
        Bs = cover_hessian(WithDiagnostics(), cancellation_summed, x0).B
        @test Hs ≈ He
        @test Bs ≈ I + ones(n, n) ./ n
        @test maximum(Bs[i, i] / abs(Hs[i, i]) for i in 1:n) ≈ (n + 1) / (n - 1)
    end

    @testset "covariance" begin
        cases = [("rosenbrock n=2", rosenbrock, [-1.2, 1.0], [1e-6, 1e3]),
                 ("quad matvec", quad_matvec(AQ), XQ, [1e-6, 1e3, 1.0, 1e-2, 1e4]),
                 ("logsumexp n=8", logsumexp,
                  [-3.0, -1.0, -2.0, 0.0, 1.0, 2.0, -0.5, 3.0],
                  [1e-3, 1e2, 1.0, 1e-1, 1e4, 1e-2, 1e1, 1e3])]
        @testset "$name" for (name, f, x0, d) in cases
            res = cover_hessian(WithDiagnostics(), f, x0)

            # x -> D*x leaves every intermediate value and adjoint unchanged and puts a factor
            # 1/d[a] on each input edge, so h -> h ./ d exactly.
            resd = cover_hessian(WithDiagnostics(), xt -> f(xt ./ d), d .* x0)
            @test resd.h_sum ≈ res.h_sum ./ d rtol=1e-12
            @test resd.h_quad ≈ res.h_quad ./ d rtol=1e-12

            # f -> c*f scales every adjoint by c, hence every rho and every h by sqrt(c).
            c = 7.3
            resc = cover_hessian(WithDiagnostics(), x -> c * f(x), x0)
            @test resc.h_sum ≈ sqrt(c) .* res.h_sum rtol=1e-12
            @test resc.h_quad ≈ sqrt(c) .* res.h_quad rtol=1e-12
        end
    end

    @testset "balanced_split" begin
        for (c11, c12, c22, z1, z2) in ((0.0, 1.0, 0.0, 2.0, 5.0),      # multiplication node
                                        (1.0, 4.0, 2.0, 0.1, 10.0),
                                        (0.0, 3.0, 0.0, 0.0, 1.0),      # zero-value fallback
                                        (2.0, 5.0, 1.0, 1.0, 0.0))
            rho1, rho2 = balanced_split(c11, c12, c22, z1, z2)
            @test rho1 >= sqrt(c11) * (1 - 1e-12)
            @test rho2 >= sqrt(c22) * (1 - 1e-12)
            @test rho1 * rho2 >= c12 * (1 - 1e-12)
        end
        # With both parent values positive the split is balanced against them: for a pure
        # multiplication node (c11 = c22 = 0), rho1/rho2 = z2/z1 and rho1*rho2 = c12.
        rho1, rho2 = balanced_split(0.0, 1.0, 0.0, 2.0, 8.0)
        @test rho1 / rho2 ≈ 8.0 / 2.0
        @test rho1 * rho2 ≈ 1.0

        # The split lands on the feasible segment [sqrt(c11), c12/sqrt(c22)], so the cover is
        # exactly tight on the off-diagonal however extreme the entries and values.
        rng = MersenneTwister(11)
        for _ in 1:10_000
            c11 = rand(rng) < 0.25 ? 0.0 : exp(12 * randn(rng))
            c22 = rand(rng) < 0.25 ? 0.0 : exp(12 * randn(rng))
            c12 = exp(12 * randn(rng))
            c12^2 > c11 * c22 || continue
            z1 = rand(rng) < 0.1 ? 0.0 : exp(8 * randn(rng))
            z2 = rand(rng) < 0.1 ? 0.0 : exp(8 * randn(rng))
            rho1, rho2 = balanced_split(c11, c12, c22, z1, z2)
            @test isfinite(rho1) && isfinite(rho2)
            @test rho1 >= sqrt(c11) * (1 - 1e-12)
            @test rho2 >= sqrt(c22) * (1 - 1e-12)
            @test rho1 * rho2 ≈ c12 rtol=1e-12
        end

        # Continuity where the off-diagonal constraint takes over from the diagonal ones: at
        # c12^2 == c11*c22 the segment collapses to the diagonal cover (sqrt(c11), sqrt(c22)).
        for eps in (1e-2, 1e-4, 1e-8)
            rho1, rho2 = balanced_split(1.0, 10.0 * (1 + eps), 100.0, 1.0, 1.0)
            @test rho1 ≈ 1.0 rtol=2eps
            @test rho2 ≈ 10.0 rtol=2eps
        end

        # Continuity as a parent value vanishes: the equalizing target diverges, but the
        # segment endpoint it is clamped to does not.
        for z1 in (1e-3, 1e-6, 1e-12, 0.0)
            rho1, rho2 = balanced_split(1.0, 1.0, 1e-2, z1, 1.0)
            @test rho1 ≈ 10.0
            @test rho2 ≈ 0.1
        end

        # A value ratio extreme enough to overflow or underflow the equalizing target still
        # lands on the endpoint it was heading for, rather than reverting to the interior.
        for (z1, z2) in ((1.0, 1e300), (1e-300, 1e300), (1e-300, 1.0), (1.0, Inf))
            rho1, rho2 = balanced_split(1.0, 10.0, 1.0, z1, z2)
            @test rho1 ≈ 10.0
            @test rho2 ≈ 1.0
        end
        for (z1, z2) in ((1e300, 1.0), (1e300, 1e-300), (1.0, 1e-300), (Inf, 1.0))
            rho1, rho2 = balanced_split(1.0, 10.0, 1.0, z1, z2)
            @test rho1 ≈ 1.0
            @test rho2 ≈ 10.0
        end
    end

    @testset "non-finite second partials" begin
        # A program evaluated where it is not twice-differentiable is named as such, rather than
        # blamed on the split: the covering invariant compares against entries that are NaN or
        # infinite, so it would reject a split that covers everything coverable.

        # Binary node: a divided difference whose denominator vanishes is 0/0, giving
        # C = [0 -Inf; -Inf NaN].
        divdiff = x -> (exp(x[1]) - exp(x[2])) / (x[1] - x[2])
        @test_throws "not twice-differentiable" cover_hessian(divdiff, [0.5, 0.5])

        # The offending node is named where the sweep reaches it first. The paired sweep runs in
        # reverse order and so meets the division itself; the forward pass of `local_covers`
        # meets one of its parents, whose adjoint has already come through the division.
        tsing, osing = tape_function(divdiff, [0.5, 0.5])
        @test_throws "non-finite second partials" cover_hessian(CoverOnly(), tsing, osing)
        @test_throws "node value NaN" cover_hessian(CoverOnly(), tsing, osing)
        @test_throws "from a node downstream" local_covers(tsing, adjoints(tsing, osing))

        # Inner-product node: its covering entry is the absolute adjoint, which the reciprocal of
        # a vanishing inner product makes infinite.
        recipdot = x -> 1 / dot(x[1:2], x[3:4])
        @test_throws "not twice-differentiable" cover_hessian(recipdot, [1.0, 2.0, 2.0, -1.0])

        # `check=false` remains the way to get IEEE propagation instead: the cover comes back
        # non-finite for the caller to test, as `abs.(H) <= h*h'` cannot hold at such a point.
        @test !all(isfinite, cover_hessian(WithDiagnostics(), divdiff, [0.5, 0.5]; check=false).h_sum)
        @test !all(isfinite,
                   cover_hessian(WithDiagnostics(), recipdot, [1.0, 2.0, 2.0, -1.0];
                                 check=false).h_sum)

        # Away from the singularity both programs cover normally.
        @test all(isfinite, cover_hessian(WithDiagnostics(), divdiff, [0.5, 1.5]).h_sum)
        @test all(isfinite, cover_hessian(WithDiagnostics(), recipdot, [1.0, 2.0, 2.0, 1.0]).h_sum)
    end

    @testset "split rules" begin
        # Passing ValueSplit explicitly is the default, bitwise.
        @testset "ValueSplit is the default — $name" for (name, f, x0) in PROBLEMS
            res = cover_hessian(WithDiagnostics(), f, x0)
            resv = cover_hessian(WithDiagnostics(), f, x0; rule=ValueSplit())
            @test resv.h_sum == res.h_sum
            @test resv.h_quad == res.h_quad
        end

        # SensitivitySplit satisfies the same covering chain on the whole battery.
        @testset "SensitivitySplit covering chain — $name" for (name, f, x0) in PROBLEMS
            H = ForwardDiff.hessian(f, x0)
            res = cover_hessian(WithDiagnostics(), f, x0; rule=SensitivitySplit())
            for h in (res.h_sum, res.h_quad)
                @test all(>=(0), h)
                @test rank1_ratio(res.B, h) <= GATE
                @test cover_ratio(H, h) <= GATE
            end
            @test res.h_sum_rev ≈ res.h_sum rtol=1e-12
            # The one-group hybrid reproduces the sum merge under any rule.
            @test cover_hybrid(f, x0, 1; rule=SensitivitySplit()) ≈ res.h_sum rtol=1e-12
        end

        # A multiplication node with both operands exactly zero: the value channel is empty
        # (both scale references vanish), so ValueSplit's split at a nearby point is decided
        # by the ratio of the two now-tiny values, while the sensitivity channel sees
        # sigma = 1 on both sides at every point. The perturbations are dyadic, so the
        # parent values at the displaced point are exact and the value ratio is exactly 4.
        fmul = x -> (x[1] - 1) * (x[2] - 1)
        x0 = [1.0, 1.0]
        xp = [1.0 + 2.0^-40, 1.0 + 2.0^-38]
        @test cover_hessian(WithDiagnostics(), fmul, x0; rule=SensitivitySplit()).h_sum == [1.0, 1.0]
        @test cover_hessian(WithDiagnostics(), fmul, xp; rule=SensitivitySplit()).h_sum == [1.0, 1.0]
        @test cover_hessian(WithDiagnostics(), fmul, x0; rule=ValueSplit()).h_sum == [1.0, 1.0]
        @test cover_hessian(WithDiagnostics(), fmul, xp; rule=ValueSplit()).h_sum == [2.0, 0.5]

        # Dead channel: sigma == 0 certifies that the slot's total gradient vanishes, so the
        # multiplication node deposits nothing there and its partner covers only its own
        # (zero) diagonal; the whole cover on that side comes from the x[1]^2 node.
        fdead = x -> x[1]^2 * x[2]
        xd = [0.0, 3.0]
        Hd = ForwardDiff.hessian(fdead, xd)
        resd = cover_hessian(WithDiagnostics(), fdead, xd; rule=SensitivitySplit())
        @test resd.h_sum ≈ [sqrt(6), 0.0]
        @test cover_ratio(Hd, resd.h_sum) <= GATE
        # Approaching the dead point, the response of h is bounded by the sqrt(sigma1*sigma2)
        # envelope — a square-root cusp, not a jump.
        rese = cover_hessian(WithDiagnostics(), fdead, [2.0^-20, 3.0]; rule=SensitivitySplit())
        @test maximum(abs, rese.h_sum .- resd.h_sum) <= 2.0^-8

        # A dead inner-product pair: the (x[1]^2, x[3]) pair contributes nothing at x[1] == 0,
        # the (x[2], x[4]) pair splits 1:1, and the x[1]^2 node covers the (1,1) entry.
        fdot = x -> dot([x[1]^2, x[2]], [x[3], x[4]])
        xdot = [0.0, 1.0, 2.0, 3.0]
        resdot = cover_hessian(WithDiagnostics(), fdot, xdot; rule=SensitivitySplit())
        @test resdot.h_sum ≈ [2.0, 1.0, 0.0, 1.0]
        @test cover_ratio(ForwardDiff.hessian(fdot, xdot), resdot.h_sum) <= GATE

        # Input covariance holds when the weights are rescaled alongside the inputs.
        w = [0.5, 4.0, 1.0, 2.0, 0.25]
        d = [1e-6, 1e3, 1.0, 1e-2, 1e4]
        res = cover_hessian(WithDiagnostics(), quad_matvec(AQ), XQ; rule=SensitivitySplit(w))
        resd2 = cover_hessian(WithDiagnostics(), xt -> quad_matvec(AQ)(xt ./ d), d .* XQ;
                              rule=SensitivitySplit(d .* w))
        @test resd2.h_sum ≈ res.h_sum ./ d rtol=1e-12
        @test resd2.h_quad ≈ res.h_quad ./ d rtol=1e-12

        # Weights must be one per input, strictly positive and finite: a zero weight could
        # zero a live slot's scale and falsely certify it dead.
        @test_throws DimensionMismatch cover_hessian(fmul, x0; rule=SensitivitySplit([1.0]))
        @test_throws "strictly positive" cover_hessian(fmul, x0; rule=SensitivitySplit([1.0, 0.0]))
        @test_throws "strictly positive" cover_hessian(fmul, x0; rule=SensitivitySplit([1.0, Inf]))

        # Both channels of the variance pair are seeded from the same splits, so the rule
        # carries through: the hard channel reproduces the sum merge under the same rule.
        vc = cover_variance(fmul, xp; rule=SensitivitySplit())
        @test vc.h_sum == [1.0, 1.0]
        @test all(>=(1 - 1e-12), vc.severity)
        @test cover_variance(fmul, xp).h_sum == [2.0, 0.5]

        # A declaration's curvature covers follow the rule; its sensitivities are relative to
        # the sub-map's own arguments.
        Fsub = x -> [(x[1] - 1) * (x[2] - 1)]
        @test declare_submap(Fsub, xp; rule=SensitivitySplit()).S == [1.0 1.0]
        @test declare_submap(Fsub, xp).S == [2.0 0.5]
    end

    @testset "node levels and partitions" begin
        f = x -> exp(x[1] * x[2]) + x[2]
        t, outidx = tape_function(f, [1.0, 2.0])
        levels = node_levels(t)
        @test all(iszero, levels[1:t.ninputs])
        @test all(k -> levels[k] > maximum(levels[t.parents[k]]; init=-1), (t.ninputs + 1):nnodes(t))

        # A tape shallower than `ngroups` gives every depth its own group.
        lmax = maximum(levels)
        groups = level_partition(t, 10 * lmax)
        for k in (t.ninputs + 1):nnodes(t), l in (t.ninputs + 1):nnodes(t)
            @test (groups[k] == groups[l]) == (levels[k] == levels[l])
        end
        @test level_partition(t, 1) == ones(Int, nnodes(t))
        @test_throws ArgumentError level_partition(t, 0)

        N = nnodes(t)
        @test cyclic_partition(t, 1) == ones(Int, N)
        @test cyclic_partition(t, N) == 1:N
        @test sort(cyclic_partition(t, 3)) == sort(repeat(1:3, outer=N)[1:N])
        @test_throws ArgumentError cyclic_partition(t, 0)
    end

    @testset "hybrid merge — $name" for (name, f, x0) in PROBLEMS
        H = ForwardDiff.hessian(f, x0)
        t, outidx = tape_function(f, x0)
        N = nnodes(t)
        res = cover_hessian(WithDiagnostics(), t, outidx)

        # One group sums every node: the sum merge. One node per group: the quadrature merge.
        @test cover_hybrid(t, outidx, 1) ≈ res.h_sum rtol=1e-12
        @test cover_hybrid(t, outidx, N) ≈ res.h_quad rtol=1e-12
        @test cover_hybrid(t, outidx, collect(1:N)) ≈ res.h_quad rtol=1e-12

        rng = MersenneTwister(hash(name))
        for groups in (cyclic_partition(t, 4), level_partition(t, 4),
                       rand(rng, 1:5, N), rand(rng, 1:2, N))
            h = cover_hybrid(t, outidx, groups)
            @test all(>=(0), h)
            @test cover_ratio(H, h) <= GATE                  # valid for any partition
            @test rank1_ratio(res.B, h) <= GATE
            # Bracketed by the two extremes it interpolates.
            @test all(res.h_quad .<= h .* GATE)
            @test all(h .<= res.h_sum .* GATE)

            # Coarsening a partition (merging groups pairwise) can only loosen the cover.
            coarse = (groups .+ 1) .÷ 2
            @test all(h .<= cover_hybrid(t, outidx, coarse) .* GATE)
        end
    end

    @testset "hybrid merge covariance" begin
        # Rescaling is applied inside the program, so it adds nodes to the tape and shifts the
        # depth levels the default partition bins. Comparing two tapes of the *same* shape — the
        # unscaled one written with the same wrapper — isolates the covariance from that shift.
        cases = [("rosenbrock n=2", rosenbrock, [-1.2, 1.0], [1e-6, 1e3]),
                 ("logsumexp n=8", logsumexp,
                  [-3.0, -1.0, -2.0, 0.0, 1.0, 2.0, -0.5, 3.0],
                  [1e-3, 1e2, 1.0, 1e-1, 1e4, 1e-2, 1e1, 1e3])]
        @testset "$name" for (name, f, x0, d) in cases
            e = ones(length(x0))
            for G in (2, 4)
                href = cover_hybrid(xt -> f(xt ./ e), x0, G)
                hd = cover_hybrid(xt -> f(xt ./ d), d .* x0, G)
                @test hd ≈ href ./ d rtol=1e-12

                c = 7.3
                hc = cover_hybrid(x -> c * f(x), x0, G)
                h1 = cover_hybrid(x -> 1.0 * f(x), x0, G)
                @test hc ≈ sqrt(c) .* h1 rtol=1e-12
            end
        end
    end

    # Independent reference for the variance channel: propagate the squared partials forward from
    # each input, so `u2[k][a]` is the sum over paths from input `a` to node `k` of the squared
    # partials along the path. Contracting with the squared seeds gives the channel's value.
    function squared_sensitivities(t)
        n, N = t.ninputs, nnodes(t)
        u2 = [zeros(n) for _ in 1:N]
        for a in 1:n
            u2[a][a] = 1.0
        end
        for k in (n + 1):N, (i, p) in enumerate(t.parents[k])
            u2[k] .+= t.ds[k][i]^2 .* u2[p]
        end
        return u2
    end

    @testset "variance channel — $name" for (name, f, x0) in PROBLEMS
        t, outidx = tape_function(f, x0)
        res = cover_hessian(WithDiagnostics(), t, outidx)
        vc = cover_variance(t, outidx)

        @test vc.grad ≈ res.grad
        @test vc.h_sum ≈ res.h_sum rtol=1e-12             # the hard channel is the sum merge
        @test all(>=(0), vc.h_var)
        # Quadrature over paths, then over nodes, then summation: h_var <= h_quad <= h_sum.
        @test all(vc.h_var .<= res.h_quad .* GATE)
        @test all(res.h_quad .<= vc.h_sum .* GATE)
        # The two channels carry seed weight along the same paths, so they vanish together.
        @test (vc.h_var .== 0) == (vc.h_sum .== 0)
        @test all(vc.severity .>= 1 / GATE)
        @test all(isfinite, vc.severity)

        rhos, _ = local_covers(t, adjoints(t, outidx))
        theta2 = variance_seeds(t, rhos)
        u2 = squared_sensitivities(t)
        ref = [sqrt(sum(theta2[j] * u2[j][a] for j in 1:nnodes(t))) for a in 1:t.ninputs]
        @test vc.h_var ≈ ref rtol=1e-12
    end

    @testset "variance channel is exact on tree-structured graphs" begin
        # Each input reaches each node along a single path, and no node has two parents sharing an
        # input, so there is no covariance to drop: the estimate attains the quadrature merge.
        for (f, x0) in ((tree_program, TREE_X), (quad_factored(WQ), XQ))
            @test cover_variance(f, x0).h_var ≈
                  cover_hessian(WithDiagnostics(), f, x0).h_quad rtol=1e-12
        end
        # One nonlinear node carries the whole Hessian of the factored quadratic, so the estimate
        # lands on the exact minimal cover.
        @test cover_variance(quad_factored(WQ), XQ).h_var ≈ abs.(WQ)
    end

    @testset "variance channel underestimates where paths reconverge" begin
        # The entrywise quadratic routes each input through n nonlinear nodes whose contributions
        # overlap — exactly the covariances the independence assumption drops.
        f, x0 = quad_entrywise(AQ), XQ
        @test all(cover_variance(f, x0).h_var .< cover_hessian(WithDiagnostics(), f, x0).h_quad)
        @test maximum(cover_variance(f, x0).severity) > 1.5

        # The estimate is not a cover. The diamond makes that concrete: its quadrature merge
        # attains the minimal cover [3, 1], so any undershoot loses the guarantee. x₁ reaches the
        # squaring node along two paths of strength 2 and 1, which the variance channel combines
        # in quadrature as sqrt(5) instead of summing to 3.
        H = ForwardDiff.hessian(reconvergent_diamond, [0.4, -0.9])
        res = cover_hessian(WithDiagnostics(), reconvergent_diamond, [0.4, -0.9])
        vc = cover_variance(reconvergent_diamond, [0.4, -0.9])
        @test H ≈ [9.0 -3.0; -3.0 1.0]
        @test res.h_quad ≈ [3.0, 1.0]
        @test cover_ratio(H, res.h_quad) ≈ 1
        @test vc.h_var ≈ [sqrt(5.0), 1.0]
        @test cover_ratio(H, vc.h_var) ≈ 9 / 5     # the (1,1) entry is no longer covered
    end

    @testset "variance channel covariance" begin
        cases = [("rosenbrock n=2", rosenbrock, [-1.2, 1.0], [1e-6, 1e3]),
                 ("quad matvec", quad_matvec(AQ), XQ, [1e-6, 1e3, 1.0, 1e-2, 1e4]),
                 ("logsumexp n=8", logsumexp,
                  [-3.0, -1.0, -2.0, 0.0, 1.0, 2.0, -0.5, 3.0],
                  [1e-3, 1e2, 1.0, 1e-1, 1e4, 1e-2, 1e1, 1e3])]
        @testset "$name" for (name, f, x0, d) in cases
            vc = cover_variance(f, x0)

            # Rescaling the inputs puts a factor 1/d[a] on each input edge, whose square the
            # variance channel picks up; the severity, a ratio of the two channels, is invariant.
            vcd = cover_variance(xt -> f(xt ./ d), d .* x0)
            @test vcd.h_var ≈ vc.h_var ./ d rtol=1e-12
            @test vcd.severity ≈ vc.severity rtol=1e-12

            # f -> c*f scales every rho by sqrt(c), hence every squared seed by c.
            c = 7.3
            vcc = cover_variance(x -> c * f(x), x0)
            @test vcc.h_var ≈ sqrt(c) .* vc.h_var rtol=1e-12
            @test vcc.severity ≈ vc.severity rtol=1e-12
        end
    end

    @testset "dependency sweep and severity" begin
        t, outidx = tape_function(x -> exp(x[1] * x[2]), [0.5, -2.0])
        N = nnodes(t)
        theta = Float64[k == outidx ? 1.0 : 0.0 for k in 1:N]

        # A zero variance seed leaves the hard channel exactly the single-channel sweep.
        abar, vbar = dependency_sweep(t, theta, zeros(N))
        @test abar ≈ abs_reverse(t, theta)
        @test all(iszero, vbar)
        # Seeding only the variance channel squares the coefficients along the way.
        _, vbar = dependency_sweep(t, zeros(N), theta)
        @test vbar[1] ≈ (t.ds[outidx][1] * t.ds[outidx - 1][1])^2

        @test_throws DimensionMismatch dependency_sweep(t, theta, zeros(N + 1))
        @test_throws DimensionMismatch dependency_sweep(t, zeros(N - 1), zeros(N))

        @test dependency_severity([2.0, 0.0], [1.0, 0.0]) == [2.0, 1.0]
        @test dependency_severity([1.0], [0.0]) == [Inf]
        @test_throws DimensionMismatch dependency_severity([1.0], [1.0, 2.0])
    end

    @testset "hvp — $name" for (name, f, x0) in PROBLEMS
        H = ForwardDiff.hessian(f, x0)
        v = Float64[cos(3i) for i in eachindex(x0)]
        @test hvp(f, x0, v) ≈ H * v rtol=1e-10
        @test hvp(f, x0, zero(x0)) ≈ zero(x0) atol=1e-14
        @test_throws DimensionMismatch hvp(f, x0, [1.0; v])
    end

    @testset "spectral shrink — $name" for (name, f, x0) in PROBLEMS
        H = ForwardDiff.hessian(f, x0)
        t, outidx = tape_function(f, x0)
        h = cover_hessian(WithDiagnostics(), t, outidx).h_sum

        # Every entry of the scaled Hessian is bounded by its spectral norm.
        lambda = scaled_spectral_radius(H, h)
        @test lambda >= cover_ratio(H, h) * (1 - 1e-12)

        hs = spectral_shrink(h, lambda)
        @test cover_ratio(H, hs) <= GATE                     # shrinking preserves validity
        @test all(hs .<= h .* GATE)                          # and never inflates
        if lambda < 1
            @test hs ≈ h .* sqrt(lambda)
            @test all(hs[h .> 0] .< h[h .> 0])
        else
            @test hs == h
        end

        # The matrix-free estimate approaches the exact norm from below, where it applies at all:
        # a coordinate the Hessian ignores gets a zero cover, which the power iteration rejects.
        if all(>(0), h)
            est = scaled_spectral_radius(t, outidx, h; maxiter=1000, tol=1e-10)
            @test est <= lambda * GATE
            @test est ≈ lambda rtol=1e-4
        else
            @test_throws "strictly positive" scaled_spectral_radius(t, outidx, h)
        end
    end

    @testset "spectral shrink algebra" begin
        H = [4.0 -2.0; -2.0 1.0]
        h = [2.0, 1.0]
        # Hhat = [1 -1; -1 1] has spectral norm 2, so this cover is spectrally saturated.
        @test scaled_spectral_radius(H, h) ≈ 2
        @test spectral_shrink(h, 2.0) == h
        @test spectral_shrink(h, 0.25) ≈ h ./ 2
        @test spectral_shrink(h, 0.25; safety=4) == h        # safety = 4 saturates the cover

        # The scaled Hessian, and hence the shrink factor, is invariant under both rescalings.
        d = [1e-3, 1e5]
        @test scaled_spectral_radius(H .* (d * d'), h .* d) ≈ scaled_spectral_radius(H, h)
        @test scaled_spectral_radius(9 .* H, 3 .* h) ≈ scaled_spectral_radius(H, h)

        # A coordinate the Hessian ignores may carry a zero cover; one it does not, may not.
        @test scaled_spectral_radius([1.0 0.0; 0.0 0.0], [1.0, 0.0]) ≈ 1
        @test scaled_spectral_radius(zeros(2, 2), [0.0, 0.0]) == 0
        @test_throws "does not cover" scaled_spectral_radius(H, [2.0, 0.0])

        t, outidx = tape_function(rosenbrock, [-1.2, 1.0])
        @test_throws "strictly positive" scaled_spectral_radius(t, outidx, [1.0, 0.0])
        @test_throws "did not converge" scaled_spectral_radius(t, outidx, [1.0, 1.0];
                                                               maxiter=1, tol=1e-12)
        @test_throws ArgumentError spectral_shrink(h, -1.0)
        @test_throws ArgumentError spectral_shrink(h, 1.0; safety=0)
    end

    @testset "metrics" begin
        H = [4.0 -2.0; -2.0 1.0]
        @test cover_ratio(H, [2.0, 1.0]) ≈ 1
        @test cover_ratio(H, [4.0, 2.0]) ≈ 0.25
        @test cover_ratio(H, [0.0, 1.0]) == Inf
        @test cover_ratio(zeros(2, 2), [0.0, 0.0]) == 0     # nothing to cover

        B = [4.0 2.0; 2.0 2.0]
        @test bound_ratio(H, B) ≈ 1                         # attained on the (1,1) entry
        @test rank1_ratio(B, [2.0, 1.0]) ≈ 2                # B[2,2] = 2 exceeds h[2]^2 = 1

        mx, gm, nextra = tightness([2.0, 8.0, 1.0], [1.0, 2.0, 0.0])
        @test mx ≈ 4
        @test gm ≈ sqrt(2 * 4)
        @test nextra == 1
        @test all(isnan, tightness([1.0], [0.0])[1:2])

        @test_throws DimensionMismatch cover_ratio(H, [1.0])
        @test_throws DimensionMismatch bound_ratio(H, zeros(3, 3))
        @test_throws DimensionMismatch tightness([1.0], [1.0, 2.0])
    end

    # The generated problems are the substrate of the growth and tolerance studies, so what is
    # tested here is that each family is the function it claims to be, has the graph structure it
    # claims to have, and stays within the covering chain.
    GENERATED = [(sigma, g(n))
                 for sigma in (0.0, 2.0)
                 for (_, g) in families(; sigma, seed=7)
                 for n in (3, 6)]

    @testset "generated problems are well-formed — $(p.name) sigma=$sigma" for (sigma, p) in GENERATED
        @test nvars(p) == length(p.x0)
        @test isfinite(p.f(p.x0))
        @test all(isfinite, ForwardDiff.gradient(p.f, p.x0))
        # A problem with no curvature poses no covering problem at all, and a family that
        # degenerates to one as `n` grows would silently make every method look perfect.
        @test any(!iszero, ForwardDiff.hessian(p.f, p.x0))
        # The twin computes the same function by an independent route (array operations the
        # scalarized program cannot express), so agreement is evidence the scalarization is right.
        if has_twin(p)
            @test p.twin(p.x0) ≈ p.f(p.x0)
            @test ForwardDiff.gradient(p.twin, p.x0) ≈ ForwardDiff.gradient(p.f, p.x0)
        end
        if has_exact_hessian(p)
            @test exact_hessian(p) ≈ ForwardDiff.hessian(p.f, p.x0)
        end
    end

    @testset "covering chain on generated problems — $(p.name) sigma=$sigma" for (sigma, p) in GENERATED
        H = ForwardDiff.hessian(p.f, p.x0)
        res = cover_hessian(WithDiagnostics(), p.f, p.x0)
        @test res.grad ≈ ForwardDiff.gradient(p.f, p.x0)
        @test bound_ratio(H, res.B) <= GATE
        for h in (res.h_sum, res.h_quad, cover_hybrid(p.f, p.x0, 3))
            @test all(>=(0), h)
            @test cover_ratio(H, h) <= GATE
        end
    end

    @testset "quadratic forms share a Hessian and differ as programs" begin
        n = 5
        ps = [quadratic(n; form=fm, rng=MersenneTwister(3)) for fm in (:entrywise, :matvec, :factored)]
        A = exact_hessian(ps[1])
        @test issymmetric(A)
        @test all(p -> exact_hessian(p) ≈ A, ps)          # one problem, three programs
        @test all(p -> p.x0 ≈ ps[1].x0, ps)
        # The factored form's rank-many nonlinear nodes do not overlap, so its cover is far
        # tighter than the entrywise form's n² overlapping ones.
        hs = [cover_hessian(WithDiagnostics(), p.f, p.x0).h_sum for p in ps]
        @test all(hs[3] .< hs[1])

        # rank 1: H = w*w' and the minimal cover abs.(w) = sqrt.(diag(A)) is attained.
        p1 = quadratic(6; rank=1, form=:factored, rng=MersenneTwister(4))
        @test rank(exact_hessian(p1)) == 1
        @test cover_hessian(WithDiagnostics(), p1.f, p1.x0).h_quad ≈ sqrt.(diag(exact_hessian(p1)))
    end

    @testset "random_dag sharing controls reconvergence" begin
        # sharing = 0 consumes each node exactly once, so the graph is a tree: every input reaches
        # every node along a single path. The variance channel drops covariances between
        # reconverging paths, so it attains the quadrature merge exactly iff there are none.
        tree = random_dag(8; nops=40, sharing=0.0, rng=MersenneTwister(11))
        @test cover_variance(tree.f, tree.x0).h_var ≈
              cover_hessian(WithDiagnostics(), tree.f, tree.x0).h_quad rtol=1e-10

        # With reuse, paths reconverge; the variance channel drops exactly those covariances and
        # so falls strictly below the quadrature merge on the coordinates that reconverge.
        shared = random_dag(8; nops=40, sharing=0.6, rng=MersenneTwister(11))
        hv = cover_variance(shared.f, shared.x0).h_var
        hq = cover_hessian(WithDiagnostics(), shared.f, shared.x0).h_quad
        @test all(hv .<= hq .* GATE)
        @test count(hv .< hq .* (1 - 1e-8)) >= 2

        # Each input reaches the output: nothing is dropped on the floor.
        @test all(!iszero, ForwardDiff.gradient(shared.f, shared.x0))

        # Values stay bounded no matter where the recipe is replayed, so `*` and `^2` chains
        # cannot overflow.
        @test isfinite(shared.f(100 .* shared.x0))
    end

    @testset "random_dag keeps every derivative alive at scale" begin
        # tanh saturates: `1 - tanh(v)^2` evaluates to exactly zero in Float64 for |v| ≳ 19,
        # and a derivative-dead node erases its whole subtree from the gradient and the
        # Hessian. At sharing = 0 the program's tail is a single unary chain (binary ops need
        # two unconsumed nodes and run out), so one saturated tanh there zeroes *every*
        # derivative of the program. The generator normalizes tanh inputs to prevent this;
        # small-n spot checks miss it because the chance of an untamed input grows with nops.
        function tanhderivs(r, x0)
            v = Vector{Float64}(undef, r.ninputs + length(r.ops))
            v[1:r.ninputs] = x0
            ds = Float64[]
            for (k, (op, ps, cs)) in enumerate(r.ops)
                v[r.ninputs + k] = DerivativeCovers.Problems._apply(op, v, ps, cs)
                op === :tanh && push!(ds, 1 - tanh(v[ps[1]])^2)
            end
            return ds
        end
        for sharing in (0.0, 0.6), n in (16, 32, 64, 128, 256, 512)
            p = random_dag(n; sharing, rng=MersenneTwister(1))
            @test all(>(1e-3), tanhderivs(p.f, p.x0))
            g = ForwardDiff.gradient(p.f, p.x0)
            H = ForwardDiff.hessian(p.f, p.x0)
            @test any(!iszero, H)
            # An input can be genuinely dead — consumed only through linear paths whose unit
            # weights cancel exactly in the output sum — but then it must be dead at second
            # order too, and deadness stays the rare exception.
            dead = findall(iszero, g)
            @test all(i -> all(iszero, view(H, i, :)), dead)
            @test length(dead) <= max(1, n ÷ 8)
        end
    end

    @testset "rescale is exactly the input-units transformation" begin
        p = quadratic(4; rng=MersenneTwister(5))
        d = logscales(MersenneTwister(6), 4; sigma=4.0)
        @test maximum(d) / minimum(d) > 1e2                # a genuinely wide spread
        q = rescale(p, d)

        @test q.x0 ≈ d .* p.x0
        @test q.f(q.x0) ≈ p.f(p.x0)
        @test exact_hessian(q) ≈ ForwardDiff.hessian(q.f, q.x0)
        @test exact_hessian(q) ≈ Diagonal(inv.(d)) * exact_hessian(p) * Diagonal(inv.(d))
        # The cover is covariant under it: this is the property the whole method rests on.
        @test cover_hessian(WithDiagnostics(), q.f, q.x0).h_sum ≈
              cover_hessian(WithDiagnostics(), p.f, p.x0).h_sum ./ d rtol=1e-10

        @test_throws DimensionMismatch rescale(p, [1.0, 2.0])
        @test_throws ArgumentError rescale(p, [1.0, 2.0, 0.0, 1.0])
    end

    @testset "sigma widens the coefficient dynamic range" begin
        # `sigma` is defined as the standard deviation of the coefficients' log-magnitudes, so
        # read the dynamic range the same way. A `maximum(abs) / minimum(abs)` ratio instead
        # tracks the single smallest entry, a near-cancellation artifact that swings by orders
        # of magnitude between draws and says little about the bulk.
        function logspread(A)
            l = log.(abs.(vec(A)))
            m = sum(l) / length(l)
            return sqrt(sum(abs2, l .- m) / length(l))
        end
        narrow = exact_hessian(quadratic(32; rng=MersenneTwister(2), sigma=0.0))
        wide = exact_hessian(quadratic(32; rng=MersenneTwister(2), sigma=3.0))
        @test logspread(wide) > 1.5 * logspread(narrow)

        @test_throws ArgumentError quadratic(4; rank=5)
        @test_throws ArgumentError quadratic(4; form=:bogus)
        @test_throws ArgumentError deep_composition(4; width=0)
        @test_throws ArgumentError random_dag(4; sharing=1.5)
    end

    @testset "every pattern inflates its worst coordinate to exactly the peak" begin
        n, peak = 12, 4.0
        patterns = [UniformInflation(), LognormalInflation(), SpikeInflation(5), RampInflation()]
        for p in patterns
            f = inflation_factors(p, n, peak)
            @test length(f) == n
            @test all(x -> 1 <= x <= peak, f)
            @test maximum(f) ≈ peak
            # A peak of 1 is the identity distortion, whatever the shape.
            @test inflation_factors(p, n, 1.0) ≈ ones(n)
        end
        @test_throws ArgumentError inflation_factors(UniformInflation(), n, 0.5)
    end

    @testset "each pattern realizes the profile it names" begin
        n, peak = 16, 10.0

        f = inflation_factors(UniformInflation(), n, peak)
        @test f ≈ fill(peak, n)
        @test anisotropy(f) ≈ 0 atol=1e-12

        f = inflation_factors(SpikeInflation(3), n, peak)
        @test f[3] ≈ peak
        @test f[setdiff(1:n, 3)] ≈ ones(n - 1)
        @test anisotropy(f) ≈ log(peak) * (1 - 1 / n)
        @test_throws BoundsError inflation_factors(SpikeInflation(n + 1), n, peak)

        f = inflation_factors(RampInflation(), n, peak)
        @test issorted(f)
        @test f[1] ≈ 1
        @test f[end] ≈ peak
        @test anisotropy(f) ≈ log(peak) / 2 atol=0.05
        @test inflation_factors(RampInflation(), 1, peak) ≈ [peak]

        f = inflation_factors(LognormalInflation(), n, peak; rng=MersenneTwister(3))
        @test f ≈ inflation_factors(LognormalInflation(), n, peak; rng=MersenneTwister(3))
        @test f != inflation_factors(LognormalInflation(), n, peak; rng=MersenneTwister(4))
        # Its head is more concentrated than a ramp's: the bulk of a normal draw sits well below
        # its maximum, and the gap widens with n.
        big = 256
        lgn = inflation_factors(LognormalInflation(), big, peak; rng=MersenneTwister(3))
        @test anisotropy(lgn) > anisotropy(inflation_factors(RampInflation(), big, peak))
    end

    @testset "an inflated cover is still a cover" begin
        p = quadratic(6; form=:entrywise, rng=MersenneTwister(7))
        H = exact_hessian(p)
        h = cover_hessian(WithDiagnostics(), p.f, p.x0).h_sum
        @test cover_ratio(H, h) <= 1

        for pat in (UniformInflation(), LognormalInflation(), SpikeInflation(2), RampInflation())
            f = inflation_factors(pat, h, 3.0; rng=MersenneTwister(8))
            hd = inflate(h, f)
            # Inflation can only weaken the bound, never break it...
            @test cover_ratio(H, hd) <= cover_ratio(H, h)
            @test cover_ratio(H, hd) <= 1
            # ...and the looseness it adds is exactly the factors that were asked for.
            @test hd ./ h ≈ f
            @test tightness(hd, h)[1] ≈ maximum(f)
        end

        @test inflate(h, UniformInflation(), 2.5) ≈ 2.5 .* h
        @test_throws ArgumentError inflate(h, fill(0.5, length(h)))
        @test_throws DimensionMismatch inflate(h, ones(length(h) + 1))
    end

    @testset "inflation leaves a zero cover coordinate at zero" begin
        # A cover of a Hessian that ignores an input has an exactly-zero coordinate there; no
        # multiplicative distortion can loosen a bound that is already exact.
        H = [2.0 1.0 0.0; 1.0 3.0 0.0; 0.0 0.0 0.0]
        h = [2.0, 2.0, 0.0]
        @test cover_ratio(H, h) <= 1

        f = inflation_factors(LognormalInflation(), h, 5.0; rng=MersenneTwister(9))
        hd = inflate(h, f)
        @test hd[3] == 0
        @test cover_ratio(H, hd) <= 1
        @test tightness(hd, h)[3] == 0          # no coordinate the reference leaves at zero
    end

    @testset "inflation follows the cover's axes" begin
        h = [1.0, 2.0, 3.0, 4.0, 5.0]
        v = view(h, 2:4)
        f = inflation_factors(RampInflation(), v, 4.0)
        @test axes(f) == axes(v)
        @test f ≈ [1.0, 2.0, 4.0]
        @test inflate(v, f) ≈ [2.0, 6.0, 16.0]
        @test inflation_shape(SpikeInflation(2), v) ≈ [0.0, 1.0, 0.0]
        @test_throws BoundsError inflation_shape(SpikeInflation(4), v)
    end

    @testset "the convergence test is scale-invariant" begin
        g, h = [3.0, 4.0], [1.0, 2.0]
        @test cover_norm(g, h) ≈ sqrt(13)
        D = Diagonal([1e3, 1e-4])
        @test cover_norm(D * g, D * h) ≈ cover_norm(g, h)
        @test cover_norm(2 * g, 2 * h) ≈ cover_norm(g, h)     # objective scaling f -> c*f
        # A dead coordinate: no curvature to divide by.
        @test cover_norm([1.0, 0.0], [0.0, 3.0]) == Inf
        @test cover_norm([0.0, 3.0], [0.0, 3.0]) ≈ 1
        @test_throws DimensionMismatch cover_norm([1.0], [1.0, 2.0])
    end

    @testset "an inflated cover stops the run early" begin
        # A gradient halving each iteration, against a cover held fixed along the path.
        grads = [[2.0^-k, 0.0] for k in 0:10]
        covers = [ones(2) for _ in 0:10]
        dcovers = [inflate(h, UniformInflation(), 4.0) for h in covers]
        tol = 0.1
        # 2^-k <= 0.1 first at k = 4, and 2^-k <= 0.4 (the inflated test) first at k = 2.
        @test stopping_index(grads, covers, tol) == 5
        @test stopping_index(grads, dcovers, tol) == 3
        s = stopping_shift(grads, covers, dcovers, tol)
        @test s.k_ref == 5 && s.k_dist == 3
        @test s.shift == 2
        @test s.overshoot ≈ 0.25 / tol      # the true cover-norm where the run declared success
        @test stopping_shift(grads, covers, covers, tol).shift == 0
        @test stopping_index(grads, covers, 1e-9) === nothing
        @test_throws "never fires" stopping_shift(grads, covers, dcovers, 1e-9)
    end

    @testset "regularization bias" begin
        A, g = Matrix(1.0I, 2, 2), [1.0, 1.0]
        h, hd = [1.0, 1.0], [2.0, 1.0]
        @test regularized_step(A, g, h, 1.0) ≈ [-0.5, -0.5]
        @test regularized_step(A, g, hd, 1.0) ≈ [-0.2, -0.5]    # diag(5, 2) \ (-g)
        @test regularization_bias(A, g, h, hd, 1.0) ≈ 0.3 / (0.5 * sqrt(2))
        @test regularization_bias(A, g, h, h, 1.0) == 0
        @test regularized_step(A, g, h, 0.0) ≈ A \ (-g)
        @test_throws ArgumentError regularized_step(A, g, h, -1.0)
    end

    @testset "regularization bias is scale-invariant" begin
        rng = MersenneTwister(7)
        A = Symmetric(randn(rng, 5, 5)) |> Matrix
        g = randn(rng, 5)
        h = abs.(randn(rng, 5)) .+ 0.5
        hd = inflate(h, LognormalInflation(), 8.0; rng)
        b = regularization_bias(A, g, h, hd, 0.5)
        # Variables: x -> D*x sends g -> D*g, A -> D*A*D, and any cover h -> D*h.
        D = Diagonal([1e3, 1e-2, 1.0, 7.0, 1e-4])
        @test regularization_bias(D * A * D, D * g, D * h, D * hd, 0.5) ≈ b
        # Objective: f -> c*f sends g -> c*g, A -> c*A, h -> sqrt(c)*h.
        c = 100.0
        @test regularization_bias(c * A, c * g, sqrt(c) * h, sqrt(c) * hd, 0.5) ≈ b
    end

    @testset "PSD boost feels anisotropy only" begin
        A = [1.0 0.0; 0.0 -1.0]
        @test psd_shift(A, [1.0, 1.0]) ≈ 1
        @test psd_shift(A, [2.0, 1.0]) ≈ 1          # scaling coordinate 1 cannot fix coordinate 2
        @test psd_shift(Matrix(1.0I, 2, 2), [3.0, 0.5]) == 0

        uni = psd_excess(A, [1.0, 1.0], [2.0, 2.0])
        @test uni.tau_ref ≈ 1 && uni.tau_dist ≈ 0.25
        @test uni.excess ≈ [1.0, 1.0]               # tau*Diagonal(h.^2) is unchanged by h -> c*h

        spike = psd_excess(A, [1.0, 1.0], [10.0, 1.0])
        @test spike.tau_dist ≈ 1
        @test spike.excess ≈ [100.0, 1.0]           # the untouched coordinate is shifted 100x too hard

        psd = psd_excess(Matrix(1.0I, 2, 2), [1.0, 1.0], [10.0, 1.0])
        @test psd.tau_ref == 0 && psd.tau_dist == 0 && psd.excess ≈ [1.0, 1.0]

        @test_throws "strictly positive" psd_shift(A, [0.0, 1.0])
        @test_throws "symmetric" psd_shift([1.0 2.0; 3.0 4.0], [1.0, 1.0])
    end

    @testset "the PSD shift is the minimal one" begin
        rng = MersenneTwister(11)
        for _ in 1:5
            A = Matrix(Symmetric(randn(rng, 6, 6)))
            h = exp.(randn(rng, 6))                 # a wide range of curvature scales
            tau = psd_shift(A, h)
            @test tau > 0                           # a random symmetric matrix is indefinite
            @test eigmin(Symmetric(A + tau * Diagonal(h .^ 2))) > -1e-10
            @test eigmin(Symmetric(A + 0.99tau * Diagonal(h .^ 2))) < 0
        end
    end

    @testset "cover-preconditioned CG" begin
        A = Matrix(Diagonal([1.0, 4.0, 9.0]))
        b = [1.0, 1.0, 1.0]
        h = [1.0, 2.0, 3.0]                         # the exact cover of this A
        exact = cover_cg(A, b, h)
        @test exact.x ≈ A \ b
        @test exact.iters == 1                      # the scaled system is the identity

        # Uniform inflation rescales the metric without reshaping it: still one iteration.
        @test cover_cg(A, b, inflate(h, UniformInflation(), 5.0)).iters == 1
        # A spike leaves the scaled system with two distinct eigenvalues.
        spiked = cover_cg(A, b, inflate(h, SpikeInflation(1), 10.0))
        @test spiked.iters == 2
        @test spiked.x ≈ A \ b

        @test_throws "strictly positive" cover_cg(A, b, [0.0, 2.0, 3.0])
    end

    @testset "cover quality is step quality under a truncated CG" begin
        rng = MersenneTwister(3)
        M = randn(rng, 12, 12)
        A = M * M' + 12I                            # symmetric positive definite
        g = randn(rng, 12)
        h = sqrt.(diag(A))                          # a cover of a diagonally dominant A
        hd = inflate(h, LognormalInflation(), 1e4; rng)

        # Preconditioning moves the path, not the fixed point: given iterations enough, both runs
        # reach the target, and the loose cover pays for it in iterations.
        long = step_degradation(A, g, h, hd; itermax=40)
        @test long.error_ref < 1e-6 && long.error_dist < 1e-6
        @test long.iters_dist > long.iters_ref

        # On the usual budget of one iteration per coordinate the loose cover simply fails to
        # deliver a step: it does not reach the target, and its final iterate is orders of
        # magnitude further from exact than the reference run's. The gap between the two runs is
        # the property; where each lands in absolute terms depends on the draw.
        budget = step_degradation(A, g, h, hd)
        @test budget.iters_ref == long.iters_ref
        @test budget.iters_dist === nothing
        @test budget.error_dist > 1e3 * budget.error_ref

        # Uniform inflation is invisible here too: only the shape of the metric steers CG.
        uni = step_degradation(A, g, h, inflate(h, UniformInflation(), 50.0); itermax=4)
        @test uni.error_dist ≈ uni.error_ref
    end

end
