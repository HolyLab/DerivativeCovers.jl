using DerivativeCovers
using DerivativeCovers.Diagnostics
# Load the unregistered sibling package directly; test targets cannot declare path dependencies.
include(
    joinpath(
        @__DIR__, "..", "..", "DerivativeCoverProblems", "src",
        "DerivativeCoverProblems.jl"
    )
)
using .DerivativeCoverProblems
const Problems = DerivativeCoverProblems
# Import internal helpers explicitly to keep test call sites readable without expanding exports.
using DerivativeCovers: CoverVar, submapnode, value, isconstant, nnodes, natoms, cover_ratio,
    bound_ratio,
    rank1_ratio, tightness, jacobian_ratio, jacobian_representability,
    jacobian_minimax, dependency_severity, scaled_spectral_radius,
    scaled_singular_value, spectral_shrink, jacobian_shrink, equilibrate,
    compose_hessian_cover, compose_jacobian_cover, gn_ab, nls_loss_tape,
    declare_submap, submap, cyclic_partition, level_partition, node_levels,
    nls_partition, adjoints, sensitivities, abs_reverse, abs_forward,
    local_covers, balanced_split, variance_seeds, dependency_sweep,
    jacobian_surrogate, marginal_weights, balance_cover, tape_potentials,
    MergeRule, SumMerge, QuadMerge, MaxMerge
# This package adds `Tape` methods to these MatrixCovers functions but does not re-export them.
using MatrixCovers: support_components, gramcover, gramcover!, SupportComponents, ncomponents,
    rowcomponent, colcomponent
using Aqua
using ExplicitImports
using ForwardDiff
using LinearAlgebra
using OffsetArrays
using Random
using SparseArrays: sparse, sparsevec
using StaticArrays
using Test

include("programs.jl")

# The covering chain abs.(H) <= B <= sum_k s_k s_k' <= h*h' is exact in exact arithmetic;
# roundoff can push a measured ratio a hair above 1.
const GATE = 1 + 1.0e-10

include("constants.jl")
include("elementtype.jl")
include("primitives.jl")
include("composites.jl")
include("arrayops.jl")
include("mlp.jl")
include("cancellation.jl")
include("lattice.jl")
include("forward.jl")
include("nls.jl")
include("vectoroutput.jl")
include("support.jl")
include("jacobian.jl")
include("covariant.jl")
include("gram.jl")
include("jacobianconverse.jl")
include("equilibrate.jl")
include("compose.jl")
include("paired.jl")
include("genericaxes.jl")
include("intervals.jl")

@testset "DerivativeCovers.jl" begin

    @testset "Aqua" begin
        Aqua.test_all(DerivativeCovers)
    end

    @testset "ExplicitImports" begin
        # Julia 1.10 cannot distinguish public, unexported names, so public-access checks would
        # incorrectly reject Base.broadcastable.
        public_ok = VERSION >= v"1.11"
        # `generic_matvecmul!` and `generic_matmatmul!` are where a package supplies its own
        # product rules, and `wrap` and `MulAddMul` appear in their signatures. They are slated to
        # become public; until the release that declares them, every version this package supports
        # needs them excused here. The SparseArrays types below resolve product-method ties.
        test_explicit_imports(
            DerivativeCovers;
            all_explicit_imports_are_public = public_ok,
            all_qualified_accesses_are_public = public_ok ?
                (;
                    ignore = (
                        :broadcastable, :generic_matvecmul!, :generic_matmatmul!, :wrap, :MulAddMul,
                        :SparseVectorUnion, :AbstractSparseMatrixCSC, :UpperOrLowerTriangular,
                    ),
                ) : false
        )
    end

    @testset "public API boundary" begin
        # A backend variable observes operations rather than standing in for a number: it must not
        # be a Real, so that generic `::Real` methods reject it instead of silently recording
        # something the package has no rule for.
        @test !(CoverVar <: Real)
        @test TVar <: CoverVar

        # Pin the export set so internal names cannot drift into the top-level namespace.
        exported = filter(s -> Base.isexported(DerivativeCovers, s), names(DerivativeCovers))
        @test Set(setdiff(exported, [:DerivativeCovers])) == Set(
            [
                :Tape, :TVar, :tape_function,
                :FVar, :forward_cover, :ForwardCover,
                :cover_hessian, :cover_jacobian, :cover_nls, :cover_variance,
                :cover_hybrid, :cover_jacobian_hybrid, :cover_lattice,
                :CoverOnly, :WithDiagnostics,
                :TangentChannel, :AbsoluteTangents, :SignedTangents,
                :CoverLattice, :HessianCover, :HessianDiagnostics,
                :JacobianCover, :JacobianDiagnostics, :NLSCover,
                :NLSDiagnostics, :VarianceCover,
                :hvp, :hvp!, :jvp, :jvp!, :vjp, :vjp!,
            ]
        )

        # Public extension points and utilities remain available only by qualification.
        internals = (
            :pushnode!, :unarynode, :binarynode, :dotnode, :linearnode, :noderef,
            :constantnode, :atom_slots, :check_finite_partials,
            :submapnode, :split_scales, :split_block, :split_ratio, :guards_any_block,
            :sensitivity_scales, :SplitScale, :dead_slot, :value, :isconstant, :nnodes, :natoms,
            :CoverVar, :constantvar, :tangent_channel, :forward_scale,
            :magnitude, :magtype, :separation, :check_values,
            :Curvature, :NoCurvature, :DenseCurvature, :PairedCurvature, :DeclaredCurvature,
            :cover_slots, :cover_slots!, :bound_contribution!, :curvature_apply!, :has_exact_partials,
            :CoverSite, :TapeSite, :site_label, :site_values,
            :cover_ratio, :bound_ratio, :rank1_ratio, :tightness, :jacobian_ratio,
            :jacobian_representability, :jacobian_minimax, :dependency_severity,
            :scaled_spectral_radius, :scaled_singular_value, :spectral_shrink,
            :jacobian_shrink, :equilibrate,
            :compose_hessian_cover, :compose_jacobian_cover, :gn_ab, :nls_loss_tape,
            :declare_submap, :submap,
        )
        for sym in internals
            @test isdefined(DerivativeCovers, sym)
            @test !Base.isexported(DerivativeCovers, sym)
            VERSION >= v"1.11" && @test Base.ispublic(DerivativeCovers, sym)
        end

        # The sweep machinery and the partition policies behind `cover_hybrid`'s integer argument
        # carry no promise: they are what a reimplementation of the sweeps is free to restructure.
        private = (
            :cyclic_partition, :level_partition, :node_levels, :nls_partition,
            :group_seeds, :adjoints, :sensitivities, :tangents, :abs_reverse, :abs_forward,
            :hessian_bound, :merge_atoms, :local_covers, :local_cover_slots, :balanced_split,
            :split_fallback,
            :value_bounds_split, :dispatchcurvature, :live_nodes,
            :variance_seeds, :dependency_sweep, :jacobian_surrogate, :marginal_weights,
            :MergeRule, :SumMerge, :QuadMerge, :MaxMerge,
            :balance_cover, :tape_potentials,
        )
        for sym in private
            @test isdefined(DerivativeCovers, sym)
            VERSION >= v"1.11" && @test !Base.ispublic(DerivativeCovers, sym)
        end

        # MatrixCovers' functions, which this package extends with `Tape` methods rather than
        # owning. Reaching them takes `using MatrixCovers`.
        for sym in (
                :support_components, :gramcover, :gramcover!, :SupportComponents,
                :ncomponents, :rowcomponent, :colcomponent,
            )
            @test !Base.isexported(DerivativeCovers, sym)
        end

        # The study instruments live in their own namespace, reached by
        # `using DerivativeCovers.Diagnostics`, and are not re-exported by the parent.
        diagnostics = (
            :InflationPattern, :UniformInflation, :LognormalInflation,
            :SpikeInflation, :RampInflation, :inflation_shape, :inflation_factors,
            :inflate, :anisotropy, :cover_norm, :stopping_index, :stopping_shift,
            :regularized_step, :regularization_bias, :psd_shift, :psd_excess,
            :cover_cg, :step_degradation,
        )
        for sym in diagnostics
            @test Base.isexported(DerivativeCovers.Diagnostics, sym)
            @test !isdefined(DerivativeCovers, sym)
        end
    end

    @testset "show summarizes rather than dumps the tape" begin
        f = xs -> sum(xs .^ 2)
        t, _ = tape_function(f, collect(1.0:60.0))
        x = TVar(t, 5)

        @test sprint(show, t) == "Tape($(nnodes(t)) nodes, 60 inputs)"
        @test occursin("$(nnodes(t)) nodes", sprint(show, MIME"text/plain"(), t))
        @test sprint(show, x) == "TVar(node 5, 5.0)"
        @test occursin("node 5", sprint(show, MIME"text/plain"(), x))
        @test occursin("constant", sprint(show, MIME"text/plain"(), TVar(2.5)))
        @test sprint(show, TVar(2.5)) == "TVar(2.5)"

        # An FVar carries its channels rather than a tape, but they are as long as the input
        # vector, so its displays report the input count instead of the channels themselves.
        function intermediate_fvar(n)
            captured = Ref{Any}(nothing)
            forward_cover(collect(1.0:n)) do zs
                y = zs[1] * zs[2]
                captured[] = y
                return y + sum(zs)
            end
            return captured[]
        end
        fv = intermediate_fvar(60)
        @test sprint(show, fv) == "FVar(2.0)"
        @test occursin("60 inputs", sprint(show, MIME"text/plain"(), fv))
        fconst = FVar{Float64, Vector{Float64}}(2.5)
        @test occursin("constant", sprint(show, MIME"text/plain"(), fconst))
        @test sprint(show, fconst) == "FVar(2.5)"

        # A TVar carries a reference to the whole tape, so the default display would grow with
        # the program. Both displays must stay bounded as the tape gets ten times longer.
        tbig, _ = tape_function(f, collect(1.0:600.0))
        for v in (tbig, TVar(tbig, 5), intermediate_fvar(600))
            @test length(sprint(show, v)) < 50
            @test length(sprint(show, MIME"text/plain"(), v)) < 50
        end

        x0 = [0.6, -0.4, 0.9]
        results = (
            cover_hessian(f, x0),
            cover_hessian(WithDiagnostics(), f, x0),
            cover_lattice(f, x0),
            forward_cover(f, x0),
            cover_jacobian(xs -> [xs[1] * xs[2], sin(xs[3])], x0),
            cover_jacobian(WithDiagnostics(), xs -> [xs[1] * xs[2], sin(xs[3])], x0),
            cover_nls(xs -> [xs[1] * xs[2] - 1, sin(xs[3])], x0),
            cover_nls(WithDiagnostics(), xs -> [xs[1] * xs[2] - 1, sin(xs[3])], x0),
            cover_variance(f, x0),
        )
        for c in results
            # Each names its own type and its dimensions, and holds vectors it must not print.
            for s in (sprint(show, c), sprint(show, MIME"text/plain"(), c))
                @test occursin(string(nameof(typeof(c))), s)
                @test occursin("3", s)
                @test length(s) < 100
            end
        end
        @test sprint(show, results[5]) == "JacobianCover(2x3)"
        @test sprint(show, results[6]) == "JacobianDiagnostics(2x3)"
        @test sprint(show, results[7]) == "NLSCover(2 residuals, 3 inputs)"
        @test sprint(show, results[8]) == "NLSDiagnostics(2 residuals, 3 inputs)"
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

    # Node indices and scalar magnitudes are annotated no more narrowly than the sweeps need, so
    # any `Integer` indexes an output and any `Real` supplies an absolute adjoint.
    @testset "index and magnitude arguments accept any Integer/Real" begin
        f = x -> x[1] * x[2] + exp(x[3])
        x0 = [2.0, -3.0, 0.5]
        t, outidx = tape_function(f, x0)
        oi = Int32(outidx)
        @test adjoints(t, oi) == adjoints(t, outidx)
        @test cover_hessian(t, oi).h_sum == cover_hessian(t, outidx).h_sum
        @test cover_variance(t, oi).h_var == cover_variance(t, outidx).h_var
        @test cover_hybrid(t, oi, 2) == cover_hybrid(t, outidx, 2)
        @test hvp(t, oi, [1.0, 0.0, 0.0]) == hvp(t, outidx, [1.0, 0.0, 0.0])
        h = cover_hessian(t, outidx).h_sum
        @test scaled_spectral_radius(t, oi, h; maxiter = Int32(50)) ≈
            scaled_spectral_radius(t, outidx, h; maxiter = 50)
        # `av` is a magnitude, so an integer-valued one is as admissible as a Float64.
        k = something(findfirst(C -> !(C isa DerivativeCovers.NoCurvature), t.Cs))
        @test DerivativeCovers.local_cover_slots(t, Int32(k), 1) ==
            DerivativeCovers.local_cover_slots(t, k, 1.0)
    end

    # For a single elemental, B = abs.(H), exposing both recorded derivative orders directly.
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

    @testset "an already-tight pair needs no balancing" begin
        # When C12² <= C11*C22, the diagonal square roots already cover the block.
        f = x -> x[1]^x[2]
        x0 = [0.5, 1.6]
        t, outidx = tape_function(f, x0)
        C = t.Cs[outidx]
        c11, c12, c22 = abs(C[1, 1]), abs(C[1, 2]), abs(C[2, 2])
        @test c12^2 <= c11 * c22
        @test DerivativeCovers.local_cover_slots(t, outidx, 1.0) == [sqrt(c11), sqrt(c22)]

        res = cover_hessian(WithDiagnostics(), f, x0)
        @test res.grad ≈ ForwardDiff.gradient(f, x0)
        @test cover_ratio(ForwardDiff.hessian(f, x0), res.h_sum) <= GATE
    end

    @testset "a tiny cross term is still off-diagonal" begin
        # The classification must not square c12: below the square root of the smallest
        # subnormal the square underflows to zero, and a multiplication block (c11 == c22 == 0)
        # with a representable cross term would be covered by (0, 0).
        @test DerivativeCovers.offdiag_active(0.0, 1.0e-162, 0.0)
        @test DerivativeCovers.offdiag_active(0.0, nextfloat(0.0), 0.0)
        @test DerivativeCovers.offdiag_active(0.0f0, 1.0f-30, 0.0f0)
        @test !DerivativeCovers.offdiag_active(1.0, 10.0, 100.0)
        @test DerivativeCovers.offdiag_active(1.0, 10.0 * (1 + 1.0e-8), 100.0)
        # Nor may it overflow: the squared form has both sides at Inf here.
        @test !DerivativeCovers.offdiag_active(1.0e300, 0.5e300, 1.0e300)
        @test DerivativeCovers.offdiag_active(1.0e300, 2.0e300, 1.0e300)

        # A running product of 541 halves: the product squared is below the smallest subnormal,
        # while the blocks along the chain carry cross terms around 1e-162.
        n = 541
        x0 = fill(0.5, n)
        f = x -> (prod(x) - 1)^2
        res = cover_hessian(WithDiagnostics(), f, x0)
        @test all(isfinite, res.h_sum)
        @test cover_ratio(ForwardDiff.hessian(f, x0), res.h_sum) <= GATE
        @test all(isfinite, cover_variance(f, x0).h_sum)

        # Deeper still, where the adjoints themselves approach the subnormal range.
        f2 = x -> prod(x)^2
        res2 = cover_hessian(WithDiagnostics(), f2, fill(0.5, 1000))
        @test all(h -> isfinite(h) && h > 0, res2.h_sum)
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
        @test !all(
            isfinite,
            cover_hessian(
                WithDiagnostics(), x -> x[1]^0.5 + x[1] * x[2], x0;
                check = false
            ).h_sum
        )
    end

    @testset "gradient matches ForwardDiff — $name" for (name, f, x0) in PROBLEMS
        res = cover_hessian(WithDiagnostics(), f, x0)
        @test res.grad ≈ ForwardDiff.gradient(f, x0)
    end

    # A test problem's declared dimension is its real one: every coordinate reaches the output.
    # A parameter that is dead in both the gradient and the Hessian contributes nothing, so the
    # problem silently exercises a smaller program than its `n` claims — and an indexing slip that
    # aliases one parameter onto another looks exactly like this while leaving the count right.
    @testset "no coordinate is dead — $name" for (name, f, x0) in PROBLEMS
        g = ForwardDiff.gradient(f, x0)
        H = ForwardDiff.hessian(f, x0)
        dead = [i for i in eachindex(g) if g[i] == 0 && all(iszero, view(H, i, :))]
        @test isempty(dead)
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
        @test res.h_sum_rev ≈ res.h_sum rtol = 1.0e-12
        # For H ⪰ 0 the minimal cover is sqrt.(diag(H)); no valid cover can go below it.
        if isposdef(H)
            @test all(res.h_quad .>= sqrt.(diag(H)) .* (1 - 1.0e-10))
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
        cases = [
            ("rosenbrock n=2", rosenbrock, [-1.2, 1.0], [1.0e-6, 1.0e3]),
            ("quad matvec", quad_matvec(AQ), XQ, [1.0e-6, 1.0e3, 1.0, 1.0e-2, 1.0e4]),
            (
                "logsumexp n=8", logsumexp,
                [-3.0, -1.0, -2.0, 0.0, 1.0, 2.0, -0.5, 3.0],
                [1.0e-3, 1.0e2, 1.0, 1.0e-1, 1.0e4, 1.0e-2, 1.0e1, 1.0e3],
            ),
        ]
        @testset "$name" for (name, f, x0, d) in cases
            res = cover_hessian(WithDiagnostics(), f, x0)

            # x -> D*x leaves every intermediate value and adjoint unchanged and puts a factor
            # 1/d[a] on each input edge, so h -> h ./ d exactly.
            resd = cover_hessian(WithDiagnostics(), xt -> f(xt ./ d), d .* x0)
            @test resd.h_sum ≈ res.h_sum ./ d rtol = 1.0e-12
            @test resd.h_quad ≈ res.h_quad ./ d rtol = 1.0e-12

            # f -> c*f scales every adjoint by c, hence every rho and every h by sqrt(c).
            c = 7.3
            resc = cover_hessian(WithDiagnostics(), x -> c * f(x), x0)
            @test resc.h_sum ≈ sqrt(c) .* res.h_sum rtol = 1.0e-12
            @test resc.h_quad ≈ sqrt(c) .* res.h_quad rtol = 1.0e-12
        end
    end

    @testset "balanced_split" begin
        for (c11, c12, c22, z1, z2) in (
                (0.0, 1.0, 0.0, 2.0, 5.0),      # multiplication node
                (1.0, 4.0, 2.0, 0.1, 10.0),
                (1.0, 3.0, 1.0, 0.0, 0.0),      # no ratio to read: the geometric mean
                (2.0, 5.0, 1.0, 1.0, 0.0),
            )
            rho1, rho2 = balanced_split(c11, c12, c22, z1, z2)
            @test rho1 >= sqrt(c11) * (1 - 1.0e-12)
            @test rho2 >= sqrt(c22) * (1 - 1.0e-12)
            @test rho1 * rho2 >= c12 * (1 - 1.0e-12)
        end
        # With both parent values positive the split is balanced against them: for a pure
        # multiplication node (c11 = c22 = 0), rho1/rho2 = z2/z1 and rho1*rho2 = c12.
        rho1, rho2 = balanced_split(0.0, 1.0, 0.0, 2.0, 8.0)
        @test rho1 / rho2 ≈ 8.0 / 2.0
        @test rho1 * rho2 ≈ 1.0

        # The split lands on the feasible segment [sqrt(c11), c12/sqrt(c22)], so the cover is
        # exactly tight on the off-diagonal however extreme the entries and values. A vanishing
        # reference leaves a determined split only where the endpoint it runs toward is a positive
        # finite scale, and where it does not there is no cover to return.
        determined(c11, c22, z1, z2) = (z1 > 0 || c22 > 0) && (z2 > 0 || c11 > 0)
        rng = MersenneTwister(11)
        for _ in 1:10_000
            c11 = rand(rng) < 0.25 ? 0.0 : exp(12 * randn(rng))
            c22 = rand(rng) < 0.25 ? 0.0 : exp(12 * randn(rng))
            c12 = exp(12 * randn(rng))
            c12^2 > c11 * c22 || continue
            z1 = rand(rng) < 0.1 ? 0.0 : exp(8 * randn(rng))
            z2 = rand(rng) < 0.1 ? 0.0 : exp(8 * randn(rng))
            if !determined(c11, c22, z1, z2)
                @test_throws "outside the floating-point range" balanced_split(
                    c11, c12, c22, z1, z2
                )
                continue
            end
            rho1, rho2 = balanced_split(c11, c12, c22, z1, z2)
            @test isfinite(rho1) && isfinite(rho2)
            @test rho1 >= sqrt(c11) * (1 - 1.0e-12)
            @test rho2 >= sqrt(c22) * (1 - 1.0e-12)
            @test rho1 * rho2 ≈ c12 rtol = 1.0e-12
        end

        # Continuity where the off-diagonal constraint takes over from the diagonal ones: at
        # c12^2 == c11*c22 the segment collapses to the diagonal cover (sqrt(c11), sqrt(c22)).
        for eps in (1.0e-2, 1.0e-4, 1.0e-8)
            rho1, rho2 = balanced_split(1.0, 10.0 * (1 + eps), 100.0, 1.0, 1.0)
            @test rho1 ≈ 1.0 rtol = 2eps
            @test rho2 ≈ 10.0 rtol = 2eps
        end

        # Continuity as a parent value vanishes: the equalizing target diverges, but the
        # segment endpoint it is clamped to does not.
        for z1 in (1.0e-3, 1.0e-6, 1.0e-12, 0.0)
            rho1, rho2 = balanced_split(1.0, 1.0, 1.0e-2, z1, 1.0)
            @test rho1 ≈ 10.0
            @test rho2 ≈ 0.1
        end

        # A value ratio extreme enough to overflow or underflow the equalizing target still
        # lands on the endpoint it was heading for, rather than reverting to the interior.
        for (z1, z2) in ((1.0, 1.0e300), (1.0e-300, 1.0e300), (1.0e-300, 1.0), (1.0, Inf))
            rho1, rho2 = balanced_split(1.0, 10.0, 1.0, z1, z2)
            @test rho1 ≈ 10.0
            @test rho2 ≈ 1.0
        end
        for (z1, z2) in ((1.0e300, 1.0), (1.0e300, 1.0e-300), (1.0, 1.0e-300), (Inf, 1.0))
            rho1, rho2 = balanced_split(1.0, 10.0, 1.0, z1, z2)
            @test rho1 ≈ 1.0
            @test rho2 ≈ 10.0
        end

        # Where the segment has no endpoint to fall back on, the equalizing point is the answer
        # and the exponents carry it: the product under the root overflows while the point itself
        # is an ordinary number.
        @test balanced_split(0.0, 1.0, 0.0, 1.0e-200, 1.0e200) === (1.0e200, 1.0e-200)
        @test balanced_split(0.0, 1.0, 0.0, 1.0e200, 1.0e-200) === (1.0e-200, 1.0e200)
        @test balanced_split(0.0f0, 1.0f0, 0.0f0, 1.0f-30, 1.0f30) === (1.0f30, 1.0f-30)

        # That path is covariant to the bit under a rescaling by powers of two, which moves
        # exponents and leaves every mantissa alone.
        for l1 in (2.0^-40, 1.0, 2.0^40), l2 in (2.0^-40, 1.0, 2.0^40)
            base = balanced_split(0.0, 1.0, 0.0, 1.0e-200, 1.0e200)
            r1, r2 = balanced_split(
                0.0, 1.0 / (l1 * l2), 0.0, l1 * 1.0e-200, l2 * 1.0e200
            )
            @test r1 * l1 === base[1]
            @test r2 * l2 === base[2]
        end

        # A ratio no exponent can represent is refused rather than covered by something else.
        @test_throws "outside the floating-point range" balanced_split(
            0.0, 1.0e300, 0.0, 1.0e-300, 1.0e300
        )
        # So is a segment with no endpoint and a pair with no ratio, which between them leave
        # nothing to place `rho1` by.
        @test_throws "outside the floating-point range" balanced_split(
            0.0, 3.0, 0.0, 0.0, 1.0
        )

        # Selecting the fallback with `ifelse` rather than a branch gives bit-identical results, so
        # a backend that cannot branch on values has an equivalent form. The grid includes zero in
        # every argument, which covers the three families where the result is discontinuous.
        function split_select(c11, c12, c22, z1, z2)
            lo = sqrt(c11)
            hi = c12 / sqrt(c22)
            rho1 = clamp(sqrt(c12 * z2 / z1), lo, hi)
            fallback = clamp(DerivativeCovers.split_fallback(lo, hi, c12, z1, z2), lo, hi)
            rho1 = ifelse(isfinite(rho1) & (rho1 > 0), rho1, fallback)
            return rho1, c12 / rho1
        end
        same(x, y) = x === y || (isnan(x) && isnan(y))
        vals = (0.0, 1.0e-8, 1.0e-3, 0.5, 1.0, 2.0, 7.3, 1.0e3, 1.0e8)
        differing = Tuple{NTuple{5, Float64}, NTuple{2, Float64}, NTuple{2, Float64}}[]
        ncases = 0
        for c11 in vals, c22 in vals, c12 in vals, z1 in vals, z2 in vals
            (c12 > 0 && c12^2 > c11 * c22) || continue
            determined(c11, c22, z1, z2) || continue
            ncases += 1
            branched = balanced_split(c11, c12, c22, z1, z2)
            selected = split_select(c11, c12, c22, z1, z2)
            all(same.(branched, selected)) ||
                push!(differing, ((c11, c12, c22, z1, z2), branched, selected))
        end
        @test ncases > 25_000
        @test isempty(differing)
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
        @test !all(isfinite, cover_hessian(WithDiagnostics(), divdiff, [0.5, 0.5]; check = false).h_sum)
        @test !all(
            isfinite,
            cover_hessian(
                WithDiagnostics(), recipdot, [1.0, 2.0, 2.0, -1.0];
                check = false
            ).h_sum
        )

        # Away from the singularity both programs cover normally.
        @test all(isfinite, cover_hessian(WithDiagnostics(), divdiff, [0.5, 1.5]).h_sum)
        @test all(isfinite, cover_hessian(WithDiagnostics(), recipdot, [1.0, 2.0, 2.0, 1.0]).h_sum)
    end

    @testset "the split" begin
        # The degeneracy threshold resolves against the tape's element type, and the quantity it
        # bounds is representability: `eps(T)` is the floor below which one parent value is
        # indistinguishable from zero at the scale of the other.
        @test DerivativeCovers.split_ratio(nothing, Float64) === sqrt(eps(Float64))
        @test DerivativeCovers.split_ratio(nothing, Float32) === sqrt(eps(Float32))
        @test DerivativeCovers.split_ratio(1.0e-8, Float64) === 1.0e-8
        @test_throws "must lie in [0, 1)" DerivativeCovers.split_ratio(1.0, Float64)
        @test_throws "must lie in [0, 1)" DerivativeCovers.split_ratio(-1.0e-8, Float64)
        @test_throws "must lie in [0, 1)" DerivativeCovers.split_ratio(NaN, Float64)

        # A threshold stored as a constant is silently wrong at another precision. `1e-8` lies
        # *below* `eps(Float32)`, so at single precision it collapses the guard to an
        # exactly-vanishing test: this product's operands are in a ratio of `1e-5`, degenerate
        # against `sqrt(eps(Float32))` and ordinary against `1e-8`, and the two give different
        # covers. Nothing in a `Float64` battery can catch that.
        fdeg = x -> x[1] * x[2]
        x32 = Float32[1.0e-5, 1.0]
        @test cover_hessian(fdeg, x32).h_sum == Float32[1.0, 1.0]
        @test cover_hessian(fdeg, x32; ratio = 1.0e-8).h_sum ≈ Float32[sqrt(1.0f5), sqrt(1.0f-5)]
        @test cover_hessian(fdeg, Float64.(x32)).h_sum ≈ [sqrt(1.0e5), sqrt(1.0e-5)]

        # At a zero-zero product no value ratio exists, so the declared scales replace it and the
        # cover stops tracking a quantity that carries none. Perturbing the vanishing operand does
        # not move it.
        for z1 in (1.0e-20, 3.0e-20)
            xdeg = [z1, 1.0]
            res = cover_hessian(WithDiagnostics(), fdeg, xdeg)
            @test res.h_sum == [1.0, 1.0]
            @test cover_ratio(ForwardDiff.hessian(fdeg, xdeg), res.h_sum) <= GATE
        end
        # Reading the values there instead sends the cover to `1e10` and moves it by `sqrt(3)`
        # between the two points above.
        @test cover_hessian(fdeg, [1.0e-20, 1.0]; ratio = 0.0).h_sum ≈ [1.0e10, 1.0e-10]

        # The guard belongs to a block, not to a program: it fires at the zero-zero product and
        # not at a perturbed point whose values are tiny but in a ratio of 4, which is a scale
        # reference like any other. Dyadic perturbations make that ratio exact.
        fmul = x -> (x[1] - 1) * (x[2] - 1)
        x0 = [1.0, 1.0]
        xp = [1.0 + 2.0^-40, 1.0 + 2.0^-38]
        @test cover_hessian(WithDiagnostics(), fmul, x0).h_sum == [1.0, 1.0]
        @test cover_hessian(WithDiagnostics(), fmul, xp).h_sum == [2.0, 0.5]
        # Widening the threshold past that ratio condemns the pair and hands it to the scales.
        @test cover_hessian(WithDiagnostics(), fmul, xp; ratio = 0.5).h_sum == [1.0, 1.0]

        # A bounded feasible segment keeps the value channel however lopsided the pair, because
        # the clamp already fixes the split and both endpoints are covariant. Only a degenerate
        # pair meeting a degenerate endpoint hands over.
        @test DerivativeCovers.value_bounds_split(1.0, 10.0, 1.0, 1.0e-20, 1.0, 1.0e-8)
        @test DerivativeCovers.value_bounds_split(1.0, 10.0, 1.0, 1.0, 1.0e-20, 1.0e-8)
        @test !DerivativeCovers.value_bounds_split(0.0, 10.0, 0.0, 1.0e-20, 1.0, 1.0e-8)
        @test !DerivativeCovers.value_bounds_split(1.0, 10.0, 0.0, 1.0e-20, 1.0, 1.0e-8)
        @test DerivativeCovers.value_bounds_split(1.0, 10.0, 0.0, 1.0, 1.0e-20, 1.0e-8)
        @test !DerivativeCovers.value_bounds_split(0.0, 10.0, 0.0, 0.0, 0.0, 1.0e-8)
        # `ratio = 0` narrows the guard to the exactly vanishing pair, which is the only degeneracy
        # test that survives a rescaling of the intermediates unchanged — and the only one that
        # does not survive a perturbation of the evaluation point, which is why it is not the
        # default.
        @test DerivativeCovers.value_bounds_split(0.0, 10.0, 0.0, 1.0e-20, 1.0, 0.0)
        @test !DerivativeCovers.value_bounds_split(0.0, 10.0, 0.0, 0.0, 1.0, 0.0)

        # Scale references whose ratio no product of them can hold: the split is theirs all the
        # same, and the cover is the exact one for `H = [0 1; 1 0]`.
        fwide = x -> (1.0e-200 * x[1]) * (1.0e200 * x[2])
        xwide = [0.0, 0.0]
        Hwide = [0.0 1.0; 1.0 0.0]
        @test cover_hessian(WithDiagnostics(), fwide, xwide).h_sum == [1.0, 1.0]
        @test cover_ratio(Hwide, cover_hessian(WithDiagnostics(), fwide, xwide).h_quad) <= GATE
        @test forward_cover(fwide, xwide).h_quad == [1.0, 1.0]

        # `guards_any_block` predicts which case a tape is in without running a cover. It may
        # overcount, since a block whose adjoint vanishes contributes no cover term, so it is
        # checked in the direction it claims: a tape it clears takes the value channel at every
        # block, which is what makes the cover exactly covariant under an input rescaling with the
        # declared scales held fixed.
        d5 = [1.0e-6, 1.0e3, 1.0, 1.0e-2, 1.0e4]
        w5 = [0.5, 4.0, 1.0, 2.0, 0.25]
        @testset "off the guard the scales are inert — $name" for (name, f, x0) in PROBLEMS
            t, _ = tape_function(f, x0)
            if !DerivativeCovers.guards_any_block(t)
                res = cover_hessian(WithDiagnostics(), f, x0)
                resw = cover_hessian(WithDiagnostics(), f, x0; scales = fill(3.0, length(x0)))
                @test resw.h_sum == res.h_sum
                @test resw.h_quad == res.h_quad
            end
        end
        @test !DerivativeCovers.guards_any_block(first(tape_function(quad_matvec(AQ), XQ)))
        resq = cover_hessian(WithDiagnostics(), quad_matvec(AQ), XQ; scales = w5)
        resq2 = cover_hessian(
            WithDiagnostics(), xt -> quad_matvec(AQ)(xt ./ d5), d5 .* XQ; scales = w5
        )
        @test resq2.h_sum ≈ resq.h_sum ./ d5 rtol = 1.0e-12
        @test resq2.h_quad ≈ resq.h_quad ./ d5 rtol = 1.0e-12

        # Where a block does reach the fallback, covariance needs the declared scales rescaled
        # alongside the inputs: they are the units the cover borrows there.
        @test DerivativeCovers.guards_any_block(first(tape_function(fmul, x0)))
        dm = [4.0, 0.25]
        resm = cover_hessian(WithDiagnostics(), fmul, x0; scales = [2.0, 3.0])
        resm2 = cover_hessian(
            WithDiagnostics(), xt -> fmul(xt ./ dm), dm .* x0; scales = dm .* [2.0, 3.0]
        )
        @test resm2.h_sum ≈ resm.h_sum ./ dm rtol = 1.0e-12

        # The declared scales are built by the first block that asks and not before: a tape whose
        # blocks all keep a usable value pair leaves the store empty after a whole cover sweep,
        # which is the cost claim stated as a property rather than as a timing.
        let fplain = x -> exp(x[1] * x[2]) + x[1]^3 + sin(x[2])
            t, outidx = tape_function(fplain, [0.7, -1.3])
            # The references are indexed by node, and a sweep gathers them into slot order through
            # a parent lookup, so spanning the tape linearly is what keeps that lookup in range.
            let sc = DerivativeCovers.split_scales(nothing, t)
                @test size(sc) == (nnodes(t),)
                @test IndexStyle(typeof(sc)) === IndexLinear()
                @test sc[nnodes(t)].value == abs(t.vals[nnodes(t)])
            end
            @test isempty(DerivativeCovers.split_scales(nothing, t).store.sigmas)
            DerivativeCovers.local_covers(t, adjoints(t, outidx))
            @test isempty(DerivativeCovers.split_scales(nothing, t).store.sigmas)
            # A guarded block fills it, and refills it for no one.
            tg, _ = tape_function(fmul, x0)
            sg = DerivativeCovers.split_scales(nothing, tg)
            @test isempty(sg.store.sigmas)
            @test DerivativeCovers.declared_scale(sg[1]) == 1
            @test length(sg.store.sigmas) == DerivativeCovers.nnodes(tg)
        end

        # A zero *value* certifies nothing about a slot, so none is dropped on that account. A zero
        # *tangent* does: the slot multiplies its block's entries by the zero vector, so they are
        # absent from the Hessian and the cover leaves them uncovered. Here the two coincide —
        # `x[1]^2` has both — and the cover is the exact one, `H = [6 0; 0 0]` admitting no cover
        # smaller than `[sqrt(6), 0]`.
        fdead = x -> x[1]^2 * x[2]
        xd = [0.0, 3.0]
        Hd = ForwardDiff.hessian(fdead, xd)
        resdead = cover_hessian(WithDiagnostics(), fdead, xd)
        @test cover_ratio(Hd, resdead.h_sum) <= GATE
        @test resdead.h_sum ≈ [sqrt(6.0), 0.0]
        @test resdead.h_quad ≈ [sqrt(6.0), 0.0]
        @test cover_hybrid(fdead, xd, 1) ≈ [sqrt(6.0), 0.0]
        @test forward_cover(fdead, xd).h_quad ≈ [sqrt(6.0), 0.0]

        # A slot the value channel sees no problem with: `cos` is `1` at `0`, so the block keeps a
        # usable value pair and only the tangent reports the slot dead.
        fung = x -> cos(x[1]) * x[2]
        xu = [0.0, 3.0]
        resung = cover_hessian(WithDiagnostics(), fung, xu)
        @test cover_ratio(ForwardDiff.hessian(fung, xu), resung.h_sum) <= GATE
        @test resung.h_sum ≈ [sqrt(3.0), 0.0]

        # The converse: a vanishing value with a live tangent keeps its weight, since the block's
        # off-diagonal entry is in the Hessian and has to be covered.
        xlivecos = [pi / 2, 3.0]
        reslive = cover_hessian(WithDiagnostics(), fung, xlivecos)
        @test cover_ratio(ForwardDiff.hessian(fung, xlivecos), reslive.h_sum) <= GATE
        @test all(>(0), reslive.h_sum)

        # A term switched off by a zero coefficient contributes nothing to the Hessian, so it
        # contributes nothing to the cover either: what is left is the cover of the live term.
        fbeta = b -> (x -> (b * x[1]^2) * (x[2] + x[2]^2)^2 + x[1] * x[2])
        xb = [1.5, 2.0]
        resbeta = cover_hessian(WithDiagnostics(), fbeta(0.0), xb)
        @test cover_ratio(ForwardDiff.hessian(fbeta(0.0), xb), resbeta.h_sum) <= GATE
        @test resbeta.h_sum ≈ cover_hessian(WithDiagnostics(), x -> x[1] * x[2], xb).h_sum
        @test resbeta.h_quad ≈ cover_hessian(WithDiagnostics(), x -> x[1] * x[2], xb).h_quad

        # Dropping a dead slot's entries is a covariant operation: `x -> D*x` puts `1/d[a]` on each
        # input edge and leaves every value and adjoint alone, so `h -> h ./ d` exactly.
        ddead = [4.0, 0.25]
        @testset "dead slots stay covariant — $i" for (i, (fz, xz)) in
            enumerate(((fdead, xd), (fung, xu), (fbeta(0.0), xb)))
            rz = cover_hessian(WithDiagnostics(), fz, xz)
            rzd = cover_hessian(WithDiagnostics(), xt -> fz(xt ./ ddead), ddead .* xz)
            @test rzd.h_sum ≈ rz.h_sum ./ ddead rtol = 1.0e-12
            @test rzd.h_quad ≈ rz.h_quad ./ ddead rtol = 1.0e-12
        end

        fdot = x -> dot([x[1]^2, x[2]], [x[3], x[4]])
        xdot = [0.0, 1.0, 2.0, 3.0]
        resdot = cover_hessian(WithDiagnostics(), fdot, xdot)
        @test cover_ratio(ForwardDiff.hessian(fdot, xdot), resdot.h_sum) <= GATE
        # The `(x[1]^2, x[3])` pair is dead at `xdot`, so `x[3]` carries no weight at all and the
        # remaining pair is covered exactly.
        @test resdot.h_sum[3] == 0
        @test resdot.h_sum[2] * resdot.h_sum[4] ≈ 1.0

        # An inner product puts the same question to each of its pairs, and one degenerate pair
        # decides the node: the `(x[1]^2, x[3])` pair vanishes at `xdot`, no pair does at `xlive`,
        # and a pair further along the product is read as readily as the first.
        xlive = [1.0, 1.0, 2.0, 3.0]
        xtail = [1.0, 1.0, 2.0, 0.0]
        @test DerivativeCovers.guards_any_block(first(tape_function(fdot, xdot)))
        @test !DerivativeCovers.guards_any_block(first(tape_function(fdot, xlive)))
        @test DerivativeCovers.guards_any_block(first(tape_function(fdot, xtail)))
        # The prediction is measured against the sweep it predicts, in the direction it claims: a
        # tape it clears never fills the declared-scale store. A tape it names may leave the store
        # empty all the same, since a guarded pair whose slot has no tangent is dropped outright
        # and asks for no scale — which is `xdot`, where `x[1]^2` is the vanishing side.
        stored = Dict{Vector{Float64}, Bool}()
        for xf in (xdot, xlive, xtail)
            t, outidx = tape_function(fdot, xf)
            sc = DerivativeCovers.split_scales(nothing, t)
            r = DerivativeCovers.split_ratio(nothing, eltype(t))
            vbar = adjoints(t, outidx)
            for k in (t.ninputs + 1):nnodes(t)
                DerivativeCovers.local_cover_slots(r, t, k, abs(vbar[k]), sc)
            end
            stored[xf] = !isempty(sc.store.sigmas)
            @test DerivativeCovers.guards_any_block(t) || isempty(sc.store.sigmas)
        end
        @test stored == Dict(xdot => false, xlive => false, xtail => true)

        # Declared scales must be strictly positive and finite: a zero entry would blank a live
        # slot's scale where the guard falls back on them. Both keywords are validated at the
        # entry point, before any sweep runs.
        @test_throws "strictly positive" cover_hessian(fmul, x0; scales = [1.0, 0.0])
        @test_throws "strictly positive" cover_hessian(fmul, x0; scales = [1.0, Inf])
        @test_throws "strictly positive" cover_hessian(fmul, x0; scales = [1.0, NaN])
        @test_throws DimensionMismatch cover_hessian(fmul, x0; scales = [1.0])
        @test_throws "must lie in [0, 1)" cover_hessian(fmul, x0; ratio = 1.0)

        # Every entry point that splits threads both keywords. `ratio = 0.5` condemns `xp`'s value
        # pair, which the default keeps, so each of these separates.
        vc = cover_variance(fmul, xp; ratio = 0.5)
        @test vc.h_sum == [1.0, 1.0]
        @test all(>=(1 - 1.0e-12), vc.severity)
        @test cover_variance(fmul, xp).h_sum == [2.0, 0.5]

        Fsub = x -> [(x[1] - 1) * (x[2] - 1)]
        @test declare_submap(Fsub, xp; ratio = 0.5).S == [1.0 1.0]
        @test declare_submap(Fsub, xp).S == [2.0 0.5]

        # The NLS loss reaches the same splits through its hybrid merge, so `h` follows. The
        # residual channels carry unit atoms, which admit no split, and the gradient is an ordinary
        # adjoint sweep, so neither moves.
        nv = cover_nls(Fsub, xp)
        ns = cover_nls(Fsub, xp; ratio = 0.5)
        @test nv.h != ns.h
        @test nv.h_gn == ns.h_gn
        @test nv.grad == ns.grad
        let H = ForwardDiff.hessian(x -> 0.5 * sum(abs2, Fsub(x)), xp)
            @test cover_ratio(H, nv.h) <= GATE
            @test cover_ratio(H, ns.h) <= GATE
        end

        # A forward propagation carries both scale references itself, so it splits by the same
        # rule a tape does and reaches the lattice's forward corner under either threshold.
        for ratio in (nothing, 0.5)
            fc = forward_cover(fmul, xp; ratio)
            lat = cover_lattice(fmul, xp; ratio)
            @test fc.h_quad ≈ lat.h_quad_forward rtol = 1.0e-12
        end
        @test_throws "strictly positive" forward_cover(fmul, x0; scales = [1.0, 0.0])
        @test_throws DimensionMismatch forward_cover(fmul, x0; scales = [1.0])

        # The covering chain holds on the whole battery under a threshold and scales that are not
        # the defaults, since neither may affect validity.
        @testset "covering chain off the defaults — $name" for (name, f, x0) in PROBLEMS
            H = ForwardDiff.hessian(f, x0)
            res = cover_hessian(
                WithDiagnostics(), f, x0; ratio = 1.0e-4, scales = fill(2.0, length(x0))
            )
            for h in (res.h_sum, res.h_quad)
                @test all(>=(0), h)
                @test rank1_ratio(res.B, h) <= GATE
                @test cover_ratio(H, h) <= GATE
            end
            @test res.h_sum_rev ≈ res.h_sum rtol = 1.0e-12
            # The one-group hybrid reproduces the sum merge whatever the split.
            @test cover_hybrid(f, x0, 1; ratio = 1.0e-4, scales = fill(2.0, length(x0))) ≈
                res.h_sum rtol = 1.0e-12
        end
    end

    @testset "node levels and partitions" begin
        f = x -> exp(x[1] * x[2]) + x[2]
        t, outidx = tape_function(f, [1.0, 2.0])
        levels = node_levels(t)
        @test all(iszero, levels[1:t.ninputs])
        @test all(k -> levels[k] > maximum(levels[t.parents[k]]; init = -1), (t.ninputs + 1):nnodes(t))

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
        @test sort(cyclic_partition(t, 3)) == sort(repeat(1:3, outer = N)[1:N])
        @test_throws ArgumentError cyclic_partition(t, 0)
        @test_throws ArgumentError cover_hybrid(t, outidx, 0)

        # The product and the exponential each carry one atom; the addition and the inputs none.
        @test natoms(t) == 2
        # A program of linear nodes carries none, whatever their arity.
        tl, _ = tape_function(x -> 3 * x[1] - x[2] / 2, [1.0, 2.0])
        @test natoms(tl) == 0
        # An inner-product node carries one atom per parent pair.
        td, _ = tape_function(z -> dot(z[1:3], z[4:6]), [0.3, -1.1, 0.7, 1.2, -0.4, 0.9])
        @test natoms(td) == 3
    end

    @testset "hybrid merge — $name" for (name, f, x0) in PROBLEMS
        H = ForwardDiff.hessian(f, x0)
        t, outidx = tape_function(f, x0)
        N = nnodes(t)
        res = cover_hessian(WithDiagnostics(), t, outidx)

        # One group sums every atom: the sum merge. One atom per group: the quadrature merge.
        @test cover_hybrid(t, outidx, 1) ≈ res.h_sum rtol = 1.0e-12
        @test cover_hybrid(t, outidx, natoms(t)) ≈ res.h_quad rtol = 1.0e-12
        @test cover_hybrid(t, outidx, 10 * natoms(t) + 1) ≈ res.h_quad rtol = 1.0e-12
        @test cover_hybrid(t, outidx, collect(1:N)) ≈ res.h_quad rtol = 1.0e-12

        # Dealing over 2G groups refines dealing over G, so it can only tighten.
        for G in (1, 2, 3, 7)
            @test all(cover_hybrid(t, outidx, 2G) .<= cover_hybrid(t, outidx, G) .* GATE)
        end

        rng = MersenneTwister(hash(name))
        for groups in (
                cyclic_partition(t, 4), level_partition(t, 4),
                rand(rng, 1:5, N), rand(rng, 1:2, N),
            )
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
        cases = [
            ("rosenbrock n=2", rosenbrock, [-1.2, 1.0], [1.0e-6, 1.0e3]),
            (
                "logsumexp n=8", logsumexp,
                [-3.0, -1.0, -2.0, 0.0, 1.0, 2.0, -0.5, 3.0],
                [1.0e-3, 1.0e2, 1.0, 1.0e-1, 1.0e4, 1.0e-2, 1.0e1, 1.0e3],
            ),
        ]
        @testset "$name" for (name, f, x0, d) in cases
            e = ones(length(x0))
            for G in (2, 4)
                href = cover_hybrid(xt -> f(xt ./ e), x0, G)
                hd = cover_hybrid(xt -> f(xt ./ d), d .* x0, G)
                @test hd ≈ href ./ d rtol = 1.0e-12

                c = 7.3
                hc = cover_hybrid(x -> c * f(x), x0, G)
                h1 = cover_hybrid(x -> 1.0 * f(x), x0, G)
                @test hc ≈ sqrt(c) .* h1 rtol = 1.0e-12
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
        @test vc.h_sum ≈ res.h_sum rtol = 1.0e-12             # the hard channel is the sum merge
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
        @test vc.h_var ≈ ref rtol = 1.0e-12
    end

    @testset "variance channel is exact on tree-structured graphs" begin
        # Each input reaches each node along a single path, and no node has two parents sharing an
        # input, so there is no covariance to drop: the estimate attains the quadrature merge.
        for (f, x0) in ((tree_program, TREE_X), (quad_factored(WQ), XQ))
            @test cover_variance(f, x0).h_var ≈
                cover_hessian(WithDiagnostics(), f, x0).h_quad rtol = 1.0e-12
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
        cases = [
            ("rosenbrock n=2", rosenbrock, [-1.2, 1.0], [1.0e-6, 1.0e3]),
            ("quad matvec", quad_matvec(AQ), XQ, [1.0e-6, 1.0e3, 1.0, 1.0e-2, 1.0e4]),
            (
                "logsumexp n=8", logsumexp,
                [-3.0, -1.0, -2.0, 0.0, 1.0, 2.0, -0.5, 3.0],
                [1.0e-3, 1.0e2, 1.0, 1.0e-1, 1.0e4, 1.0e-2, 1.0e1, 1.0e3],
            ),
        ]
        @testset "$name" for (name, f, x0, d) in cases
            vc = cover_variance(f, x0)

            # Rescaling the inputs puts a factor 1/d[a] on each input edge, whose square the
            # variance channel picks up; the severity, a ratio of the two channels, is invariant.
            vcd = cover_variance(xt -> f(xt ./ d), d .* x0)
            @test vcd.h_var ≈ vc.h_var ./ d rtol = 1.0e-12
            @test vcd.severity ≈ vc.severity rtol = 1.0e-12

            # f -> c*f scales every rho by sqrt(c), hence every squared seed by c.
            c = 7.3
            vcc = cover_variance(x -> c * f(x), x0)
            @test vcc.h_var ≈ sqrt(c) .* vc.h_var rtol = 1.0e-12
            @test vcc.severity ≈ vc.severity rtol = 1.0e-12
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
        @test hvp(f, x0, v) ≈ H * v rtol = 1.0e-10
        @test hvp(f, x0, zero(x0)) ≈ zero(x0) atol = 1.0e-14
        @test_throws DimensionMismatch hvp(f, x0, [1.0; v])

        # The in-place form writes the caller's destination and returns it.
        t, outidx = tape_function(f, x0)
        Hv = fill(NaN, length(x0))
        @test hvp!(Hv, t, outidx, v) === Hv
        @test Hv == hvp(t, outidx, v)
        @test_throws DimensionMismatch hvp!(similar(x0, length(x0) + 1), t, outidx, v)
    end

    @testset "spectral shrink — $name" for (name, f, x0) in PROBLEMS
        H = ForwardDiff.hessian(f, x0)
        t, outidx = tape_function(f, x0)
        h = cover_hessian(WithDiagnostics(), t, outidx).h_sum

        # Every entry of the scaled Hessian is bounded by its spectral norm.
        lambda = scaled_spectral_radius(H, h)
        @test lambda >= cover_ratio(H, h) * (1 - 1.0e-12)

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
            est = scaled_spectral_radius(t, outidx, h; maxiter = 1000, rtol = 1.0e-10)
            @test est <= lambda * GATE
            @test est ≈ lambda rtol = 1.0e-4
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
        @test spectral_shrink(h, 0.25; safety = 4) == h        # safety = 4 saturates the cover

        # The scaled Hessian, and hence the shrink factor, is invariant under both rescalings.
        d = [1.0e-3, 1.0e5]
        @test scaled_spectral_radius(H .* (d * d'), h .* d) ≈ scaled_spectral_radius(H, h)
        @test scaled_spectral_radius(9 .* H, 3 .* h) ≈ scaled_spectral_radius(H, h)

        # A coordinate the Hessian ignores may carry a zero cover; one it does not, may not.
        @test scaled_spectral_radius([1.0 0.0; 0.0 0.0], [1.0, 0.0]) ≈ 1
        @test scaled_spectral_radius(zeros(2, 2), [0.0, 0.0]) == 0
        @test_throws "does not cover" scaled_spectral_radius(H, [2.0, 0.0])

        t, outidx = tape_function(rosenbrock, [-1.2, 1.0])
        @test_throws "strictly positive" scaled_spectral_radius(t, outidx, [1.0, 0.0])
        @test_throws "did not converge" scaled_spectral_radius(
            t, outidx, [1.0, 1.0];
            maxiter = 1, rtol = 1.0e-12
        )
        @test_throws ArgumentError spectral_shrink(h, -1.0)
        @test_throws ArgumentError spectral_shrink(h, 1.0; safety = 0)
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
    GENERATED = [
        (sigma, g(n))
            for sigma in (0.0, 2.0)
            for (_, g) in families(; sigma, seed = 7)
            for n in (3, 6)
    ]

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
        ps = [quadratic(n; form = fm, rng = MersenneTwister(3)) for fm in (:entrywise, :matvec, :factored)]
        A = exact_hessian(ps[1])
        @test issymmetric(A)
        @test all(p -> exact_hessian(p) ≈ A, ps)          # one problem, three programs
        @test all(p -> p.x0 ≈ ps[1].x0, ps)
        # The factored form's rank-many nonlinear nodes do not overlap, so its cover is far
        # tighter than the entrywise form's n² overlapping ones.
        hs = [cover_hessian(WithDiagnostics(), p.f, p.x0).h_sum for p in ps]
        @test all(hs[3] .< hs[1])

        # rank 1: H = w*w' and the minimal cover abs.(w) = sqrt.(diag(A)) is attained.
        p1 = quadratic(6; rank = 1, form = :factored, rng = MersenneTwister(4))
        @test rank(exact_hessian(p1)) == 1
        @test cover_hessian(WithDiagnostics(), p1.f, p1.x0).h_quad ≈ sqrt.(diag(exact_hessian(p1)))
    end

    @testset "random_dag sharing controls reconvergence" begin
        # sharing = 0 consumes each node exactly once, so the graph is a tree: every input reaches
        # every node along a single path. The variance channel drops covariances between
        # reconverging paths, so it attains the quadrature merge exactly iff there are none.
        tree = random_dag(8; nops = 40, sharing = 0.0, rng = MersenneTwister(11))
        @test cover_variance(tree.f, tree.x0).h_var ≈
            cover_hessian(WithDiagnostics(), tree.f, tree.x0).h_quad rtol = 1.0e-10

        # With reuse, paths reconverge; the variance channel drops exactly those covariances and
        # so falls strictly below the quadrature merge on the coordinates that reconverge.
        shared = random_dag(8; nops = 40, sharing = 0.6, rng = MersenneTwister(11))
        hv = cover_variance(shared.f, shared.x0).h_var
        hq = cover_hessian(WithDiagnostics(), shared.f, shared.x0).h_quad
        @test all(hv .<= hq .* GATE)
        @test count(hv .< hq .* (1 - 1.0e-8)) >= 2

        # Each input reaches the output: nothing is dropped on the floor.
        @test all(!iszero, ForwardDiff.gradient(shared.f, shared.x0))

        # Values stay bounded no matter where the recipe is replayed, so `*` and `^2` chains
        # cannot overflow.
        @test isfinite(shared.f(100 .* shared.x0))
    end

    @testset "random_dag keeps every derivative alive at scale" begin
        # Large random DAGs expose tanh saturation and multiplicative underflow that small cases
        # miss. The generator must keep every nonlinear derivative active.
        function tanhderivs(r, x0)
            v = Vector{Float64}(undef, r.ninputs + length(r.ops))
            v[1:r.ninputs] = x0
            ds = Float64[]
            for (k, (op, ps, cs)) in enumerate(r.ops)
                v[r.ninputs + k] = Problems._apply(op, v, ps, cs)
                op === :tanh && push!(ds, 1 - tanh(v[ps[1]])^2)
            end
            return ds
        end
        for sharing in (0.0, 0.6), n in (16, 32, 64, 128, 256, 512)
            p = random_dag(n; sharing, rng = MersenneTwister(1))
            @test all(>(1.0e-3), tanhderivs(p.f, p.x0))
            g = ForwardDiff.gradient(p.f, p.x0)
            H = ForwardDiff.hessian(p.f, p.x0)
            @test any(!iszero, H)
            # A zero gradient is acceptable only when the same input is also dead at second order.
            dead = findall(iszero, g)
            @test all(i -> all(iszero, view(H, i, :)), dead)
            @test length(dead) <= max(1, n ÷ 8)
        end
    end

    @testset "rescale is exactly the input-units transformation" begin
        p = quadratic(4; rng = MersenneTwister(5))
        d = logscales(MersenneTwister(6), 4; sigma = 4.0)
        @test maximum(d) / minimum(d) > 1.0e2                # a genuinely wide spread
        q = rescale(p, d)

        @test q.x0 ≈ d .* p.x0
        @test q.f(q.x0) ≈ p.f(p.x0)
        @test exact_hessian(q) ≈ ForwardDiff.hessian(q.f, q.x0)
        @test exact_hessian(q) ≈ Diagonal(inv.(d)) * exact_hessian(p) * Diagonal(inv.(d))
        # The cover is covariant under it: this is the property the whole method rests on.
        @test cover_hessian(WithDiagnostics(), q.f, q.x0).h_sum ≈
            cover_hessian(WithDiagnostics(), p.f, p.x0).h_sum ./ d rtol = 1.0e-10

        @test_throws DimensionMismatch rescale(p, [1.0, 2.0])
        @test_throws ArgumentError rescale(p, [1.0, 2.0, 0.0, 1.0])
    end

    @testset "sigma widens the coefficient dynamic range" begin
        # Measure spread in log magnitude, matching the definition of sigma and avoiding
        # sensitivity to one near-zero coefficient.
        function logspread(A)
            l = log.(abs.(vec(A)))
            m = sum(l) / length(l)
            return sqrt(sum(abs2, l .- m) / length(l))
        end
        narrow = exact_hessian(quadratic(32; rng = MersenneTwister(2), sigma = 0.0))
        wide = exact_hessian(quadratic(32; rng = MersenneTwister(2), sigma = 3.0))
        @test logspread(wide) > 1.5 * logspread(narrow)

        @test_throws ArgumentError quadratic(4; rank = 5)
        @test_throws ArgumentError quadratic(4; form = :bogus)
        @test_throws ArgumentError deep_composition(4; width = 0)
        @test_throws ArgumentError random_dag(4; sharing = 1.5)
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
        @test anisotropy(f) ≈ 0 atol = 1.0e-12

        f = inflation_factors(SpikeInflation(3), n, peak)
        @test f[3] ≈ peak
        @test f[setdiff(1:n, 3)] ≈ ones(n - 1)
        @test anisotropy(f) ≈ log(peak) * (1 - 1 / n)
        @test_throws BoundsError inflation_factors(SpikeInflation(n + 1), n, peak)

        f = inflation_factors(RampInflation(), n, peak)
        @test issorted(f)
        @test f[1] ≈ 1
        @test f[end] ≈ peak
        @test anisotropy(f) ≈ log(peak) / 2 atol = 0.05
        @test inflation_factors(RampInflation(), 1, peak) ≈ [peak]

        f = inflation_factors(LognormalInflation(), n, peak; rng = MersenneTwister(3))
        @test f ≈ inflation_factors(LognormalInflation(), n, peak; rng = MersenneTwister(3))
        @test f != inflation_factors(LognormalInflation(), n, peak; rng = MersenneTwister(4))
        # Its head is more concentrated than a ramp's: the bulk of a normal draw sits well below
        # its maximum, and the gap widens with n.
        big = 256
        lgn = inflation_factors(LognormalInflation(), big, peak; rng = MersenneTwister(3))
        @test anisotropy(lgn) > anisotropy(inflation_factors(RampInflation(), big, peak))
    end

    @testset "an inflated cover is still a cover" begin
        p = quadratic(6; form = :entrywise, rng = MersenneTwister(7))
        H = exact_hessian(p)
        h = cover_hessian(WithDiagnostics(), p.f, p.x0).h_sum
        @test cover_ratio(H, h) <= 1

        for pat in (UniformInflation(), LognormalInflation(), SpikeInflation(2), RampInflation())
            f = inflation_factors(pat, h, 3.0; rng = MersenneTwister(8))
            hd = inflate(h, f)
            # Inflation can only weaken the bound, never break it...
            @test cover_ratio(H, hd) <= cover_ratio(H, h)
            @test cover_ratio(H, hd) <= 1
            # ...and the looseness it adds is exactly the factors that were asked for.
            @test hd ./ h ≈ f
            @test tightness(hd, h)[1] ≈ maximum(f)
        end

        @test inflate(UniformInflation(), h, 2.5) ≈ 2.5 .* h
        @test_throws ArgumentError inflate(h, fill(0.5, length(h)))
        @test_throws DimensionMismatch inflate(h, ones(length(h) + 1))
    end

    @testset "inflation leaves a zero cover coordinate at zero" begin
        # A cover of a Hessian that ignores an input has an exactly-zero coordinate there; no
        # multiplicative distortion can loosen a bound that is already exact.
        H = [2.0 1.0 0.0; 1.0 3.0 0.0; 0.0 0.0 0.0]
        h = [2.0, 2.0, 0.0]
        @test cover_ratio(H, h) <= 1

        f = inflation_factors(LognormalInflation(), h, 5.0; rng = MersenneTwister(9))
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
        D = Diagonal([1.0e3, 1.0e-4])
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
        dcovers = [inflate(UniformInflation(), h, 4.0) for h in covers]
        tol = 0.1
        # 2^-k <= 0.1 first at k = 4, and 2^-k <= 0.4 (the inflated test) first at k = 2.
        @test stopping_index(grads, covers, tol) == 5
        @test stopping_index(grads, dcovers, tol) == 3
        s = stopping_shift(grads, covers, dcovers, tol)
        @test s.k_ref == 5 && s.k_dist == 3
        @test s.shift == 2
        @test s.overshoot ≈ 0.25 / tol      # the true cover-norm where the run declared success
        @test stopping_shift(grads, covers, covers, tol).shift == 0
        @test stopping_index(grads, covers, 1.0e-9) === nothing
        @test_throws "never fires" stopping_shift(grads, covers, dcovers, 1.0e-9)
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
        hd = inflate(LognormalInflation(), h, 8.0; rng)
        b = regularization_bias(A, g, h, hd, 0.5)
        # Variables: x -> D*x sends g -> D*g, A -> D*A*D, and any cover h -> D*h.
        D = Diagonal([1.0e3, 1.0e-2, 1.0, 7.0, 1.0e-4])
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
            @test eigmin(Symmetric(A + tau * Diagonal(h .^ 2))) > -1.0e-10
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
        @test cover_cg(A, b, inflate(UniformInflation(), h, 5.0)).iters == 1
        # A spike leaves the scaled system with two distinct eigenvalues.
        spiked = cover_cg(A, b, inflate(SpikeInflation(1), h, 10.0))
        @test spiked.iters == 2
        @test spiked.x ≈ A \ b

        @test_throws "strictly positive" cover_cg(A, b, [0.0, 2.0, 3.0])

        # An indefinite A gives a direction of nonpositive curvature; without the guard the
        # division by it would fill x and the whole history with NaN.
        Aneg = Matrix(Diagonal([1.0, -4.0, 9.0]))
        @test_throws "nonpositive curvature" cover_cg(Aneg, b, h)
        @test_throws ArgumentError cover_cg(Aneg, b, h)
        # A singular direction is caught on the same test.
        @test_throws "nonpositive curvature" cover_cg(zeros(3, 3), b, h)
    end

    @testset "cover quality is step quality under a truncated CG" begin
        rng = MersenneTwister(3)
        M = randn(rng, 12, 12)
        A = M * M' + 12I                            # symmetric positive definite
        g = randn(rng, 12)
        h = sqrt.(diag(A))                          # a cover of a diagonally dominant A
        hd = inflate(LognormalInflation(), h, 1.0e4; rng)

        # Preconditioning moves the path, not the fixed point: given iterations enough, both runs
        # reach the target, and the loose cover pays for it in iterations.
        long = step_degradation(A, g, h, hd; maxiter = 40)
        @test long.error_ref < 1.0e-6 && long.error_dist < 1.0e-6
        @test long.iters_dist > long.iters_ref

        # On the usual budget of one iteration per coordinate the loose cover simply fails to
        # deliver a step: it does not reach the target, and its final iterate is orders of
        # magnitude further from exact than the reference run's. The gap between the two runs is
        # the property; where each lands in absolute terms depends on the draw.
        budget = step_degradation(A, g, h, hd)
        @test budget.iters_ref == long.iters_ref
        @test budget.iters_dist === nothing
        @test budget.error_dist > 1.0e3 * budget.error_ref

        # Uniform inflation is invisible here too: only the shape of the metric steers CG.
        uni = step_degradation(A, g, h, inflate(UniformInflation(), h, 50.0); maxiter = 4)
        @test uni.error_dist ≈ uni.error_ref
    end

end
