"""
    DerivativeCoverProblems

Parameterized families of scalar programs for studying cover quality.

The programs are written in generic Julia and depend on nothing here, so any operator-overloading
tool can run them; DerivativeCovers tapes them by virtue of `DerivativeCovers.TVar` overloading the
elementals they use.

Each generator returns a [`CoverProblem`](@ref): a program `f` written from those elementals, the
point `x0` to evaluate it at, an optional idiomatic-Julia `twin` that computes the same function by
an independent route, and an optional analytic `hessian`. The families differ in the property that
drives cover looseness — how much the paths from an input to the output reconverge — and each has
knobs for size, reconvergence, and dynamic range:

- [`quadratic`](@ref): `x'Ax/2` with `A = W*W'` of prescribed rank, written entrywise, as a
  matrix-vector product, or in factored form. The same `A` in three programs, so the three
  differ *only* in graph structure.
- [`deep_composition`](@ref): layers of `width` channels around a shared trunk, so each layer
  fans a single node out to `width` consumers that immediately reconverge.
- [`random_dag`](@ref): a random program whose `sharing` parameter interpolates from a tree
  (`sharing = 0`) to heavy subexpression reuse (`sharing = 1`).
- [`adjoint_cancellation`](@ref): a chain of multiplication nodes the output is nearly stationary
  in, so the paths from each node to the output cancel in the *adjoints* rather than the
  intermediate gradients.
  [`converged_inner_solve`](@ref) reaches the same condition by the envelope theorem.

Wide dynamic range enters along two independent axes: `sigma` widens the log-magnitude spread of
a family's coefficients, while [`rescale`](@ref) (with scales from [`logscales`](@ref)) spreads
the units of the *inputs*.
"""
module DerivativeCoverProblems

using LinearAlgebra: Diagonal, I, Symmetric, dot, eigvals, norm
using Random: AbstractRNG, MersenneTwister

export CoverProblem, nvars, exact_hessian, has_exact_hessian, has_twin, has_xstar
export quadratic, deep_composition, random_dag, mlp, lotka_volterra, cancellation, rescale, logscales, families
export adjoint_cancellation, converged_inner_solve
export JacobianProblem, noutputs, exact_jacobian, has_exact_jacobian, jacobian_cancellation

"""
    CoverProblem

A scalar program together with the references needed to judge a cover of its Hessian.

- `name`: a short identifier, unique within a family sweep.
- `f`: the program. Built only from the elementals overloaded for `DerivativeCovers.TVar`, so
  `f(x)` records a tape when `x` is a vector of `TVar`s, and evaluates normally on any other
  numeric vector (`Float64`, `ForwardDiff.Dual`, ...).
- `x0`: the point at which the Hessian is covered.
- `twin`: an idiomatic-Julia implementation of the same function, or `nothing`. It is written
  independently of `f`, by an independent route — array operations `f` cannot express (most
  families), or an explicit scalar loop when `f` is itself written in array notation (see
  [`mlp`](@ref)) — so agreement between the two is evidence that `f` is the function it claims to
  be.
- `hessian`: `x -> H`, the analytic Hessian, or `nothing` when no closed form is available. Use
  [`exact_hessian`](@ref) rather than the field.
- `xstar`: a global minimizer of `f`, or `nothing` when none is known. Problems built with a
  planted solution carry it.
"""
struct CoverProblem{F, G, H, X}
    name::String
    f::F
    x0::Vector{Float64}
    twin::G
    hessian::H
    xstar::X
end

CoverProblem(name::AbstractString, f, x0::AbstractVector; twin = nothing, hessian = nothing, xstar = nothing) =
    CoverProblem(String(name), f, Vector{Float64}(x0), twin, hessian,
                 xstar === nothing ? nothing : Vector{Float64}(xstar))

"""
    nvars(p::CoverProblem)

Number of inputs of `p`.
"""
nvars(p::CoverProblem) = length(p.x0)

"""
    has_exact_hessian(p::CoverProblem)

Whether `p` carries an analytic Hessian. When it does not, a reference Hessian must come from
elsewhere (`ForwardDiff.hessian(p.f, p.x0)`).
"""
has_exact_hessian(p::CoverProblem) = p.hessian !== nothing

"""
    has_twin(p::CoverProblem)

Whether `p` carries an independent reference implementation of its program.
"""
has_twin(p::CoverProblem) = p.twin !== nothing

"""
    has_xstar(p::CoverProblem)

Whether `p` carries a known global minimizer `p.xstar`.
"""
has_xstar(p::CoverProblem) = p.xstar !== nothing

"""
    exact_hessian(p::CoverProblem, x=p.x0) -> H

The analytic Hessian of `p` at `x`. Errors when `p` has none; test with
[`has_exact_hessian`](@ref) first.
"""
function exact_hessian(p::CoverProblem, x::AbstractVector = p.x0)
    has_exact_hessian(p) ||
        throw(ArgumentError("problem \"$(p.name)\" has no analytic Hessian; differentiate `p.f` instead"))
    return p.hessian(x)
end

# ---- program-building helpers ---------------------------------------------------------------
#
# Start accumulations from a term so generic recorded scalars need no detached additive identity.

"Sum the entries of `v` using only the binary `+` of its element type. `v` must be non-empty."
function _sum(v)
    isempty(v) && throw(ArgumentError("cannot sum an empty collection of tape variables"))
    s = v[begin]
    for i in (firstindex(v) + 1):lastindex(v)
        s = s + v[i]
    end
    return s
end

"The linear combination `sum_i c[i] * z[i]`, recorded one term at a time."
function _lincomb(c::AbstractVector{<:Real}, z)
    axes(c, 1) == axes(z, 1) ||
        throw(DimensionMismatch("coefficients and variables must match: $(axes(c)) vs $(axes(z))"))
    return _sum([c[i] * z[i] for i in eachindex(c, z)])
end

"The left-to-right chain `w[1]*z[1] + w[2]*z[2] + ... + w[n]*z[n]`, the scalar order a dot node's parents follow."
function _rowdot(w, z, n::Integer)
    acc = w[1] * z[1]
    for j in 2:n
        acc = acc + w[j] * z[j]
    end
    return acc
end

"""
Coefficients with magnitude `abs(randn()) * exp(sigma * randn())`: unit scale at `sigma = 0`,
and log-magnitudes spread by `sigma` beyond that.
"""
_coeffs(rng::AbstractRNG, dims::Integer...; sigma::Real) =
    randn(rng, dims...) .* exp.(sigma .* randn(rng, dims...))

"""
    logscales(rng, n; sigma) -> d

Positive scale factors with `log.(d)` normal of standard deviation `sigma`. Pass to
[`rescale`](@ref) to give a problem's inputs heterogeneous units.
"""
logscales(rng::AbstractRNG, n::Integer; sigma::Real) = exp.(sigma .* randn(rng, n))

# ---- expanded quadratics ---------------------------------------------------------------------

"""
    quadratic(n; rank=n, form=:entrywise, sigma=0.0, rng=MersenneTwister(1)) -> CoverProblem

The quadratic `f(x) = x'Ax/2` with `A = W*W'` positive semidefinite of the given `rank`, where
`W` is `n × rank` and `sigma` sets the spread of its coefficients' log-magnitudes. The Hessian
is `A`, independent of `x`.

`form` selects how the *same* function is written as a program, which is what the cover sees:

- `:entrywise` — `sum_{i,j} A[i,j] * x[i] * x[j] / 2`. Every entry of `A` gets its own nonlinear
  node, and all `n^2` of them reconverge on the inputs.
- `:matvec` — `y = A*x` scalarized, then `x'y/2`. The `n` products `x[i]*y[i]` reconverge through
  the rows of `A`.
- `:factored` — `s = W'x`, then `sum(s.^2)/2`. Only `rank` nonlinear nodes, each carrying a whole
  rank-1 layer of the Hessian, so no two nonlinear contributions overlap.

For `rank = 1` the factored form's minimal cover is known exactly: `H = w*w'` forces
`h >= abs.(w)`, and `abs.(w) = sqrt.(diag(A))` attains it.

Two calls that agree in `n`, `rank`, `sigma`, and `rng` state produce the same `A` and the same
`x0` in every form, so a comparison across forms isolates the effect of program structure.
"""
function quadratic(
        n::Integer; rank::Integer = n, form::Symbol = :entrywise, sigma::Real = 0.0,
        rng::AbstractRNG = MersenneTwister(1)
    )
    n >= 1 || throw(ArgumentError("n must be positive, got $n"))
    1 <= rank <= n || throw(ArgumentError("rank must lie in 1:$n, got $rank"))
    form in (:entrywise, :matvec, :factored) ||
        throw(ArgumentError("form must be :entrywise, :matvec, or :factored, got :$form"))
    W = _coeffs(rng, n, rank; sigma)
    A = W * W'
    x0 = randn(rng, n)

    f = if form === :entrywise
        x -> _sum([A[i, j] * x[i] * x[j] for i in 1:n for j in 1:n]) / 2
    elseif form === :matvec
        function (x)
            y = [_lincomb(view(A, i, :), x) for i in 1:n]
            return _sum([x[i] * y[i] for i in 1:n]) / 2
        end
    else
        function (x)
            s = [_lincomb(view(W, :, j), x) for j in 1:rank]
            return _sum([s[j]^2 for j in 1:rank]) / 2
        end
    end

    name = "quad $form n=$n r=$rank"
    return CoverProblem(name, f, x0; twin = x -> dot(x, A * x) / 2, hessian = _ -> A)
end

# ---- deep compositions -----------------------------------------------------------------------

"""
    deep_composition(n; depth=4, width=4, sigma=0.0, planted=nothing, rng=MersenneTwister(1)) -> CoverProblem

A depth-`depth` composition of layers, each of which sums its `width` channels into a single
trunk node and then feeds that node back into all `width` channels:

    z = tanh.(W*x + b)
    repeat depth times:
        t = sum(z)                       # one node, `width` consumers
        z = tanh.(a .* t .+ c .* z)
    f = sum(z.^2) / 2

The trunk is the reconvergence: every channel of layer `l+1` depends on every channel of layer
`l`, so the paths from an input to the output multiply by `width` per layer. `width` and `depth`
therefore tune reconvergence and program length independently of `n`.

`sigma` widens the input weights `W` and biases `b` only. The recurrent weights `a` and `c` keep
unit scale (`1/sqrt(width)` per fan-in), because a wide spread there drives `tanh` into
saturation and flattens the Hessian to nothing. Heterogeneous units belong on the inputs, which is
where `sigma` puts them.

A real `planted` replaces the biases by `b = -W*xstar` with `xstar = x0 .+ planted .* randn(n)`,
so that the first layer, and with it every later layer, vanishes at `xstar`: `f(xstar) = 0`, the
minimum of `f`. For `width >= n` and `W` of full rank, `xstar` is the only zero of the first
layer. `planted` sets the distance of `x0` from `xstar` in units of the inputs; `W`, `a`, `c`, and
`x0` are those of the unplanted problem drawn from the same `rng` state. The problem carries
`xstar` as `p.xstar`.

No closed-form Hessian; differentiate `p.f`.
"""
function deep_composition(
        n::Integer; depth::Integer = 4, width::Integer = 4, sigma::Real = 0.0,
        planted::Union{Nothing, Real} = nothing, rng::AbstractRNG = MersenneTwister(1)
    )
    n >= 1 || throw(ArgumentError("n must be positive, got $n"))
    depth >= 1 || throw(ArgumentError("depth must be positive, got $depth"))
    width >= 1 || throw(ArgumentError("width must be positive, got $width"))
    (planted === nothing || (isfinite(planted) && planted >= 0)) ||
        throw(ArgumentError("planted must be nothing or a finite nonnegative distance, got $planted"))
    W = _coeffs(rng, width, n; sigma) ./ sqrt(n)
    b0 = _coeffs(rng, width; sigma)
    a = _coeffs(rng, depth, width; sigma = 0) ./ sqrt(width)
    c = _coeffs(rng, depth, width; sigma = 0)
    x0 = randn(rng, n)
    xstar = planted === nothing ? nothing : x0 .+ planted .* randn(rng, n)
    b = xstar === nothing ? b0 : -(W * xstar)

    function f(x)
        z = [tanh(_lincomb(view(W, j, :), x) + b[j]) for j in 1:width]
        for l in 1:depth
            t = _sum(z)
            z = [tanh(a[l, j] * t + c[l, j] * z[j]) for j in 1:width]
        end
        return _sum([z[j]^2 for j in 1:width]) / 2
    end

    function twin(x)
        z = tanh.(W * x .+ b)
        for l in 1:depth
            t = sum(z)
            z = tanh.(view(a, l, :) .* t .+ view(c, l, :) .* z)
        end
        return sum(abs2, z) / 2
    end

    name = planted === nothing ? "deep d=$depth w=$width n=$n" : "deep d=$depth w=$width n=$n planted=$planted"
    return CoverProblem(name, f, x0; twin, xstar)
end

# ---- random DAGs -----------------------------------------------------------------------------

const _DAG_UNARY = (:scale, :tanh, :sqr, :sin)
const _DAG_BINARY = (:add, :sub, :mul)

# Probability that a node is unary rather than binary.
const _DAG_PUNARY = 0.4

# Keep node magnitudes in [1/_DAG_BAND, _DAG_BAND]. Multiplication chains otherwise overflow or
# underflow to derivative-dead constants. Linear normalization preserves second-order structure.
const _DAG_BAND = 1.0e3

# Float64 tanh has an exactly zero derivative for large inputs. Normalize inputs outside this
# narrower band to keep both first and second derivatives active.
const _DAG_TANH_BAND = 2.0

"""
A random program, stored as the recipe that generated it: `ops[k]` builds node `ninputs + k` from
earlier nodes, and the output is the sum of `outputs`. Replaying the recipe on a vector of any
numeric type evaluates the same function, so one recipe serves the tape, the reference
derivatives, and the plain evaluation.
"""
struct DagRecipe
    ninputs::Int
    ops::Vector{Tuple{Symbol, Vector{Int}, Vector{Float64}}}
    outputs::Vector{Int}
end

_apply(op::Symbol, v, ps::Vector{Int}, cs::Vector{Float64}) =
    op === :add ? v[ps[1]] + v[ps[2]] :
    op === :sub ? v[ps[1]] - v[ps[2]] :
    op === :mul ? v[ps[1]] * v[ps[2]] :
    op === :scale ? cs[1] * v[ps[1]] :
    op === :tanh ? tanh(v[ps[1]]) :
    op === :sqr ? v[ps[1]]^2 :
    op === :sin ? sin(v[ps[1]]) :
    throw(ArgumentError("unknown DAG op :$op"))

function (r::DagRecipe)(x::AbstractVector)
    length(x) == r.ninputs ||
        throw(DimensionMismatch("recipe takes $(r.ninputs) inputs, got $(length(x))"))
    v = Vector{typeof(x[begin])}(undef, r.ninputs + length(r.ops))
    for i in 1:r.ninputs
        v[i] = x[begin + i - 1]
    end
    for (k, (op, ps, cs)) in enumerate(r.ops)
        v[r.ninputs + k] = _apply(op, v, ps, cs)
    end
    return _sum(view(v, r.outputs))
end

"""
Take a parent for a new node. With probability `sharing` (or when nothing is unconsumed) reuse an
arbitrary existing node, giving it a second consumer and so creating reconvergence; otherwise
consume a node from `frontier`, which leaves it with exactly one consumer.
"""
function _take_parent!(rng::AbstractRNG, frontier::Vector{Int}, nnodes::Int, sharing::Real)
    if isempty(frontier) || rand(rng) < sharing
        return rand(rng, 1:nnodes)
    end
    i = rand(rng, eachindex(frontier))
    p = frontier[i]
    deleteat!(frontier, i)
    return p
end

"""
    random_dag(n; nops=4n, sharing=0.5, sigma=0.0, rng=MersenneTwister(1)) -> CoverProblem

A random program of `n` inputs and at most `nops` elementary operations, drawn from `+`, `-`,
`*`, scaling, `tanh`, squaring, and `sin`. The output sums every node that no other node
consumes, so every input reaches the output. At `sharing = 0` generation also stops as soon as
a single unconsumed node remains (once any curvature-carrying op has been drawn): the only way
to grow a tree past that point is one long unary chain, whose accumulated derivative factors
wander into under- or overflow. A draw whose ops all happen to be linear gets one final
squaring node, so every generated program has a nonzero Hessian.

`sharing` controls subexpression reuse, and with it the reconvergence the cover has to contend
with. Each operand is either an unconsumed node (probability `1 - sharing`) or an arbitrary
existing node (probability `sharing`). At `sharing = 0` no node is ever consumed twice, so the
program's graph is a **tree**: each input reaches each node along at most one path. At
`sharing = 1` operands are drawn from the whole program and paths reconverge freely. Reaching
the output is graph reachability, not influence: with reuse, an input can enter only linearly
(a zero Hessian row), or even through linear paths whose unit weights cancel exactly (a zero
gradient entry as well). Both are rare, correct, and must be tolerated by downstream code.

A node whose magnitude at `x0` leaves `[1e-3, 1e3]` is followed by a normalizing scale node, so
`*` and `^2` chains can neither overflow nor collapse to a numerically zero derivative. A `tanh`
input outside `[-2, 2]` is likewise normalized to magnitude 1 first: `1 - tanh(v)^2` underflows
to exactly zero for large `v`, and a derivative-dead node erases its whole subtree from the
gradient and the Hessian. These guards are what keep the program nonlinear at every length, and
what make the recipe safe to replay at points other than `x0`.

No closed-form Hessian; differentiate `p.f`.
"""
function random_dag(
        n::Integer; nops::Integer = 4n, sharing::Real = 0.5, sigma::Real = 0.0,
        rng::AbstractRNG = MersenneTwister(1)
    )
    n >= 1 || throw(ArgumentError("n must be positive, got $n"))
    nops >= 1 || throw(ArgumentError("nops must be positive, got $nops"))
    0 <= sharing <= 1 || throw(ArgumentError("sharing must lie in 0..1, got $sharing"))
    x0 = randn(rng, n)

    vals = copy(x0)                             # values at x0, used only to keep magnitudes tame
    ops = Tuple{Symbol, Vector{Int}, Vector{Float64}}[]
    frontier = collect(1:n)

    function emit!(op, ps, cs, v)
        push!(ops, (op, ps, cs))
        push!(vals, v)
        return length(vals)
    end

    # Normalize node k's magnitude into the band and return it to the unconsumed pool.
    function finish!(k)
        v = abs(vals[k])
        if v != 0 && !(1 / _DAG_BAND <= v <= _DAG_BAND)
            k = emit!(:scale, [k], [1 / v], sign(vals[k]))
        end
        push!(frontier, k)
        return k
    end

    ncurved = 0                                 # ops with a nonzero second derivative
    for _ in 1:nops
        # Without sharing, a one-node frontier can only grow a numerically unstable unary chain.
        # Stop once the graph already contains curvature.
        sharing == 0 && length(frontier) == 1 && ncurved > 0 && break
        # Preserve the tree property when fewer than two unconsumed nodes remain.
        binary_ok = sharing > 0 || length(frontier) >= 2
        if !binary_ok || rand(rng) < _DAG_PUNARY
            op = rand(rng, _DAG_UNARY)
            p = _take_parent!(rng, frontier, length(vals), sharing)
            if op === :tanh && abs(vals[p]) > _DAG_TANH_BAND
                p = emit!(:scale, [p], [1 / abs(vals[p])], sign(vals[p]))
            end
            cs = op === :scale ? [_coeffs(rng, 1; sigma)[1]] : Float64[]
            k = emit!(op, [p], cs, _apply(op, vals, [p], cs))
        else
            op = rand(rng, _DAG_BINARY)
            p1 = _take_parent!(rng, frontier, length(vals), sharing)
            p2 = _take_parent!(rng, frontier, length(vals), sharing)
            k = emit!(op, [p1, p2], Float64[], _apply(op, vals, [p1, p2], Float64[]))
        end
        ncurved += op in (:mul, :tanh, :sqr, :sin)
        finish!(k)
    end

    if ncurved == 0
        # Ensure every generated problem has nonzero curvature.
        p = _take_parent!(rng, frontier, length(vals), 0.0)
        finish!(emit!(:sqr, [p], Float64[], vals[p]^2))
    end

    recipe = DagRecipe(n, ops, sort!(frontier))
    name = "dag n=$n share=$(round(sharing; digits = 2))"
    return CoverProblem(name, recipe, x0)
end

# ---- multilayer perceptrons -------------------------------------------------------------------

"""
    mlp(sizes; batch=4, sigma=0.0, activation=tanh, rng=MersenneTwister(1)) -> CoverProblem

The squared-error loss of a feedforward network, with the **parameters** as inputs. For
`sizes = (n0, n1, ..., nK)`, layer `l = 1:K` has a weight matrix `W_l` of size `n_l x n_{l-1}` and
a bias vector `b_l` of length `n_l`, flattened `(W1, b1, W2, b2, ..., WK, bK)` into one vector of
length `p = sum_l n_l*(n_{l-1}+1)`:

    z = data column                    # n0, fixed
    for l = 1:K-1
        z = activation.(W_l*z .+ b_l)  # hidden layers
    y = W_K*z .+ b_K                   # output layer, linear
    f = sum_over_columns( sum(abs2, y - target) ) / 2

`batch` input/target column pairs are drawn once at construction and closed over; only the
parameters vary thereafter. `f` is written in array notation: a matvec whose `TVar` factor meets a
constant factor (the data column feeding layer 1, or feeding layer `K` directly when `K = 1`)
records one linear tape node per row, while a matvec between two
`TVar` factors (every hidden-to-hidden transition) records one
inner-product node per row. Each column's error sums as
one linear node, and the batch total sums as another, so the tape stays short despite the array
notation.

`twin` computes the same loss by an explicit scalar loop: every row's dot product is the
left-to-right chain `w[1]*z[1] + w[2]*z[2] + ...` in the order a dot node's parents follow, bias
and activation are applied per scalar, and every sum (over output units, over the batch) is an
explicit binary `+` chain. It therefore matches `f` node for node under a shared accumulation
order, rather than taking an independent route through array operations as the other families'
twins do.

`sigma` spreads the log-magnitude of the initial weights and biases, mirroring
[`deep_composition`](@ref)'s convention; the weights are additionally scaled by `1/sqrt(n_{l-1})`
so a wider input layer does not by itself inflate the initial activations.

No closed-form Hessian; differentiate `p.f`.
"""
function mlp(
        sizes; batch::Integer = 4, sigma::Real = 0.0, activation = tanh,
        rng::AbstractRNG = MersenneTwister(1)
    )
    ns = collect(Int, sizes)
    length(ns) >= 2 ||
        throw(ArgumentError("sizes must have at least 2 entries (n0 and n1), got $(length(ns))"))
    all(>=(1), ns) || throw(ArgumentError("layer widths must be positive, got $ns"))
    K = length(ns) - 1

    offsets = Vector{UnitRange{Int}}(undef, 2K)
    pos = 1
    for l in 1:K
        nl, nlm1 = ns[l + 1], ns[l]
        offsets[2l - 1] = pos:(pos + nl * nlm1 - 1)
        pos += nl * nlm1
        offsets[2l] = pos:(pos + nl - 1)
        pos += nl
    end
    p = pos - 1

    x0 = Vector{Float64}(undef, p)
    for l in 1:K
        nl, nlm1 = ns[l + 1], ns[l]
        x0[offsets[2l - 1]] = vec(_coeffs(rng, nl, nlm1; sigma) ./ sqrt(nlm1))
        x0[offsets[2l]] = _coeffs(rng, nl; sigma)
    end

    X = randn(rng, ns[1], batch)
    Y = randn(rng, ns[end], batch)

    function f(params)
        length(params) == p ||
            throw(DimensionMismatch("mlp expects $p parameters, got $(length(params))"))
        Base.require_one_based_indexing(params)
        terms = map(1:batch) do bcol
            z = view(X, :, bcol)
            for l in 1:(K - 1)
                nl, nlm1 = ns[l + 1], ns[l]
                Wl = reshape(view(params, offsets[2l - 1]), nl, nlm1)
                bl = view(params, offsets[2l])
                z = activation.(Wl * z .+ bl)
            end
            nK, nKm1 = ns[K + 1], ns[K]
            WK = reshape(view(params, offsets[2K - 1]), nK, nKm1)
            bK = view(params, offsets[2K])
            y = WK * z .+ bK
            sum(abs2, y .- view(Y, :, bcol))
        end
        return sum(terms) / 2
    end

    function twin(params)
        length(params) == p ||
            throw(DimensionMismatch("mlp expects $p parameters, got $(length(params))"))
        Base.require_one_based_indexing(params)
        terms = map(1:batch) do bcol
            z = [X[i, bcol] for i in 1:ns[1]]
            for l in 1:(K - 1)
                nl, nlm1 = ns[l + 1], ns[l]
                Wl = reshape(view(params, offsets[2l - 1]), nl, nlm1)
                bl = view(params, offsets[2l])
                z = [activation(_rowdot(view(Wl, i, :), z, nlm1) + bl[i]) for i in 1:nl]
            end
            nK, nKm1 = ns[K + 1], ns[K]
            WK = reshape(view(params, offsets[2K - 1]), nK, nKm1)
            bK = view(params, offsets[2K])
            y = [_rowdot(view(WK, i, :), z, nKm1) + bK[i] for i in 1:nK]
            _sum([abs2(y[i] - Y[i, bcol]) for i in 1:nK])
        end
        return _sum(terms) / 2
    end

    name = "mlp $(join(ns, '-')) b=$batch"
    return CoverProblem(name, f, x0; twin)
end

# ---- Lotka--Volterra parameter estimation -------------------------------------------------------

"""
    lotka_volterra(m; delta=0.3, decades=2.0, ntraj=3, h=0.1, nsteps=50, every=5,
                   rng=MersenneTwister(1)) -> CoverProblem

Least-squares estimation of the parameters of a generalized Lotka--Volterra system of `m`
species,

    dy_i/dt = y_i * (r_i + sum_j A[i,j] * y_j),

from its own simulated trajectories. The inputs are the parameters `x = vcat(r, vec(A))`, so
`n = m + m^2`. The program integrates `ntraj` trajectories by `nsteps` classical Runge--Kutta
steps of size `h`, observes every `every`-th step, and sums the squared misfits to the data, each
divided by the equilibrium abundance of its species:

    f(x) = sum over trajectories, observations o, species i of ((y_i(o; x) - data_i(o)) / ybar_i)^2 / 2

The data come from planted parameters `xstar`, built around a positive equilibrium `ybar` whose
entries are spread log-uniformly over `decades` decades: `A[i,j] = B[i,j] / ybar[j]` with `B` the
negative identity plus a random perturbation whose symmetric part is kept negative definite, and
`r = -A*ybar`, so `ybar` is a stable equilibrium. The parameters of the abundant and the rare
species therefore differ in units by `decades` decades. Initial conditions are `ybar` times
log-normal factors of standard deviation 1/2. The data are simulated by `f`'s own arithmetic, so
`f(xstar) = 0` exactly, the minimum of `f`; the problem carries `xstar` as `p.xstar`.

The start is `x0 = xstar .+ delta .* abs.(xstar) .* randn(n)`: `delta` is the relative
perturbation of each parameter. Far from `xstar` the dynamics can blow up within the horizon, and
`f` then returns `Inf` or `NaN`.

Every step of the integrator reuses every species' state in every right-hand side, so paths from a
parameter to the output reconverge at each of the `4 * nsteps` stages.

No closed-form Hessian; differentiate `p.f`.
"""
function lotka_volterra(
        m::Integer; delta::Real = 0.3, decades::Real = 2.0, ntraj::Integer = 3, h::Real = 0.1,
        nsteps::Integer = 50, every::Integer = 5, rng::AbstractRNG = MersenneTwister(1)
    )
    m >= 1 || throw(ArgumentError("m must be positive, got $m"))
    (isfinite(delta) && delta >= 0) || throw(ArgumentError("delta must be finite and nonnegative, got $delta"))
    (isfinite(decades) && decades >= 0) || throw(ArgumentError("decades must be finite and nonnegative, got $decades"))
    ntraj >= 1 || throw(ArgumentError("ntraj must be positive, got $ntraj"))
    (isfinite(h) && h > 0) || throw(ArgumentError("h must be finite and positive, got $h"))
    nsteps >= 1 || throw(ArgumentError("nsteps must be positive, got $nsteps"))
    1 <= every <= nsteps || throw(ArgumentError("every must lie in 1:$nsteps, got $every"))
    n = m + m^2

    ybar = 10 .^ (decades .* (rand(rng, m) .- 0.5))
    B = -Matrix{Float64}(I, m, m) .+ 0.4 .* randn(rng, m, m) ./ sqrt(m)
    # Shift B so the largest eigenvalue of its symmetric part is at most -0.2.
    λ = maximum(eigvals(Symmetric((B .+ B') ./ 2)))
    λ > -0.2 && (B .-= (λ + 0.2) .* Matrix{Float64}(I, m, m))
    A = B ./ ybar'
    r = -(A * ybar)
    xstar = vcat(r, vec(A))
    inits = [ybar .* exp.(0.5 .* randn(rng, m)) for _ in 1:ntraj]

    # Observed states of the trajectory from `y0` under parameters `x`, by scalar arithmetic.
    function simulate(x, y0)
        Base.require_one_based_indexing(x)
        length(x) == n || throw(DimensionMismatch("lotka_volterra expects $n parameters, got $(length(x))"))
        rhs(y) = [y[i] * (x[i] + _sum([x[m + (j - 1) * m + i] * y[j] for j in 1:m])) for i in 1:m]
        obs = Vector{typeof(x[1] * y0[1])}[]
        y = y0
        for k in 1:nsteps
            k1 = rhs(y)
            k2 = rhs([y[i] + (h / 2) * k1[i] for i in 1:m])
            k3 = rhs([y[i] + (h / 2) * k2[i] for i in 1:m])
            k4 = rhs([y[i] + h * k3[i] for i in 1:m])
            y = [y[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in 1:m]
            k % every == 0 && push!(obs, y)
        end
        return obs
    end
    data = [[Float64.(o) for o in simulate(xstar, y0)] for y0 in inits]

    function f(x)
        terms = [((o[i] - d[i]) / ybar[i])^2 for (y0, dt) in zip(inits, data)
                 for (o, d) in zip(simulate(x, y0), dt) for i in 1:m]
        return _sum(terms) / 2
    end

    function twin(x)
        Base.require_one_based_indexing(x)
        rr = x[1:m]
        AA = reshape(x[(m + 1):end], m, m)
        rhs(y) = y .* (rr .+ AA * y)
        total = zero(eltype(x))
        for (y0, dt) in zip(inits, data)
            y = y0
            for k in 1:nsteps
                k1 = rhs(y)
                k2 = rhs(y .+ (h / 2) .* k1)
                k3 = rhs(y .+ (h / 2) .* k2)
                k4 = rhs(y .+ h .* k3)
                y = y .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
                k % every == 0 && (total += sum(abs2, (y .- dt[k ÷ every]) ./ ybar))
            end
        end
        return total / 2
    end

    x0 = xstar .+ delta .* abs.(xstar) .* randn(rng, n)
    name = "lotka-volterra m=$m δ=$delta K=$nsteps"
    return CoverProblem(name, f, x0; twin, xstar)
end

# ---- cancellation pairs ------------------------------------------------------------------------

"""
    cancellation(n; m=n, delta=1e-3, aligned=false, support=1:n, sigma=0.0, rng=MersenneTwister(1)) -> CoverProblem

An adversarial pair of programs built from `m` quartets of squared linear forms whose signed sum
cancels exactly, riding on a `delta`-weighted rank-1 quadratic:

    acc = (delta/2) * (q'x)^2
    for j in 1:m
        acc += 0.5*(a_j'xS)^2 + 0.5*(b_j'xS)^2 + s*(c_j'xS)^2 + s*(d_j'xS)^2
    end

where `xS = x[support]`, `c_j = (a_j+b_j)/sqrt(2)`, `d_j = (a_j-b_j)/sqrt(2)`, and `a_j`, `b_j`
are drawn orthogonal to `x0[support]`. At `s = -0.5` (`aligned = false`, the default) the rotation
identity `c_j c_j' + d_j d_j' = a_j a_j' + b_j b_j'` makes every quartet's contribution to the
Hessian cancel exactly; at `s = 0.5` (`aligned = true`) the same identity makes it add. The two
variants differ *only* in the sign of the two constants multiplying the `c_j` and `d_j` squares:
every node of the tape up to that sign carries the same value in both, because every quartet
linear form vanishes at `x0` (orthogonality to `x0[support]`), so `f`, its gradient, and the
magnitude of every elemental partial and adjoint coincide between the two variants — only the
*curvature* they carry differs.

Closed forms (the Hessian is constant in `x`):

    H_cancel  = delta * q*q'
    H_aligned = delta * q*q' + sum_j 2*(a_j*a_j' + b_j*b_j')   (scattered onto `support`)

The minimal cover of `H_cancel` is `hstar = sqrt(delta) .* abs.(q) = sqrt.(diag(H_cancel))` (a
rank-1 `|H|`, since `H_cancel` is itself rank 1). The quadrature merge of the cancel variant
attains `sqrt.(diag(H_aligned))` exactly — the sign flip that cancels the Hessian does not touch
the elementwise bound `B`, which sees only magnitudes — so its looseness against `hstar` is
`sqrt.(kappa)` with `kappa_i = diag(H_aligned)_i / diag(H_cancel)_i`, the per-coordinate
cancellation index.

`m` sets the number of quartets and hence the size of `kappa` on the support (more quartets, more
cancellation, larger `kappa`); `delta` sets the scale of the surviving rank-1 term relative to the
quartets; `support` restricts the quartets to a coordinate subset, leaving `off-support`
coordinates untouched by any cancellation; `sigma` widens the log-magnitude spread of the quartet
coefficients and of `q`. Two calls that agree in `n`, `m`, `delta`, `support`, `sigma`, and `rng`
state draw the same `x0`, `q`, and quartets regardless of `aligned`, so a cancel/aligned pair built
from fresh, identically seeded rngs shares every random draw and differs only in the program's
two constants.
"""
function cancellation(
        n::Integer; m::Integer = n, delta::Real = 1.0e-3, aligned::Bool = false,
        support = 1:n, sigma::Real = 0.0, rng::AbstractRNG = MersenneTwister(1)
    )
    n >= 2 || throw(ArgumentError("n must be at least 2, got $n"))
    m >= 1 || throw(ArgumentError("m must be positive, got $m"))
    delta > 0 || throw(ArgumentError("delta must be positive, got $delta"))
    supp = collect(Int, support)
    all(i -> 1 <= i <= n, supp) ||
        throw(ArgumentError("support must be a subset of 1:$n, got $support"))
    allunique(supp) || throw(ArgumentError("support entries must be unique, got $support"))
    length(supp) >= 2 ||
        throw(ArgumentError("support must have at least 2 entries, got $(length(supp))"))

    x0 = randn(rng, n)
    q = ifelse.(rand(rng, n) .< 0.5, -1.0, 1.0) .* exp.(sigma .* randn(rng, n))

    x0S = view(x0, supp)
    nx2 = dot(x0S, x0S)

    function draw_orth()
        for _ in 1:100
            v = _coeffs(rng, length(supp); sigma)
            nv = norm(v)
            vperp = v .- (dot(v, x0S) / nx2) .* x0S
            norm(vperp) >= 1.0e-8 * nv && return vperp
        end
        error("failed to draw a quartet vector orthogonal to x0[support] after 100 attempts")
    end

    as_, bs_, cs_, ds_ = [Vector{Float64}[] for _ in 1:4]
    for _ in 1:m
        a = draw_orth()
        b = draw_orth()
        push!(as_, a)
        push!(bs_, b)
        push!(cs_, (a .+ b) ./ sqrt(2))
        push!(ds_, (a .- b) ./ sqrt(2))
    end

    s = aligned ? 0.5 : -0.5

    function f(x)
        xS = view(x, supp)
        acc = (delta / 2) * _lincomb(q, x)^2
        for j in 1:m
            acc = acc + 0.5 * _lincomb(as_[j], xS)^2 + 0.5 * _lincomb(bs_[j], xS)^2 +
                s * _lincomb(cs_[j], xS)^2 + s * _lincomb(ds_[j], xS)^2
        end
        return acc
    end

    function twin(x)
        xS = view(x, supp)
        val = (delta / 2) * dot(q, x)^2
        if aligned
            for j in 1:m
                val += dot(as_[j], xS)^2 + dot(bs_[j], xS)^2
            end
        end
        return val
    end

    H = delta .* (q * q')
    if aligned
        for j in 1:m
            H[supp, supp] .+= 2 .* (as_[j] * as_[j]' .+ bs_[j] * bs_[j]')
        end
    end

    name = (aligned ? "aligned" : "cancel") * " n=$n m=$m delta=$delta"
    length(supp) < n && (name *= " S=$(length(supp))")
    return CoverProblem(name, f, x0; twin, hessian = _ -> H)
end

# ---- stationary outputs ----------------------------------------------------------------------
#
# Programs whose output is *stationary* in an interior node. That is what makes a signed adjoint
# collapse while the unsigned sum over paths from the same node does not, so these are the
# families that separate the two adjoint channels. Everything above cancels in the intermediate
# gradients instead, along paths from an input.

"""
    adjoint_cancellation(n; delta=1e-2, sigma=0.0, rng=MersenneTwister(1)) -> CoverProblem

A chain of `n - 1` multiplication nodes `u_j = x_j * x_{j+1}`, each reaching the output through
`u_j - sin(u_j)`, evaluated where every `u_j` is small.

`u - sin(u)` has derivative `1 - cos(u)`, which vanishes to second order at `u = 0`, so the output
is nearly stationary in each `u_j`: the signed adjoint of that node is `~u_j^2/2` while its two
paths to the output contribute `~2` in absolute-value arithmetic. The adjoint index
`acheck/abs(vbar)` at the node is therefore `~4/u_j^2` and the cover separation `~2/u_j`, both set
by `delta` alone.

`x0` alternates `delta, 1, delta, 1, ...`, so every `u_j` is `delta` and every unit carries the
same index. The units overlap — `x_j` feeds both `u_{j-1}` and `u_j` — so the Hessian is
tridiagonal rather than block diagonal, and the separation is a property of each node rather than
of the problem size: **it is the same at every `n`**, which is what makes the family usable in a
size sweep.

Each input reaches the output along one path per unit, so no intermediate gradient cancels: its
absolute-value bound is exact, and the family isolates the cancellation in the adjoints. `sigma`
spreads the log-magnitudes of the odd coordinates, spreading the per-unit index without touching
the mechanism.
"""
function adjoint_cancellation(
        n::Integer; delta::Real = 1.0e-2, sigma::Real = 0.0,
        rng::AbstractRNG = MersenneTwister(1)
    )
    n >= 2 || throw(ArgumentError("n must be at least 2, got $n"))
    delta > 0 || throw(ArgumentError("delta must be positive, got $delta"))
    nunits = n - 1
    x0 = [isodd(i) ? delta * exp(sigma * randn(rng)) : 1.0 for i in 1:n]

    function f(x)
        u = [x[j] * x[j + 1] for j in 1:nunits]
        return _sum([u[j] - sin(u[j]) for j in 1:nunits])
    end

    function twin(x)
        v = x[1:nunits] .* x[2:n]
        return sum(v .- sin.(v))
    end

    # Each unit contributes sin(u_j) * (du_j)(du_j)' + (1 - cos(u_j)) * d2u_j, and d2u_j is 1 in
    # the (j, j+1) entry alone. Off-diagonals come from one unit each; diagonals from two.
    function hessian(x)
        H = zeros(promote_type(eltype(x), Float64), n, n)
        for j in 1:nunits
            uj = x[j] * x[j + 1]
            s, c = sin(uj), 1 - cos(uj)
            H[j, j] += s * x[j + 1]^2
            H[j + 1, j + 1] += s * x[j]^2
            H[j, j + 1] = s * x[j] * x[j + 1] + c
            H[j + 1, j] = H[j, j + 1]
        end
        return H
    end

    return CoverProblem("adjcancel n=$n δ=$delta", f, x0; twin, hessian)
end

"""
    converged_inner_solve(; steps=2) -> CoverProblem

A bilevel program: `steps` Newton iterations on the inner problem `min_u (u - x1)^2/2 + x2 u^4/4`,
with the output that same inner objective at the final iterate.

By the envelope theorem the output is stationary in `u` at the exact inner minimizer, so the
adjoint through the final iterate collapses as the solve converges while the unsigned sum over
paths does not. Newton doubles the accuracy each step, so the adjoint index grows
superexponentially in `steps`. At `steps = 0` the inner variable keeps its initial value and
nothing cancels, which is the control.

Unlike [`adjoint_cancellation`](@ref) this engages the intermediate-gradient channel too — the
realistic case, since the same structure appears in implicit layers, self-consistent field
iterations, and any objective differentiated through a converged inner solve. Its knob is `steps`
rather than a size, so it takes no `n` and is not one of [`families`](@ref).

Past four steps the inner residual is at roundoff and a further step measures the floor rather
than the mechanism.
"""
function converged_inner_solve(; steps::Integer = 2)
    steps >= 0 || throw(ArgumentError("steps must be nonnegative, got $steps"))
    function f(x)
        u = x[1]
        for _ in 1:steps
            g = (u - x[1]) + x[2] * u^3
            h = 1 + 3 * x[2] * u^2
            u = u - g / h
        end
        return (u - x[1])^2 / 2 + x[2] * u^4 / 4
    end
    return CoverProblem("innersolve k=$steps", f, [1.3, 0.7])
end

# ---- vector-output cancellation pairs --------------------------------------------------------

"""
    JacobianProblem

A vector-valued program together with the references needed to judge an asymmetric cover of its
Jacobian — the vector-output counterpart of [`CoverProblem`](@ref).

- `name`: a short identifier, unique within a family sweep.
- `r`: the program, mapping an input vector to an output vector. Built only from the elementals
  overloaded for `DerivativeCovers.TVar`, so `r(x)` records a tape when `x` is a vector of `TVar`s
  and evaluates normally on any other numeric vector.
- `x0`: the point at which the Jacobian is covered.
- `nout`: the number of outputs of `r`.
- `twin`: an independent idiomatic-Julia implementation of the same map, or `nothing`.
- `jacobian`: `x -> J`, the analytic `nout × n` Jacobian, or `nothing` when no closed form is
  available. Use [`exact_jacobian`](@ref) rather than the field.
"""
struct JacobianProblem{F, G, J}
    name::String
    r::F
    x0::Vector{Float64}
    nout::Int
    twin::G
    jacobian::J
end

JacobianProblem(
    name::AbstractString, r, x0::AbstractVector, nout::Integer;
    twin = nothing, jacobian = nothing
) =
    JacobianProblem(String(name), r, Vector{Float64}(x0), Int(nout), twin, jacobian)

nvars(p::JacobianProblem) = length(p.x0)
has_twin(p::JacobianProblem) = p.twin !== nothing

"""
    noutputs(p::JacobianProblem)

Number of outputs of `p.r`.
"""
noutputs(p::JacobianProblem) = p.nout

"""
    has_exact_jacobian(p::JacobianProblem)

Whether `p` carries an analytic Jacobian. When it does not, a reference Jacobian must come from
elsewhere (`ForwardDiff.jacobian(p.r, p.x0)`).
"""
has_exact_jacobian(p::JacobianProblem) = p.jacobian !== nothing

"""
    exact_jacobian(p::JacobianProblem, x=p.x0) -> J

The analytic Jacobian of `p` at `x`. Errors when `p` has none; test with
[`has_exact_jacobian`](@ref) first.
"""
function exact_jacobian(p::JacobianProblem, x::AbstractVector = p.x0)
    has_exact_jacobian(p) ||
        throw(ArgumentError("problem \"$(p.name)\" has no analytic Jacobian; differentiate `p.r` instead"))
    return p.jacobian(x)
end

"""
    jacobian_cancellation(n; nout=4, m=n, delta=1e-3, aligned=false, support=1:n, sigma=0.0,
                          rng=MersenneTwister(1)) -> JacobianProblem

An adversarial pair of vector-valued programs built from `m` quartets of *linear* forms whose
signed sum cancels exactly, riding on a `delta`-weighted rank-1 Jacobian and fanned out to
`nout` outputs:

    T = sum_i [ 0.5*(a_i'xS) + 0.5*(b_i'xS) + s*(c_i'xS) + s*(d_i'xS) ]
    r_k = delta * q[k] * (p'x) + q[k] * T

where `xS = x[support]` and `c_i .+ d_i == a_i .+ b_i` entrywise (an unequal split
`c_i = t_i .* (a_i .+ b_i)`, `t_i` entrywise in (0, 1), so all four forms are distinct and none
is a scalar multiple of another). At `s = -0.5` (`aligned = false`, the default) each quartet's
contribution to every Jacobian entry cancels exactly; at `s = 0.5` (`aligned = true`) it adds.
The two variants differ *only* in the sign of the constant multiplying the `c_i` and `d_i`
forms: every quartet form vanishes at `x0` (orthogonality, arranged by the draw) and within each
support column all four forms share one sign, so node values and the magnitude of every
elemental partial coincide between the two variants — the surrogate `Ũ` and every merge built
from it are identical, and only the *signed* Jacobian differs.

Closed forms (the Jacobian is constant in `x`; `ghat = sum_i (a_i .+ b_i)` scattered onto
`support`, `g = abs.(ghat)`):

    J_cancel  = delta .* q * p'
    J_aligned = q * (delta .* p .+ ghat)'
    Utilde    = abs.(q) * (delta .* abs.(p) .+ g)'     (both variants)

The sign alignment makes `Utilde == abs.(J_aligned)` exactly, and both `Utilde` and the two
Jacobians are rank 1, so the minimal cover of each is exact and analytic. The per-entry
cancellation index is constant across outputs: `kappa_j = 1 + g_j / (delta * abs(p_j))` on the
support, `1` off it — more quartets or smaller `delta`, larger `kappa`; `support` restricts the
cancellation to a column subset, making the `kappa` profile anisotropic.

`sigma` widens the log-magnitude spread of the quartet coefficients *and* of the output scales
`q`, whose spread sets the sum/max merge gap `sum(abs, q)/maximum(abs, q)` — the effective
output multiplicity, `nout` at `sigma = 0` and approaching 1 as one output dominates. Two calls
that agree in everything but `aligned` share every random draw and differ only in the program's
one constant.
"""
function jacobian_cancellation(
        n::Integer; nout::Integer = 4, m::Integer = n, delta::Real = 1.0e-3,
        aligned::Bool = false, support = 1:n, sigma::Real = 0.0,
        rng::AbstractRNG = MersenneTwister(1)
    )
    n >= 2 || throw(ArgumentError("n must be at least 2, got $n"))
    nout >= 1 || throw(ArgumentError("nout must be positive, got $nout"))
    m >= 1 || throw(ArgumentError("m must be positive, got $m"))
    delta > 0 || throw(ArgumentError("delta must be positive, got $delta"))
    supp = collect(Int, support)
    all(i -> 1 <= i <= n, supp) ||
        throw(ArgumentError("support must be a subset of 1:$n, got $support"))
    allunique(supp) || throw(ArgumentError("support entries must be unique, got $support"))
    length(supp) >= 2 ||
        throw(ArgumentError("support must have at least 2 entries, got $(length(supp))"))

    x0 = randn(rng, n)
    ns = length(supp)
    x0S = x0[supp]
    all(!iszero, x0S) || error("x0 vanishes on a support coordinate; cannot balance the draw")

    # One sign per support column; all quartet forms carry it, which is what aligns the paths.
    s = ifelse.(rand(rng, ns) .< 0.5, -1.0, 1.0)
    w = s .* x0S
    if all(>(0), w) || all(<(0), w)
        # Positive balancing weights require w to contain both signs.
        i = argmin(abs.(w))
        s[i] = -s[i]
        w[i] = -w[i]
    end

    q = ifelse.(rand(rng, nout) .< 0.5, -1.0, 1.0) .* exp.(sigma .* randn(rng, nout))
    pmag = (abs.(randn(rng, n)) .+ 0.2) .* exp.(sigma .* randn(rng, n))
    psign = ifelse.(rand(rng, n) .< 0.5, -1.0, 1.0)
    pvec = psign .* pmag
    pvec[supp] .= s .* pmag[supp]         # on the support, sign(p_j) must match the forms

    # Rescale one sign group so positive alpha satisfies sum(alpha .* w) = 0.
    function draw_balanced()
        alpha = (abs.(randn(rng, ns)) .+ 0.1) .* exp.(sigma .* randn(rng, ns))
        Sp, Sm = w .> 0, w .< 0
        gamma = sum(alpha[Sp] .* w[Sp]) / -sum(alpha[Sm] .* w[Sm])
        alpha[Sm] .*= gamma
        return alpha
    end

    as_, bs_, cs_, ds_ = [Vector{Float64}[] for _ in 1:4]
    for _ in 1:m
        a = s .* draw_balanced()
        b = s .* draw_balanced()
        ab = a .+ b
        # Split ab into c and d while preserving c'x0S = d'x0S = 0. Keep t away from the
        # endpoints so both halves retain the column sign.
        mvec = ab .* x0S
        u = rand(rng, ns) .* 0.5 .- 0.25
        u .-= (sum(u .* mvec) / sum(abs2, mvec)) .* mvec
        rho = maximum(abs, u)
        rho > 0.45 && (u .*= 0.45 / rho)
        t = 0.5 .+ u
        push!(as_, a)
        push!(bs_, b)
        push!(cs_, t .* ab)
        push!(ds_, (1 .- t) .* ab)
    end

    sgn = aligned ? 0.5 : -0.5

    function r(x)
        xS = view(x, supp)
        px = _lincomb(pvec, x)
        T = 0.5 * _lincomb(as_[1], xS) + 0.5 * _lincomb(bs_[1], xS) +
            sgn * _lincomb(cs_[1], xS) + sgn * _lincomb(ds_[1], xS)
        for i in 2:m
            T = T + 0.5 * _lincomb(as_[i], xS) + 0.5 * _lincomb(bs_[i], xS) +
                sgn * _lincomb(cs_[i], xS) + sgn * _lincomb(ds_[i], xS)
        end
        return [delta * q[k] * px + q[k] * T for k in 1:nout]
    end

    ghat = zeros(n)
    for i in 1:m
        ghat[supp] .+= as_[i] .+ bs_[i]
    end

    function twin(x)
        px = dot(pvec, x)
        aligned || return (delta * px) .* q
        return (delta * px + dot(ghat, x)) .* q
    end

    # Share the delta term literally so off-support entries are bit-identical.
    J = aligned ? delta .* q * pvec' .+ q * ghat' : delta .* q * pvec'

    name = (aligned ? "aligned" : "cancel") * " jac n=$n nout=$nout m=$m delta=$delta"
    length(supp) < n && (name *= " S=$(length(supp))")
    return JacobianProblem(name, r, x0, nout; twin, jacobian = _ -> J)
end

# ---- transformations and sweeps --------------------------------------------------------------

"""
    rescale(p::CoverProblem, d::AbstractVector) -> CoverProblem

The same function in rescaled input coordinates: `g(y) = f(y ./ d)` at `y0 = d .* x0`, whose
Hessian is `Diagonal(inv.(d)) * H * Diagonal(inv.(d))`.

This is the input-units axis of dynamic range, and it is exactly the transformation a
scale-invariant cover must be covariant under: a cover `h` of `p` at `x0` must reappear as
`h ./ d` for the rescaled problem. Take `d` from [`logscales`](@ref) for a random spread.
"""
function rescale(p::CoverProblem, d::AbstractVector)
    length(d) == nvars(p) ||
        throw(DimensionMismatch("need one scale per input: got $(length(d)) for $(nvars(p)) inputs"))
    all(>(0), d) || throw(ArgumentError("scale factors must be positive"))
    dv = Vector{Float64}(d)
    f, tw, hs, xs = p.f, p.twin, p.hessian, p.xstar
    D = Diagonal(inv.(dv))
    return CoverProblem(
        p.name * " rescaled",
        y -> f(y ./ dv),
        dv .* p.x0,
        tw === nothing ? nothing : y -> tw(y ./ dv),
        hs === nothing ? nothing : y -> D * hs(y ./ dv) * D,
        xs === nothing ? nothing : dv .* xs
    )
end

"""
    families(; sigma=0.0, seed=1) -> Vector{Pair{String,Any}}

The standard sweep: named generators `n -> CoverProblem`, one per family, differing in how
severely the paths through the program reconverge. Pass the same `n` to each to compare families
at fixed problem size, or sweep `n` within one to measure how a cover's looseness grows.

The three quadratic forms are seeded alike, so at a given `n` they carry the *same* Hessian and
differ only as programs.

Every family but the last cancels in the intermediate gradients; `adjcancel` cancels in the
adjoints, and is the only one whose looseness is set by a knob rather than by `n`.
"""
function families(; sigma::Real = 0.0, seed::Integer = 1)
    return [
        "quad entrywise" => n -> quadratic(n; form = :entrywise, sigma, rng = MersenneTwister(seed)),
        "quad matvec" => n -> quadratic(n; form = :matvec, sigma, rng = MersenneTwister(seed)),
        "quad factored" => n -> quadratic(n; form = :factored, sigma, rng = MersenneTwister(seed)),
        "deep w=4" => n -> deep_composition(n; depth = 4, width = 4, sigma, rng = MersenneTwister(seed)),
        "dag tree" => n -> random_dag(n; sharing = 0.0, sigma, rng = MersenneTwister(seed)),
        "dag shared" => n -> random_dag(n; sharing = 0.6, sigma, rng = MersenneTwister(seed)),
        "adjcancel δ=0.01" => n -> adjoint_cancellation(n; delta = 1.0e-2, sigma, rng = MersenneTwister(seed)),
    ]
end

end # module
