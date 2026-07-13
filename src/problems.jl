"""
    CoverAD.Problems

Parameterized families of scalar programs for studying cover quality.

Each generator returns a [`CoverProblem`](@ref): a program `f` written from the elementals
[`TVar`](@ref) overloads (so it can be taped), the point `x0` to evaluate it at, an optional
idiomatic-Julia `twin` that computes the same function by an independent route, and an optional
analytic `hessian`. The families differ in the property that drives cover looseness — how much
the paths from an input to the output reconverge — and each has knobs for size, reconvergence,
and dynamic range:

- [`quadratic`](@ref): `x'Ax/2` with `A = W*W'` of prescribed rank, written entrywise, as a
  matrix-vector product, or in factored form. The same `A` in three programs, so the three
  differ *only* in graph structure.
- [`deep_composition`](@ref): layers of `width` channels around a shared trunk, so each layer
  fans a single node out to `width` consumers that immediately reconverge.
- [`random_dag`](@ref): a random program whose `sharing` parameter interpolates from a tree
  (`sharing = 0`) to heavy subexpression reuse (`sharing = 1`).

Wide dynamic range enters along two independent axes: `sigma` widens the log-magnitude spread of
a family's coefficients, while [`rescale`](@ref) (with scales from [`logscales`](@ref)) spreads
the units of the *inputs*.
"""
module Problems

using LinearAlgebra: Diagonal, dot
using Random: AbstractRNG, MersenneTwister, rand, randn

export CoverProblem, nvars, exact_hessian, has_exact_hessian, has_twin
export quadratic, deep_composition, random_dag, rescale, logscales, families

"""
    CoverProblem

A scalar program together with the references needed to judge a cover of its Hessian.

- `name`: a short identifier, unique within a family sweep.
- `f`: the program. Built only from the elementals overloaded for [`TVar`](@ref), so `f(x)`
  records a tape when `x` is a vector of `TVar`s, and evaluates normally on any other numeric
  vector (`Float64`, `ForwardDiff.Dual`, ...).
- `x0`: the point at which the Hessian is covered.
- `twin`: an idiomatic-Julia implementation of the same function, or `nothing`. It is written
  independently of `f` — using array operations `f` cannot express — so agreement between the two
  is evidence that the scalarized program is the function it claims to be.
- `hessian`: `x -> H`, the analytic Hessian, or `nothing` when no closed form is available. Use
  [`exact_hessian`](@ref) rather than the field.
"""
struct CoverProblem{F,G,H}
    name::String
    f::F
    x0::Vector{Float64}
    twin::G
    hessian::H
end

CoverProblem(name::AbstractString, f, x0::AbstractVector; twin=nothing, hessian=nothing) =
    CoverProblem(String(name), f, Vector{Float64}(x0), twin, hessian)

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
    exact_hessian(p::CoverProblem, x=p.x0) -> H

The analytic Hessian of `p` at `x`. Errors when `p` has none; test with
[`has_exact_hessian`](@ref) first.
"""
function exact_hessian(p::CoverProblem, x::AbstractVector=p.x0)
    has_exact_hessian(p) ||
        throw(ArgumentError("problem \"$(p.name)\" has no analytic Hessian; differentiate `p.f` instead"))
    return p.hessian(x)
end

# ---- program-building helpers ---------------------------------------------------------------
#
# `TVar` has no `zero`, and none can be defined: a tape node exists only once it is recorded.
# So every accumulation starts from a term rather than from an additive identity.

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
function quadratic(n::Integer; rank::Integer=n, form::Symbol=:entrywise, sigma::Real=0.0,
                   rng::AbstractRNG=MersenneTwister(1))
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
    return CoverProblem(name, f, x0; twin=x -> dot(x, A * x) / 2, hessian=_ -> A)
end

# ---- deep compositions -----------------------------------------------------------------------

"""
    deep_composition(n; depth=4, width=4, sigma=0.0, rng=MersenneTwister(1)) -> CoverProblem

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

No closed-form Hessian; differentiate `p.f`.
"""
function deep_composition(n::Integer; depth::Integer=4, width::Integer=4, sigma::Real=0.0,
                          rng::AbstractRNG=MersenneTwister(1))
    n >= 1 || throw(ArgumentError("n must be positive, got $n"))
    depth >= 1 || throw(ArgumentError("depth must be positive, got $depth"))
    width >= 1 || throw(ArgumentError("width must be positive, got $width"))
    W = _coeffs(rng, width, n; sigma) ./ sqrt(n)
    b = _coeffs(rng, width; sigma)
    a = _coeffs(rng, depth, width; sigma=0) ./ sqrt(width)
    c = _coeffs(rng, depth, width; sigma=0)
    x0 = randn(rng, n)

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

    return CoverProblem("deep d=$depth w=$width n=$n", f, x0; twin)
end

# ---- random DAGs -----------------------------------------------------------------------------

const _DAG_UNARY = (:scale, :tanh, :sqr, :sin)
const _DAG_BINARY = (:add, :sub, :mul)

# Probability that a node is unary rather than binary.
const _DAG_PUNARY = 0.4

# Node magnitudes are held inside [1/_DAG_BAND, _DAG_BAND] by a normalizing `:scale` node. Both
# ends matter: `:mul` and `:sqr` chains run away to overflow from above, and — the silent failure
# — collapse to an exactly zero derivative from below, since squaring a sub-unit value a dozen
# times underflows and leaves a program that is constant to second order. Normalizing is linear,
# so unlike a `tanh` squash it leaves the second-order structure of the program intact.
const _DAG_BAND = 1e3

# tanh must see a tame input: `1 - tanh(v)^2` evaluates to exactly zero in Float64 for
# |v| ≳ 19, and a derivative-dead node erases its whole subtree from the gradient and the
# Hessian — at sharing = 0, where the program's tail is a single unary chain, one saturated
# tanh zeroes every derivative of the program. The value band above is far too wide to
# prevent this, so tanh inputs beyond this band are first normalized to magnitude 1, where
# tanh has healthy first and second derivatives.
const _DAG_TANH_BAND = 2.0

"""
A random program, stored as the recipe that generated it: `ops[k]` builds node `ninputs + k` from
earlier nodes, and the output is the sum of `outputs`. Replaying the recipe on a vector of any
numeric type evaluates the same function, so one recipe serves the tape, the reference
derivatives, and the plain evaluation.
"""
struct DagRecipe
    ninputs::Int
    ops::Vector{Tuple{Symbol,Vector{Int},Vector{Float64}}}
    outputs::Vector{Int}
end

_apply(op::Symbol, v, ps::Vector{Int}, cs::Vector{Float64}) =
    op === :add   ? v[ps[1]] + v[ps[2]] :
    op === :sub   ? v[ps[1]] - v[ps[2]] :
    op === :mul   ? v[ps[1]] * v[ps[2]] :
    op === :scale ? cs[1] * v[ps[1]] :
    op === :tanh  ? tanh(v[ps[1]]) :
    op === :sqr   ? v[ps[1]]^2 :
    op === :sin   ? sin(v[ps[1]]) :
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
function random_dag(n::Integer; nops::Integer=4n, sharing::Real=0.5, sigma::Real=0.0,
                    rng::AbstractRNG=MersenneTwister(1))
    n >= 1 || throw(ArgumentError("n must be positive, got $n"))
    nops >= 1 || throw(ArgumentError("nops must be positive, got $nops"))
    0 <= sharing <= 1 || throw(ArgumentError("sharing must lie in 0..1, got $sharing"))
    x0 = randn(rng, n)

    vals = copy(x0)                             # values at x0, used only to keep magnitudes tame
    ops = Tuple{Symbol,Vector{Int},Vector{Float64}}[]
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
        # A tree whose unconsumed nodes have dwindled to one can only keep growing as a
        # single unary chain, and derivative factors compound multiplicatively along a
        # chain — hundreds of them wander into under- or overflow and erase (or explode)
        # every derivative of the program. Stop instead, once the program carries some
        # curvature; with reuse (`sharing > 0`) the frontier repopulates and no chain forms.
        sharing == 0 && length(frontier) == 1 && ncurved > 0 && break
        # Without reuse a binary op needs two unconsumed nodes; when they run out, go unary
        # rather than break the tree property that `sharing = 0` promises.
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
        # An entirely linear draw has an identically zero Hessian and poses no covering
        # problem at all; square one unconsumed node so every generated program has curvature.
        p = _take_parent!(rng, frontier, length(vals), 0.0)
        finish!(emit!(:sqr, [p], Float64[], vals[p]^2))
    end

    recipe = DagRecipe(n, ops, sort!(frontier))
    name = "dag n=$n share=$(round(sharing; digits=2))"
    return CoverProblem(name, recipe, x0)
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
    f, tw, hs = p.f, p.twin, p.hessian
    D = Diagonal(inv.(dv))
    return CoverProblem(p.name * " rescaled",
                        y -> f(y ./ dv),
                        dv .* p.x0,
                        tw === nothing ? nothing : y -> tw(y ./ dv),
                        hs === nothing ? nothing : y -> D * hs(y ./ dv) * D)
end

"""
    families(; sigma=0.0, seed=1) -> Vector{Pair{String,Any}}

The standard sweep: named generators `n -> CoverProblem`, one per family, differing in how
severely the paths through the program reconverge. Pass the same `n` to each to compare families
at fixed problem size, or sweep `n` within one to measure how a cover's looseness grows.

The three quadratic forms are seeded alike, so at a given `n` they carry the *same* Hessian and
differ only as programs.
"""
function families(; sigma::Real=0.0, seed::Integer=1)
    return [
        "quad entrywise" => n -> quadratic(n; form=:entrywise, sigma, rng=MersenneTwister(seed)),
        "quad matvec" => n -> quadratic(n; form=:matvec, sigma, rng=MersenneTwister(seed)),
        "quad factored" => n -> quadratic(n; form=:factored, sigma, rng=MersenneTwister(seed)),
        "deep w=4" => n -> deep_composition(n; depth=4, width=4, sigma, rng=MersenneTwister(seed)),
        "dag tree" => n -> random_dag(n; sharing=0.0, sigma, rng=MersenneTwister(seed)),
        "dag shared" => n -> random_dag(n; sharing=0.6, sigma, rng=MersenneTwister(seed)),
    ]
end

end # module
