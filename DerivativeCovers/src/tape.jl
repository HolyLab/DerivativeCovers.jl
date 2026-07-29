# Wengert tape and the overloaded scalar variable that records onto it.

"""
    PairedPartials(q::Int) <: AbstractMatrix{Float64}

Second-partial matrix of an inner-product node with `q` pairs of parent slots ordered
`(w_1, ..., w_q, x_1, ..., x_q)`. Entry `(i, j)` is `1` when `i` and `j` are paired
(`abs(i - j) == q`), the sole nonzero second partials `∂²y/∂w_j∂x_j` of `y = sum_j w_j*x_j`, and
`0` otherwise. Avoids materializing the dense `2q × 2q` matrix, almost all of whose entries are
zero.
"""
struct PairedPartials <: AbstractMatrix{Float64}
    q::Int
end

Base.size(P::PairedPartials) = (2P.q, 2P.q)
Base.getindex(P::PairedPartials, i::Int, j::Int) = abs(i - j) == P.q ? 1.0 : 0.0

"""
    CoveredPartials(s::Vector{Float64})

Declared bound on the second partials of a sub-map output node (see [`submapnode`](@ref)):
`abs(C[i, j]) <= s[i]*s[j]`, with the true `C` not recorded. `s` is nonnegative and slot-aligned
with the node's parents. Cover constructions consume second partials only through bounds on
`abs.(C)`, so the declaration substitutes for the recorded matrix everywhere except
[`hvp`](@ref), which needs the signed entries and fails fast. Deliberately not an
`AbstractMatrix`: it has no entries to index, only a bound.
"""
struct CoveredPartials
    s::Vector{Float64}
end

"""
    Tape

A Wengert list recorded while a scalar program runs on [`TVar`](@ref) inputs.

Node `k` stores its value `vals[k]`, its parent indices `parents[k]`, the first partials
`ds[k][i] = ∂φ_k/∂v_{p_i}`, and the second partials `Cs[k]`, a symmetric `∂²φ_k/∂v_{p_i}∂v_{p_j}`
in one of three forms:

  - a dense `Matrix{Float64}`, for unary and binary elementals;
  - a [`PairedPartials`](@ref), for an inner-product node (see [`dotnode`](@ref)): parent slots
    are ordered `(w_1, ..., w_q, x_1, ..., x_q)` and the only nonzero entries are the cross terms
    `∂²y/∂w_j∂x_j = 1`;
  - an empty `Matrix{Float64}(undef, 0, 0)`, for a node whose first partials are constant in the
    inputs (the input nodes themselves, and linear n-ary nodes such as [`linearnode`](@ref)):
    it contributes no second-order term to any cover;
  - a [`CoveredPartials`](@ref), for a declared sub-map output (see [`submapnode`](@ref)): the
    true second partials are not recorded, only the declared rank-1 bound
    `abs(C[i, j]) <= s[i]*s[j]`.

The first `ninputs` nodes are the program's inputs; they carry no parents and empty partials, as
does a constant output node ([`constantnode`](@ref)). Nodes are appended as the program executes,
so they are in topological order: every parent index is smaller than the node's own index.
"""
struct Tape
    vals::Vector{Float64}
    parents::Vector{Vector{Int}}
    ds::Vector{Vector{Float64}}
    Cs::Vector{Union{Matrix{Float64}, PairedPartials, CoveredPartials}}
    ninputs::Int
end

"""
    Tape(x0::AbstractVector)

Create a tape seeded with the entries of `x0` as its input nodes.

# Examples

Recording by hand, the way [`tape_function`](@ref) does it:

```jldoctest
julia> t = Tape([0.5, 2.0]);

julia> x = [TVar(t, i) for i in 1:t.ninputs];

julia> y = exp(x[1]) + x[1] * x[2]^2;

julia> DerivativeCovers.nnodes(t), y.idx
(6, 6)
```
"""
function Tape(x0::AbstractVector)
    Base.require_one_based_indexing(x0)
    n = length(x0)
    return Tape(
        Float64[float(x) for x in x0],
        [Int[] for _ in 1:n],
        [Float64[] for _ in 1:n],
        Union{Matrix{Float64}, PairedPartials, CoveredPartials}[Matrix{Float64}(undef, 0, 0) for _ in 1:n],
        n
    )
end

"""
    nnodes(t::Tape)

Number of nodes recorded on `t`, inputs included. This is the tape length `L`.
"""
nnodes(t::Tape) = length(t.vals)

"""
    TVar(t::Tape, idx::Integer)
    TVar(c::Real)

A scalar in a program being recorded, in one of two states. A *recording* `TVar` refers to node
`idx` of the tape `t`, and every elemental applied to it appends a node. A *constant* `TVar`
([`isconstant`](@ref)) carries a value and no tape: it is what `eltype(x)(c)` and `zero(eltype(x))`
produce in generic numeric code, where a typed literal is asked for before any tape is in reach.

Constants are closed under the elementals — an operation on constants alone folds to a constant,
and an operation mixing the two records against the recording operand with the constant as a
coefficient — so a constant never reaches a node builder. `val` is the scalar both states carry;
for a recording `TVar` it equals `t.vals[idx]`, which nodes never mutate once pushed.

# Examples

```jldoctest
julia> t = Tape([0.5, 2.0]);

julia> x = TVar(t, 1);

julia> DerivativeCovers.isconstant(x), DerivativeCovers.value(x)
(false, 0.5)

julia> c = TVar(3.0);   # a constant: a value and no tape

julia> DerivativeCovers.isconstant(c * TVar(4.0)), DerivativeCovers.value(c * TVar(4.0))
(true, 12.0)
```
"""
struct TVar
    tape::Union{Tape, Nothing}
    idx::Int
    val::Float64
end

TVar(t::Tape, idx::Integer) = TVar(t, Int(idx), t.vals[idx])
TVar(c::Real) = TVar(nothing, 0, float(c))
TVar(x::TVar) = x

"""
    isconstant(x::TVar) -> Bool

Whether `x` is a constant: a value with no tape behind it, as returned by `TVar(::Real)` and by
`zero`/`one`/`oneunit` of the element type. A constant contributes coefficients to the nodes it
enters and has no node of its own.
"""
isconstant(x::TVar) = x.tape === nothing

"""
    value(x)

The numeric value carried by `x`: the recorded value of a [`TVar`](@ref), constant or not, or `x`
itself for a plain `Real`. The `Real` fallback lets generic code read the value of either a `TVar`
or a plain number without a branch, detaching a quantity from the tape.
"""
value(x::TVar) = x.val
value(x::Real) = x

"""
    noderef(x::TVar) -> (tape, idx)

The tape and node index behind `x`. A constant has neither: every elemental folds a constant
operand into one that records, so reaching here with a constant means a caller bypassed that
folding.
"""
function noderef(x::TVar)
    t = x.tape
    t === nothing &&
        throw(ArgumentError("a constant TVar (value $(value(x))) has no node on any tape; it must be folded into an operand that does"))
    return t, x.idx
end

"""
    pushnode!(t::Tape, val, parents, d, C) -> TVar

Append a node with value `val`, parent indices `parents`, first partials `d`, and second
partials `C`, and return a [`TVar`](@ref) referring to it.
"""
function pushnode!(t::Tape, val, parents, d, C)
    push!(t.vals, val)
    push!(t.parents, parents)
    push!(t.ds, d)
    push!(t.Cs, C)
    return TVar(t, nnodes(t))
end

"""
    constantnode(t::Tape, c::Real) -> TVar

Record a node with value `c`, no parents, and empty partials. Its absolute sensitivity is zero, so
its Jacobian row vanishes and it contributes no atom to any cover.

Elementals fold a constant operand into the node that consumes it rather than giving it a node
(see [`TVar`](@ref)), so this is reached only at the boundary where a program's outputs become
node indices and a constant needs an index to be one: a fixed offset among a program's outputs,
such as the unit residual that carries an additive constant of a least-squares loss.
"""
constantnode(t::Tape, c::Real) =
    pushnode!(t, float(c), Int[], Float64[], Matrix{Float64}(undef, 0, 0))

"""
    unarynode(x::TVar, val, d, c) -> TVar

Record a node `φ(x)` with value `val`, `φ'(x) = d`, and `φ''(x) = c`.
"""
function unarynode(x::TVar, val, d, c)
    t, i = noderef(x)
    return pushnode!(t, val, [i], [float(d)], fill(float(c), 1, 1))
end

"""
    binarynode(x::TVar, y::TVar, val, d1, d2, c11, c12, c22) -> TVar

Record a node `φ(x, y)` with value `val`, first partials `(d1, d2)`, and second partials
`[c11 c12; c12 c22]`.
"""
function binarynode(x::TVar, y::TVar, val, d1, d2, c11, c12, c22)
    tx, ix = noderef(x)
    ty, iy = noderef(y)
    tx === ty || throw(ArgumentError("cannot combine TVars from different tapes"))
    return pushnode!(
        tx, val, [ix, iy], [float(d1), float(d2)],
        Float64[c11 c12; c12 c22]
    )
end

"""
    dotnode(ws::AbstractVector{TVar}, xs::AbstractVector{TVar}) -> TVar

Record an inner-product node `y = sum_j w_j*x_j`. Parent slots are ordered
`(w_1, ..., w_q, x_1, ..., x_q)`, first partials are `(x_1, ..., x_q, w_1, ..., w_q)`, and the
second-partial matrix is [`PairedPartials`](@ref)`(q)`. Parent slots are independent even when
`ws` and `xs` share entries (e.g. `dotnode(xs, xs)`): the decomposition is over slots, not values.
"""
function dotnode(ws::AbstractVector{TVar}, xs::AbstractVector{TVar})
    isempty(ws) && throw(ArgumentError("cannot record an empty inner product"))
    idxs = eachindex(ws, xs)
    q = length(idxs)
    t, _ = noderef(ws[first(idxs)])
    val = 0.0
    parents = Vector{Int}(undef, 2q)
    ds = Vector{Float64}(undef, 2q)
    for (n, j) in enumerate(idxs)
        w, x = ws[j], xs[j]
        tw, iw = noderef(w)
        tx, ix = noderef(x)
        (tw === t && tx === t) ||
            throw(ArgumentError("cannot combine TVars from different tapes"))
        val += value(w) * value(x)
        parents[n] = iw
        parents[q + n] = ix
        ds[n] = value(x)
        ds[q + n] = value(w)
    end
    return pushnode!(t, val, parents, ds, PairedPartials(q))
end

"""
    linearnode(coeffs::AbstractVector{<:Real}, xs::AbstractVector{TVar}) -> TVar

Record a node `y = sum_j coeffs[j]*xs[j]` for constant `coeffs`. First partials are the
(floated) coefficients; the second-partial matrix is empty, since a linear combination of
constants and `TVar`s has no second-order term.
"""
function linearnode(coeffs::AbstractVector{<:Real}, xs::AbstractVector{TVar})
    isempty(xs) && throw(ArgumentError("cannot record an empty linear combination"))
    idxs = eachindex(coeffs, xs)
    t, _ = noderef(xs[first(idxs)])
    val = 0.0
    parents = Vector{Int}(undef, length(idxs))
    ds = Vector{Float64}(undef, length(idxs))
    for (n, j) in enumerate(idxs)
        tx, ix = noderef(xs[j])
        tx === t || throw(ArgumentError("cannot combine TVars from different tapes"))
        val += coeffs[j] * value(xs[j])
        parents[n] = ix
        ds[n] = float(coeffs[j])
    end
    return pushnode!(t, val, parents, ds, Matrix{Float64}(undef, 0, 0))
end

"""
    submapnode(xs::AbstractVector{TVar}, y::AbstractVector{<:Real}, J::AbstractMatrix{<:Real},
               S::Union{Nothing,AbstractMatrix{<:Real}}=nothing) -> Vector{TVar}

Record a declared sub-map call `y = F(v)` as one node per output, without retracing `F`'s
internals: node `k` carries the value `y[k]`, the arguments `xs` as parents, the exact signed
Jacobian row `J[k, :]` (`J[k, i] = ∂F_k/∂v_i` at the parent values) as its first partials, and
the declared curvature cover `S[k, :]` as a [`CoveredPartials`](@ref):
`abs(∂²F_k/∂v_i∂v_j) <= S[k, i]*S[k, j]`. `S = nothing`, or an all-zero row of `S`, declares
that output linear in the arguments (an empty second-partial matrix).

Gradients and every first-order sweep through the call are exact. Hessian covers are valid
conditional on the declaration, which is the caller's assertion — nothing on the tape can verify
it — and exact Hessian-vector products are unavailable, so [`hvp`](@ref) throws on any tape
containing a declared call. For the covers to stay scale-covariant the declaration must
transform as `y -> C*y`, `J -> C*J*inv(D)`, `S -> sqrt.(C)*S*inv(D)` under a rescaling
`v -> D*v` of the arguments and a rescaling `y -> C*y` of the sub-map outputs (a symmetric
cover scales as the square root of its output's scale); [`declare_submap`](@ref) builds
declarations with both properties. The covers are then invariant to the output gauge `C`, which
cancels between each output's adjoint and its declared curvature.
"""
function submapnode(
        xs::AbstractVector{TVar}, y::AbstractVector{<:Real},
        J::AbstractMatrix{<:Real},
        S::Union{Nothing, AbstractMatrix{<:Real}} = nothing
    )
    Base.require_one_based_indexing(xs, y, J)
    isempty(xs) && throw(ArgumentError("cannot record a sub-map call with no arguments"))
    isempty(y) && throw(ArgumentError("cannot record a sub-map call with no outputs"))
    q, p = length(y), length(xs)
    size(J) == (q, p) ||
        throw(DimensionMismatch("J is $(size(J)); expected ($q, $p) (outputs × arguments)"))
    if S !== nothing
        Base.require_one_based_indexing(S)
        size(S) == (q, p) ||
            throw(DimensionMismatch("S is $(size(S)); expected ($q, $p) (outputs × arguments)"))
        all(>=(0), S) || throw(ArgumentError("curvature covers S must be nonnegative"))
    end
    t, _ = noderef(xs[1])
    all(x -> first(noderef(x)) === t, xs) ||
        throw(ArgumentError("cannot combine TVars from different tapes"))
    parents = [last(noderef(x)) for x in xs]   # shared by all q nodes; parent lists are never mutated
    return [
        begin
                C = if S === nothing || all(iszero, view(S, k, :))
                    Matrix{Float64}(undef, 0, 0)
            else
                    CoveredPartials([float(S[k, i]) for i in 1:p])
            end
                pushnode!(t, float(y[k]), parents, [float(J[k, i]) for i in 1:p], C)
            end for k in 1:q
    ]
end

"""
    tape_function(f, x0::AbstractVector) -> (tape, outidx)      # scalar output
    tape_function(f, x0::AbstractVector) -> (tape, outidxs)     # vector output

Record the program `f` evaluated at `x0`. `f` must accept a vector of [`TVar`](@ref)s and use only
the elementals overloaded for `TVar`. Returns the [`Tape`](@ref) and, depending on what `f`
returns:

  - a single `TVar` output: the `Int` index `outidx` of the output node;
  - an `AbstractVector` of `TVar`s: the `Vector{Int}` of output-node indices `outidxs`, one per
    entry. Entries may be input nodes or repeat a node (a duplicated output), and both are kept as
    given — the indices, not the values, define the outputs. A constant entry gets a node of its
    own ([`constantnode`](@ref)) and so a zero Jacobian row; a program whose outputs are *all*
    constant depends on none of its inputs and is an error.

# Examples

```jldoctest
julia> f(x) = exp(x[1]) + x[1] * x[2]^2;

julia> t, outidx = tape_function(f, [0.5, 2.0]);

julia> t.ninputs, outidx, t.vals[outidx]
(2, 6, 3.648721270700128)

julia> r(x) = [x[1] * x[2], exp(x[1]) - x[3], x[2] + 2x[3]];

julia> tr, outidxs = tape_function(r, [0.5, 2.0, -1.0]);

julia> outidxs
3-element Vector{Int64}:
 4
 6
 8
```
"""
function tape_function(f, x0::AbstractVector)
    t = Tape(x0)
    xs = [TVar(t, i) for i in 1:t.ninputs]
    return _tape_output(t, f(xs))
end

function _tape_output(t::Tape, out::TVar)
    isconstant(out) &&
        throw(ArgumentError("program output is a constant (value $(value(out))); it does not depend on the inputs"))
    out.tape === t ||
        throw(ArgumentError("program output was recorded on a different tape than its inputs"))
    return t, out.idx
end

function _tape_output(t::Tape, out::AbstractVector)
    isempty(out) && throw(ArgumentError("program returned no outputs"))
    all(v -> v isa TVar, out) ||
        throw(ArgumentError("program output vector must contain only TVars; got element types $(unique(map(typeof, out)))"))
    all(isconstant, out) &&
        throw(ArgumentError("every program output is a constant; the program does not depend on its inputs"))
    all(v -> isconstant(v) || v.tape === t, out) ||
        throw(ArgumentError("some program outputs were recorded on a different tape than the inputs"))
    return t, [isconstant(v) ? constantnode(t, value(v)).idx : v.idx for v in out]
end

_tape_output(t::Tape, out) =
    throw(ArgumentError("program output is not a TVar or a vector of TVars (got a $(typeof(out))); it must depend on the inputs"))
