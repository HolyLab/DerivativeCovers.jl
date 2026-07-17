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
    it contributes no second-order term to any cover.

The first `ninputs` nodes are the program's inputs; they carry no parents and empty partials.
Nodes are appended as the program executes, so they are in topological order: every parent index
is smaller than the node's own index.
"""
struct Tape
    vals::Vector{Float64}
    parents::Vector{Vector{Int}}
    ds::Vector{Vector{Float64}}
    Cs::Vector{Union{Matrix{Float64},PairedPartials}}
    ninputs::Int
end

"""
    Tape(x0::AbstractVector)

Create a tape seeded with the entries of `x0` as its input nodes.
"""
function Tape(x0::AbstractVector)
    Base.require_one_based_indexing(x0)
    n = length(x0)
    return Tape(Float64[float(x) for x in x0],
                [Int[] for _ in 1:n],
                [Float64[] for _ in 1:n],
                Union{Matrix{Float64},PairedPartials}[Matrix{Float64}(undef, 0, 0) for _ in 1:n],
                n)
end

"""
    nnodes(t::Tape)

Number of nodes recorded on `t`, inputs included. This is the tape length `L`.
"""
nnodes(t::Tape) = length(t.vals)

"""
    TVar

A scalar variable that records each elementary operation applied to it onto a [`Tape`](@ref).
`idx` is the index of the node holding this variable's value.
"""
struct TVar
    tape::Tape
    idx::Int
end

"""
    value(x)

The numeric value carried by `x`: `x.tape.vals[x.idx]` for a [`TVar`](@ref), or `x` itself for
a plain `Real`. The `Real` fallback lets generic code read the value of either a `TVar` or a
plain number without a branch, detaching a quantity from the tape.
"""
value(x::TVar) = x.tape.vals[x.idx]
value(x::Real) = x

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
    unarynode(x::TVar, val, d, c) -> TVar

Record a node `φ(x)` with value `val`, `φ'(x) = d`, and `φ''(x) = c`.
"""
unarynode(x::TVar, val, d, c) = pushnode!(x.tape, val, [x.idx], [float(d)], fill(float(c), 1, 1))

"""
    binarynode(x::TVar, y::TVar, val, d1, d2, c11, c12, c22) -> TVar

Record a node `φ(x, y)` with value `val`, first partials `(d1, d2)`, and second partials
`[c11 c12; c12 c22]`.
"""
function binarynode(x::TVar, y::TVar, val, d1, d2, c11, c12, c22)
    x.tape === y.tape || throw(ArgumentError("cannot combine TVars from different tapes"))
    return pushnode!(x.tape, val, [x.idx, y.idx], [float(d1), float(d2)],
                     Float64[c11 c12; c12 c22])
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
    t = ws[first(idxs)].tape
    val = 0.0
    parents = Vector{Int}(undef, 2q)
    ds = Vector{Float64}(undef, 2q)
    for (n, j) in enumerate(idxs)
        w, x = ws[j], xs[j]
        (w.tape === t && x.tape === t) ||
            throw(ArgumentError("cannot combine TVars from different tapes"))
        val += value(w) * value(x)
        parents[n] = w.idx
        parents[q + n] = x.idx
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
    isempty(xs) && throw(ArgumentError("cannot record an empty inner product"))
    idxs = eachindex(coeffs, xs)
    t = xs[first(idxs)].tape
    val = 0.0
    parents = Vector{Int}(undef, length(idxs))
    ds = Vector{Float64}(undef, length(idxs))
    for (n, j) in enumerate(idxs)
        x = xs[j]
        x.tape === t || throw(ArgumentError("cannot combine TVars from different tapes"))
        val += coeffs[j] * value(x)
        parents[n] = x.idx
        ds[n] = float(coeffs[j])
    end
    return pushnode!(t, val, parents, ds, Matrix{Float64}(undef, 0, 0))
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
    given — the indices, not the values, define the outputs.
"""
function tape_function(f, x0::AbstractVector)
    t = Tape(x0)
    xs = [TVar(t, i) for i in 1:t.ninputs]
    return _tape_output(t, f(xs))
end

function _tape_output(t::Tape, out::TVar)
    out.tape === t ||
        throw(ArgumentError("program output was recorded on a different tape than its inputs"))
    return t, out.idx
end

function _tape_output(t::Tape, out::AbstractVector)
    isempty(out) && throw(ArgumentError("program returned no outputs"))
    all(v -> v isa TVar, out) ||
        throw(ArgumentError("program output vector must contain only TVars; got element types $(unique(map(typeof, out)))"))
    all(v -> v.tape === t, out) ||
        throw(ArgumentError("some program outputs were recorded on a different tape than the inputs"))
    return t, [v.idx for v in out]
end

_tape_output(t::Tape, out) =
    throw(ArgumentError("program output is not a TVar or a vector of TVars (got a $(typeof(out))); it must depend on the inputs"))
