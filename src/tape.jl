# Wengert tape and the overloaded scalar variable that records onto it.

"""
    Tape

A Wengert list recorded while a scalar program runs on [`TVar`](@ref) inputs.

Node `k` stores its value `vals[k]`, its parent indices `parents[k]`, the first partials
`ds[k][i] = ∂φ_k/∂v_{p_i}`, and the symmetric matrix `Cs[k]` of second partials
`∂²φ_k/∂v_{p_i}∂v_{p_j}`. The first `ninputs` nodes are the program's inputs; they carry no
parents and empty partials. Nodes are appended as the program executes, so they are in
topological order: every parent index is smaller than the node's own index.
"""
struct Tape
    vals::Vector{Float64}
    parents::Vector{Vector{Int}}
    ds::Vector{Vector{Float64}}
    Cs::Vector{Matrix{Float64}}
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
                [Matrix{Float64}(undef, 0, 0) for _ in 1:n],
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

value(x::TVar) = x.tape.vals[x.idx]

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
    tape_function(f, x0::AbstractVector) -> (tape, outidx)

Record the scalar program `f` evaluated at `x0`. `f` must accept a vector of [`TVar`](@ref)s
and return a single `TVar`, using only the elementals overloaded for `TVar`. Returns the
[`Tape`](@ref) and the index of the output node.
"""
function tape_function(f, x0::AbstractVector)
    t = Tape(x0)
    xs = [TVar(t, i) for i in 1:t.ninputs]
    out = f(xs)
    out isa TVar ||
        throw(ArgumentError("program output is not a TVar (got a $(typeof(out))); it must depend on the inputs"))
    out.tape === t ||
        throw(ArgumentError("program output was recorded on a different tape than its inputs"))
    return t, out.idx
end
