# Wengert tape and the overloaded scalar variable that records onto it.

"""
    Tape{T}

A Wengert tape recorded while a function runs on [`TVar`](@ref) inputs, holding its values and
partials in the element type `T`. `eltype(t)` reports `T`, and every cover computed from `t`
returns it: `T` is the working precision of the whole construction, and a value or weight a caller
supplies is converted into it.

The first `ninputs` nodes represent the function inputs. Later nodes are appended in topological
order, so every parent precedes its child. Node `k` stores its value `vals[k]`, parent indices
`parents[k]`, first partials `ds[k][i] = ∂φ_k/∂v_{p_i}`, and the second partials `Cs[k]`, a
[`Curvature`](@ref) holding the symmetric `∂²φ_k/∂v_{p_i}∂v_{p_j}` in whatever form the elemental
supplies: [`UnaryCurvature`](@ref) or [`BinaryCurvature`](@ref) for a unary or binary elemental,
[`PairedCurvature`](@ref) for an inner-product node (see [`dotnode`](@ref)),
[`DeclaredCurvature`](@ref) for a declared sub-map output (see [`submapnode`](@ref)), and
[`NoCurvature`](@ref) for a node with no second-order term.

Input nodes and constant output nodes have no parents and no second partials.
"""
struct Tape{T}
    vals::Vector{T}
    parents::Vector{Vector{Int}}
    ds::Vector{Vector{T}}
    Cs::Vector{Curvature}
    ninputs::Int
end

Base.eltype(::Type{Tape{T}}) where {T} = T

"""
    Tape(x0::AbstractVector)
    Tape{T}(x0::AbstractVector)

Create a tape seeded with the entries of `x0` as its input nodes. The element type defaults to
`float(eltype(x0))`, so an integer starting point records in `Float64`; give `T` explicitly to
record in another precision.

# Examples

Recording by hand, the way [`tape_function`](@ref) does it:

```jldoctest
julia> t = Tape([0.5, 2.0]);

julia> x = [TVar(t, i) for i in 1:t.ninputs];

julia> y = exp(x[1]) + x[1] * x[2]^2;

julia> DerivativeCovers.nnodes(t), y.idx
(6, 6)

julia> eltype(Tape{Float32}([0.5, 2.0]))
Float32
```
"""
Tape(x0::AbstractVector) = Tape{float(eltype(x0))}(x0)

function Tape{T}(x0::AbstractVector) where {T}
    Base.require_one_based_indexing(x0)
    n = length(x0)
    return Tape{T}(
        T[x for x in x0],
        [Int[] for _ in 1:n],
        [T[] for _ in 1:n],
        Curvature[NoCurvature() for _ in 1:n],
        n
    )
end

"""
    nnodes(t::Tape)

Number of nodes recorded on `t`, inputs included. This is the tape length `L`.
"""
nnodes(t::Tape) = length(t.vals)

"""
    natoms(t::Tape)

Number of cover atoms recorded on `t`: the independent rank-1 second-order terms its nodes expose,
summed over the tape (see [`atom_slots`](@ref)). A unary or binary elemental with nonzero second
partials exposes one atom, an inner-product node one per parent pair, and a linear node none.

This is the group count at which [`cover_hybrid`](@ref) reaches the quadrature merge. It counts
what the tape records, so a node whose adjoint or curvature vanishes at the recorded point is
counted even though it seeds no channel.
"""
function natoms(t::Tape)
    total = 0
    for k in 1:nnodes(t), _ in atom_slots(t.Cs[k], length(t.parents[k]))
        total += 1
    end
    return total
end

"""
    TapeSite(t::Tape, k::Integer)

The [`CoverSite`](@ref) naming node `k` of `t`, so that a cover built there reports which node it
failed on and what the values around it were.
"""
struct TapeSite{T} <: CoverSite
    t::Tape{T}
    k::Int
end

Base.eltype(::Type{TapeSite{T}}) where {T} = T
site_label(s::TapeSite) = "node $(s.k)"
site_values(s::TapeSite) =
    "node value $(s.t.vals[s.k]), parent values $(Tuple(s.t.vals[p] for p in s.t.parents[s.k]))"

"""
    CoverVar{T}

A scalar that carries a value of type `T` through a program so that a cover construction can
observe the program's elementals as it runs. [`TVar`](@ref) records them onto a [`Tape`](@ref).

The elementals of `overloads.jl` are written against this type rather than against any one
backend. Each computes its value, first partials and second partials and hands them to one of four
*recorders*, which is where the backends differ. A concrete subtype `V` defines:

  - [`value`](@ref) and [`isconstant`](@ref);
  - [`constantvar`](@ref)`(::Type{V}, c)`. Generic numerical code obtains typed literals through
    `zero`, `one`, `oneunit` and `convert` of `V`, all of which route here, so a constant must
    carry every type parameter `V` has;
  - whichever of [`unarynode`](@ref), [`binarynode`](@ref), [`dotnode`](@ref) and
    [`linearnode`](@ref) it supports. An unsupported recorder makes the elementals that reach it
    unavailable; a subtype supporting none of them cannot run a program.

A constant has no node behind it in any backend: an elemental folds a constant operand into the
node that consumes it.
"""
abstract type CoverVar{T} end

"""
    constantvar(::Type{V}, c::Real) -> V

The constant of backend `V` carrying value `c` and nothing else. See [`CoverVar`](@ref).
"""
function constantvar end

"""
    TVar(t::Tape{T}, idx::Integer)
    TVar(c::Real)
    TVar{T}(c::Real)

A scalar used while recording a function, carrying a value of type `T`. The [`CoverVar`](@ref)
backend that records onto a [`Tape`](@ref).

`TVar(t, idx)` refers to node `idx` on `t` and takes the tape's element type. `TVar(c)` creates a
constant with a value but no tape; this supports typed literals such as `zero(eltype(x))` in
generic numerical code, which reach `TVar{T}(c)` and so stay in the program's precision.
Operations involving only constants are folded immediately. Operations that mix a constant and
a recorded variable store the constant as a coefficient.

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
struct TVar{T} <: CoverVar{T}
    tape::Union{Tape{T}, Nothing}
    idx::Int
    val::T
end

TVar(t::Tape{T}, idx::Integer) where {T} = TVar{T}(t, Int(idx), t.vals[idx])
TVar{T}(c::Real) where {T} = TVar{T}(nothing, 0, c)
TVar(c::Real) = TVar{typeof(float(c))}(float(c))
TVar(x::TVar) = x

constantvar(::Type{TVar{T}}, c::Real) where {T} = TVar{T}(c)
constantvar(::Type{TVar}, c::Real) = TVar(c)

"""
    isconstant(x) -> Bool

Whether `x` is a constant: a value with no node behind it, as returned by
[`constantvar`](@ref) and by `zero`/`one`/`oneunit` of the element type. A constant contributes
coefficients to the nodes it enters and has no node of its own.
"""
isconstant(x::TVar) = x.tape === nothing

"""
    value(x)

The numeric value carried by `x`: the value of a [`CoverVar`](@ref), constant or not, or `x`
itself for a plain `Real`. The `Real` fallback lets generic code read the value of either a
program variable or a plain number without a branch, detaching a quantity from the backend.
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

`val` and `d` are converted to the tape's element type. `C` is stored as it is given, and every
[`Curvature`](@ref) form the sweeps resolve statically holds that same element type, so an
elemental should build its curvature in `eltype(t)` too.
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
constantnode(t::Tape{T}, c::Real) where {T} =
    pushnode!(t, c, Int[], T[], NoCurvature())

"""
    unarynode(x::TVar, val, d, c) -> TVar

Record a node `φ(x)` with value `val`, `φ'(x) = d`, and `φ''(x) = c`.

A zero `c` is stored as [`NoCurvature`](@ref) rather than as a zero [`UnaryCurvature`](@ref): the
node contributes no second-order term either way, and that form says so without an entry to load
and test.
"""
function unarynode(x::TVar, val, d, c)
    t, i = noderef(x)
    T = eltype(t)
    C = iszero(c) ? NoCurvature() : UnaryCurvature(convert(T, c))
    return pushnode!(t, val, [i], T[d], C)
end

"""
    binarynode(x::TVar, y::TVar, val, d1, d2, c11, c12, c22) -> TVar

Record a node `φ(x, y)` with value `val`, first partials `(d1, d2)`, and second partials
`[c11 c12; c12 c22]`.

An all-zero block is stored as [`NoCurvature`](@ref), for the reason [`unarynode`](@ref) stores a
zero `c` that way. Addition and subtraction of two tape variables are the common case.
"""
function binarynode(x::TVar, y::TVar, val, d1, d2, c11, c12, c22)
    tx, ix = noderef(x)
    ty, iy = noderef(y)
    tx === ty || throw(ArgumentError("cannot combine TVars from different tapes"))
    T = eltype(tx)
    C = if iszero(c11) && iszero(c12) && iszero(c22)
        NoCurvature()
    else
        BinaryCurvature(convert(T, c11), convert(T, c12), convert(T, c22))
    end
    return pushnode!(tx, val, [ix, iy], T[d1, d2], C)
end

"""
    dotnode(ws::AbstractVector{<:TVar}, xs::AbstractVector{<:TVar}) -> TVar

Record an inner-product node `y = sum_j w_j*x_j`. Parent slots are ordered
`(w_1, ..., w_q, x_1, ..., x_q)`, first partials are `(x_1, ..., x_q, w_1, ..., w_q)`, and the
second-partial matrix is [`PairedCurvature`](@ref)`(q)`. Parent slots are independent even when
`ws` and `xs` share entries (e.g. `dotnode(xs, xs)`): the decomposition is over slots, not values.
"""
function dotnode(ws::AbstractVector{<:TVar}, xs::AbstractVector{<:TVar})
    isempty(ws) && throw(ArgumentError("cannot record an empty inner product"))
    idxs = eachindex(ws, xs)
    q = length(idxs)
    t, _ = noderef(ws[first(idxs)])
    T = eltype(t)
    val = zero(T)
    parents = Vector{Int}(undef, 2q)
    ds = Vector{T}(undef, 2q)
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
    return pushnode!(t, val, parents, ds, PairedCurvature(q))
end

"""
    linearnode(coeffs::AbstractVector{<:Real}, xs::AbstractVector{<:TVar}) -> TVar

Record a node `y = sum_j coeffs[j]*xs[j]` for constant `coeffs`. First partials are the
coefficients converted to the tape's element type; the second-partial matrix is empty, since a
linear combination of constants and `TVar`s has no second-order term.
"""
function linearnode(coeffs::AbstractVector{<:Real}, xs::AbstractVector{<:TVar})
    isempty(xs) && throw(ArgumentError("cannot record an empty linear combination"))
    idxs = eachindex(coeffs, xs)
    t, _ = noderef(xs[first(idxs)])
    T = eltype(t)
    val = zero(T)
    parents = Vector{Int}(undef, length(idxs))
    ds = Vector{T}(undef, length(idxs))
    for (n, j) in enumerate(idxs)
        tx, ix = noderef(xs[j])
        tx === t || throw(ArgumentError("cannot combine TVars from different tapes"))
        val += coeffs[j] * value(xs[j])
        parents[n] = ix
        ds[n] = coeffs[j]
    end
    return pushnode!(t, val, parents, ds, NoCurvature())
end

"""
    submapnode(xs::AbstractVector{<:TVar}, y::AbstractVector{<:Real}, J::AbstractMatrix{<:Real},
               S::Union{Nothing,AbstractMatrix{<:Real}}=nothing) -> Vector{TVar}

Record a declared sub-map call `y = F(v)` as one node per output, without retracing `F`'s
internals: node `k` carries the value `y[k]`, the arguments `xs` as parents, the exact signed
Jacobian row `J[k, :]` (`J[k, i] = ∂F_k/∂v_i` at the parent values) as its first partials, and
the declared curvature cover `S[k, :]` as a [`DeclaredCurvature`](@ref):
`abs(∂²F_k/∂v_i∂v_j) <= S[k, i]*S[k, j]`. `S = nothing`, or an all-zero row of `S`, declares
that output linear in the arguments ([`NoCurvature`](@ref)).

Gradients and every first-order sweep through the call are exact. Hessian covers are valid
conditional on the declaration, which is the caller's assertion — nothing on the tape can verify
it — and exact Hessian-vector products are unavailable, so [`hvp`](@ref) throws on any tape
containing a declared call. For the covers to stay scale-covariant the declaration must
transform as `y -> C*y`, `J -> C*J*inv(D)`, `S -> sqrt.(C)*S*inv(D)` under a rescaling
`v -> D*v` of the arguments and a rescaling `y -> C*y` of the sub-map outputs (a symmetric
cover scales as the square root of its output's scale); [`declare_submap`](@ref) builds
declarations with both properties. The covers are then invariant to the output gauge `C`, which
cancels between each output's adjoint and its declared curvature.

Both matrices are dense and are held per node, so a declared call costs `O(q*p)` memory and
[`declare_submap`](@ref) one adjoint sweep per output to build: diagnostic tier. Where neither is
affordable, [`compose_jacobian_cover`](@ref) and [`compose_hessian_cover`](@ref) combine covers of
the two factors with no joint tape and no dense `J`. That route bounds the composition's
first-order contribution by a rank-1 pair rather than using `J` itself, which loosens the
`J'∇²g J` term of the chain rule and leaves the declared curvature term unchanged.
"""
function submapnode(
        xs::AbstractVector{<:TVar}, y::AbstractVector{<:Real},
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
    T = eltype(t)
    parents = [last(noderef(x)) for x in xs]   # shared by all q nodes; parent lists are never mutated
    return [
        begin
                C = if S === nothing || all(iszero, view(S, k, :))
                    NoCurvature()
            else
                    DeclaredCurvature(T[S[k, i] for i in 1:p])
            end
                pushnode!(t, y[k], parents, T[J[k, i] for i in 1:p], C)
            end for k in 1:q
    ]
end

"""
    tape_function(f, x0::AbstractVector) -> (tape, outidx)      # scalar output
    tape_function(f, x0::AbstractVector) -> (tape, outidxs)     # vector output

Record `f` at `x0` and return its [`Tape`](@ref) and output node index or indices.

`f` must accept a vector of [`TVar`](@ref)s and use operations supported by `TVar`. The second
return value is:

  - a single `TVar` output: the `Int` index `outidx` of the output node;
  - an `AbstractVector` of `TVar`s: the `Vector{Int}` of output-node indices `outidxs`, one per
    entry. Input nodes and repeated outputs are preserved. A constant output receives its own
    node and therefore contributes a zero Jacobian row.

A function whose outputs are all constant raises an error.

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
