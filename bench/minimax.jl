# The rank-1 cover of a nonnegative matrix whose largest excess is smallest.

"""
    jacobian_minimax(M; atol=eps(float(eltype(M)))^(3//4)) -> (; a, b, excess)

The exact minimax rank-1 cover of the nonnegative matrix `M`: a pair `a_k * b_j ≥ M_kj` whose
worst-case excess `max_kj (a_k*b_j)/M_kj` over the nonzero entries of `M` is the smallest any
rank-1 cover can achieve. The optimal value equals the maximum over bipartite cycles of the
support — rows `k_1, …, k_r` and columns `j_1, …, j_r`, all entries `(k_i, j_i)` and
`(k_i, j_{i+1})` nonzero — of the alternating geometric mean

    [ (M_{k_1 j_1} ⋯ M_{k_r j_r}) / (M_{k_1 j_2} ⋯ M_{k_r j_1}) ]^{1/r} .

Every unit cancels around a cycle (each row and column appears once in the numerator and once in
the denominator), so `excess` is dimensionless and the pair is equivariant under
`M → diag(c)*M*diag(d)` modulo the gauge `(a, b) → (γa, γ⁻¹b)`. The optimal pair is generally
not unique off the critical cycles.

Computed by bisection on the log-space excess with Bellman–Ford feasibility (difference
constraints `log M_kj ≤ log a_k + log b_j ≤ log M_kj + E`), `O(nnz·(m+n))` per probe. `atol` is
the absolute log-space bisection width: `excess` is exact to a relative `atol` and the returned
pair attains it. A zero row of `M` gets `a_k = 0`, a zero column `b_j = 0`.
"""
function jacobian_minimax(M::AbstractMatrix; atol::Real = eps(float(eltype(M)))^(3 // 4))
    Base.require_one_based_indexing(M)
    all(>=(0), M) || throw(ArgumentError("M must be nonnegative"))
    atol > 0 || throw(ArgumentError("atol must be positive, got $atol"))
    m, n = size(M)
    T = float(eltype(M))
    ks, js, cs = Int[], Int[], T[]
    for j in 1:n, k in 1:m
        M[k, j] > 0 && (push!(ks, k); push!(js, j); push!(cs, log(M[k, j])))
    end
    isempty(cs) && return (; a = zeros(T, m), b = zeros(T, n), excess = one(T))

    lo, hi = zero(T), maximum(cs) - minimum(cs)
    feasible, pot = _minimax_feasible(m, n, ks, js, cs, hi)
    feasible || error("Bellman–Ford infeasible at the row-span upper bound; this cannot happen")
    while hi - lo > atol
        mid = lo + (hi - lo) / 2
        f, p = _minimax_feasible(m, n, ks, js, cs, mid)
        f ? (hi = mid; pot = p) : (lo = mid)
    end

    a, b = exp.(pot[1:m]), exp.(.-pot[(m + 1):(m + n)])
    liverow, livecol = falses(m), falses(n)
    for (k, j) in zip(ks, js)
        liverow[k] = livecol[j] = true
    end
    a[.!liverow] .= 0
    b[.!livecol] .= 0
    excess = maximum(a[k] * b[j] / M[k, j] for (k, j) in zip(ks, js))
    excess >= 1 - eps(T)^(2 // 3) ||
        error("minimax pair does not cover M: worst ratio $(1 / excess)")
    return (; a, b, excess)
end

# The minimax constraints are feasible iff their difference-constraint graph has no negative
# cycle. Multi-source Bellman–Ford both tests feasibility and returns valid potentials.
function _minimax_feasible(m, n, ks, js, cs, E)
    pot = zeros(eltype(cs), m + n)
    for _ in 1:(m + n)
        changed = false
        for (k, j, c) in zip(ks, js, cs)
            if pot[k] - c < pot[m + j]
                pot[m + j] = pot[k] - c
                changed = true
            end
            if pot[m + j] + c + E < pot[k]
                pot[k] = pot[m + j] + c + E
                changed = true
            end
        end
        changed || return true, pot
    end
    return false, pot
end
