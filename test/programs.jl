# Scalar test programs, written generically so the same source runs on `TVar` inputs (recording a
# tape) and on `ForwardDiff.Dual` inputs (supplying reference gradients and Hessians).
#
# Each is built only from the elementals CoverAD overloads. Accumulators are seeded from the
# inputs (`0.0 * x[1]`, or a `first` flag) rather than from a literal zero, so that every value
# that reaches the output is a tape node.

using Random

function rosenbrock(x)
    local acc
    first = true
    for i in 1:(length(x) - 1)
        term = 100 * (x[i + 1] - x[i]^2)^2 + (1 - x[i])^2
        acc = first ? term : acc + term
        first = false
    end
    return acc
end

"f = x'Ax/2, written as a double loop over the entries of A."
function quad_entrywise(A)
    return function (x)
        acc = 0.0 * x[1]
        first = true
        for i in eachindex(x), j in eachindex(x)
            term = A[i, j] * x[i] * x[j]
            acc = first ? term : acc + term
            first = false
        end
        return acc / 2
    end
end

"f = x'Ax/2, written as a matrix-vector product followed by a dot product."
function quad_matvec(A)
    return function (x)
        acc = 0.0 * x[1]
        first = true
        for i in eachindex(x)
            yi = 0.0 * x[1]
            firstj = true
            for j in eachindex(x)
                t = A[i, j] * x[j]
                yi = firstj ? t : yi + t
                firstj = false
            end
            term = x[i] * yi
            acc = first ? term : acc + term
            first = false
        end
        return acc / 2
    end
end

"""
f = (w'x)²/2, the factored form of the rank-1 quadratic with A = w*w'. Its exact minimal cover
is abs.(w): the Hessian is w*w', so h[i]*h[j] >= abs(w[i]*w[j]) forces h >= abs.(w).
"""
function quad_factored(w)
    return function (x)
        s = 0.0 * x[1]
        first = true
        for i in eachindex(x)
            t = w[i] * x[i]
            s = first ? t : s + t
            first = false
        end
        return s^2 / 2
    end
end

function logsumexp(x)
    s = exp(x[1])
    for i in 2:length(x)
        s = s + exp(x[i])
    end
    return log(s)
end

"Squared loss of a 2-4-1 tanh MLP on one datapoint. Parameters are [vec(W1); b1; vec(W2); b2]."
function mlp_loss(xin, ytar)
    nin, nh = length(xin), 4
    return function (p)
        h = Vector{typeof(p[1])}(undef, nh)
        for k in 1:nh
            acc = 0.0 * p[1]
            first = true
            for j in 1:nin
                t = p[(j - 1) * nh + k] * xin[j]      # W1[k,j], column-major
                acc = first ? t : acc + t
                first = false
            end
            acc = acc + p[nin * nh + k]              # + b1[k]
            h[k] = tanh(acc)
        end
        off2 = nin * nh + nh                          # start of W2
        yhat = 0.0 * p[1]
        first = true
        for k in 1:nh
            t = p[off2 + k] * h[k]
            yhat = first ? t : yhat + t
            first = false
        end
        yhat = yhat + p[off2 + nh]                    # + b2
        return (yhat - ytar)^2
    end
end

"Terms whose coefficients span 10^-8 to 10^6, so the Hessian's rows differ wildly in scale."
function mixed_scales(x)
    t1 = (1e6 * x[1] * x[2] - 3)^2
    t2 = exp(x[3] * 1e3)
    t3 = x[4]^4 / 1e8
    return t1 + t2 + t3
end

"f = x'(I - e*e'/n)x/2 written entrywise, so each Hessian entry comes from exactly one tape node."
function cancellation(x)
    n = length(x)
    acc = 0.0 * x[1]
    first = true
    for i in 1:n, j in 1:n
        Mij = (i == j ? 1.0 : 0.0) - 1.0 / n
        term = Mij * x[i] * x[j]
        acc = first ? term : acc + term
        first = false
    end
    return acc / 2
end

"""
The same Hessian I - e*e'/n as [`cancellation`](@ref), written as sum(x²)/2 - (sum x)²/(2n).
The diagonal Hessian entries now arise by cancellation between the two terms, which the
absolute-value bound cannot see: B exceeds abs.(H) on the diagonal by (1 + 1/n)/(1 - 1/n).
"""
function cancellation_summed(x)
    n = length(x)
    s2 = x[1]^2
    s = x[1]
    for i in 2:n
        s2 = s2 + x[i]^2
        s = s + x[i]
    end
    return s2 / 2 - s^2 / (2n)
end

# ---- fixed problem instances --------------------------------------------------------------

const WQ = [1e-2, 1e-1, 1e0, 1e1, 1e2]     # rank-1 quadratic: A = WQ*WQ'
const AQ = WQ * WQ'
const XQ = [1.0, -2.0, 0.5, -1.5, 3.0]

const MLP_XIN = [0.5, -0.3]
const MLP_YTAR = 0.7
const MLP_P = randn(MersenneTwister(20240607), 17)

const CANCEL_X = [1.0, -2.0, 3.0, -0.5, 2.5, -1.5, 0.7, -3.0]

"Name, program, and evaluation point for each test problem."
const PROBLEMS = [
    ("rosenbrock n=2", rosenbrock, [-1.2, 1.0]),
    ("rosenbrock n=16", rosenbrock,
     Float64[isodd(i) ? -1.2 + 0.05i : 1.0 - 0.03i for i in 1:16]),
    ("quad entrywise", quad_entrywise(AQ), XQ),
    ("quad matvec", quad_matvec(AQ), XQ),
    ("quad factored", quad_factored(WQ), XQ),
    ("logsumexp n=8", logsumexp, [-3.0, -1.0, -2.0, 0.0, 1.0, 2.0, -0.5, 3.0]),
    ("mlp 2-4-1", mlp_loss(MLP_XIN, MLP_YTAR), MLP_P),
    ("mixed scales", mixed_scales, [2e-3, 2e-3, 1e-3, 100.0]),
    ("cancellation n=8", cancellation, CANCEL_X),
    ("cancel-summed n=8", cancellation_summed, CANCEL_X),
]
