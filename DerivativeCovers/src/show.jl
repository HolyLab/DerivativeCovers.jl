# Compact displays prevent a TVar from printing its entire tape.

Base.show(io::IO, t::Tape) = print(io, "Tape(", nnodes(t), " nodes, ", t.ninputs, " inputs)")

Base.show(io::IO, ::MIME"text/plain", t::Tape) =
    print(io, "Tape: ", nnodes(t), " nodes, ", t.ninputs, " inputs")

function Base.show(io::IO, x::TVar)
    return if isconstant(x)
        print(io, "TVar(", x.val, ")")
    else
        print(io, "TVar(node ", x.idx, ", ", x.val, ")")
    end
end

function Base.show(io::IO, ::MIME"text/plain", x::TVar)
    return if isconstant(x)
        print(io, "TVar: ", x.val, " (constant)")
    else
        print(io, "TVar: ", x.val, " (node ", x.idx, " of ", nnodes(x.tape), ")")
    end
end

Base.show(io::IO, x::FVar) = print(io, "FVar(", x.val, ")")

function Base.show(io::IO, ::MIME"text/plain", x::FVar)
    return if isconstant(x)
        print(io, "FVar: ", x.val, " (constant)")
    else
        print(io, "FVar: ", x.val, " (", length(channels(x).g), " inputs)")
    end
end

Base.show(io::IO, c::ForwardCover) = print(io, "ForwardCover(", length(c.grad), " inputs)")

Base.show(io::IO, ::MIME"text/plain", c::ForwardCover) =
    print(io, "ForwardCover: ", length(c.grad), " inputs; grad, h_quad")

Base.show(io::IO, c::HessianCover) = print(io, "HessianCover(", length(c.grad), " inputs)")

Base.show(io::IO, ::MIME"text/plain", c::HessianCover) =
    print(io, "HessianCover: ", length(c.grad), " inputs; grad, h_sum")

Base.show(io::IO, c::HessianDiagnostics) =
    print(io, "HessianDiagnostics(", length(c.grad), " inputs)")

Base.show(io::IO, ::MIME"text/plain", c::HessianDiagnostics) =
    print(
    io, "HessianDiagnostics: ", length(c.grad),
    " inputs; grad, B, h_sum, h_quad, h_sum_rev"
)

Base.show(io::IO, c::CoverLattice) = print(io, "CoverLattice(", length(c.grad), " inputs)")

Base.show(io::IO, ::MIME"text/plain", c::CoverLattice) =
    print(
    io, "CoverLattice: ", length(c.grad),
    " inputs; grad, B, B_signed, and h_sum/h_quad in four sign channels"
)

Base.show(io::IO, c::JacobianCover) = print(io, "JacobianCover(", c.m, "x", c.n, ")")

Base.show(io::IO, ::MIME"text/plain", c::JacobianCover) =
    print(io, "JacobianCover: ", c.m, " outputs x ", c.n, " inputs; a, b_sum")

Base.show(io::IO, c::JacobianDiagnostics) =
    print(io, "JacobianDiagnostics(", c.m, "x", c.n, ")")

Base.show(io::IO, ::MIME"text/plain", c::JacobianDiagnostics) = print(
    io, "JacobianDiagnostics: ", c.m, " outputs x ", c.n,
    " inputs; a, b_sum, b_quad, b_max, Utilde"
)

Base.show(io::IO, c::NLSCover) =
    print(io, "NLSCover(", c.m, " residuals, ", length(c.grad), " inputs)")

Base.show(io::IO, ::MIME"text/plain", c::NLSCover) = print(
    io, "NLSCover: ", c.m, " residuals, ", length(c.grad), " inputs; grad, h, h_gn, h_var_gn"
)

Base.show(io::IO, c::NLSDiagnostics) =
    print(io, "NLSDiagnostics(", c.m, " residuals, ", length(c.grad), " inputs)")

Base.show(io::IO, ::MIME"text/plain", c::NLSDiagnostics) =
    print(
    io, "NLSDiagnostics: ", c.m, " residuals, ", length(c.grad),
    " inputs; grad, h, h_gn, h_var_gn, Utilde"
)

Base.show(io::IO, c::VarianceCover) = print(io, "VarianceCover(", length(c.grad), " inputs)")

Base.show(io::IO, ::MIME"text/plain", c::VarianceCover) =
    print(io, "VarianceCover: ", length(c.grad), " inputs; grad, h_sum, h_var, severity")
