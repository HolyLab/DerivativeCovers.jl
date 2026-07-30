# Compare least-squares covers with ForwardDiff. The Gauss–Newton cover must bound J'J, remain
# below the full-Hessian cover, and equal sqrt.(diag(J'J)) for linear residuals.

"Loss `f = ½∑ₖ rₖ²` of a residual map, as a plain scalar program for ForwardDiff."
nls_loss(rmap) = x -> 0.5 * sum(abs2, rmap(x))

@testset "nls.jl" begin

    @testset "single linear residual, hand-checkable" begin
        # r = x₁ + 2x₂, so f = ½(x₁+2x₂)², H = [1 2; 2 4], J = [1 2], √diag(JᵀJ) = [1, 2].
        rmap(x) = [x[1] + 2 * x[2]]
        x0 = [0.7, -0.4]
        nls = cover_nls(WithDiagnostics(), rmap, x0)
        @test nls.m == 1
        @test nls.h_gn ≈ [1.0, 2.0]
        @test nls.h ≈ nls.h_gn
        @test nls.Utilde ≈ [1.0 2.0]
        H = ForwardDiff.hessian(nls_loss(rmap), x0)
        @test cover_ratio(H, nls.h) <= GATE
    end

    @testset "nonlinear residuals: validity and separation" begin
        rng = MersenneTwister(1)
        n, m = 4, 6
        W = randn(rng, m, n)
        tk = randn(rng, m)
        x0 = 0.3 .* randn(rng, n)
        rmap(x) = [sin(sum(W[k, j] * x[j] for j in 1:n)) - tk[k] for k in 1:m]
        nls = cover_nls(WithDiagnostics(), rmap, x0)

        f = nls_loss(rmap)
        H = ForwardDiff.hessian(f, x0)
        J = ForwardDiff.jacobian(rmap, x0)

        @test nls.m == m
        @test nls.grad ≈ ForwardDiff.gradient(f, x0)
        # Full cover of the whole Hessian, Gauss–Newton cover of JᵀJ, with h_gn beneath h.
        @test cover_ratio(H, nls.h) <= GATE
        @test cover_ratio(J' * J, nls.h_gn) <= GATE
        @test all(nls.h_gn .<= nls.h .* GATE)
        # The residual sensitivities dominate |J| entrywise.
        @test all(nls.Utilde .>= abs.(J) .- 1.0e-12)
    end

    @testset "linear residuals: exactness and no curvature" begin
        rng = MersenneTwister(2)
        n, m = 5, 9
        W = randn(rng, m, n)
        tk = randn(rng, m)
        x0 = randn(rng, n)
        rmap(x) = [W[k, :]' * x - tk[k] for k in 1:m]
        nls = cover_nls(rmap, x0)

        J = ForwardDiff.jacobian(rmap, x0)
        @test nls.h_gn ≈ sqrt.(diag(J' * J))
        # A linear residual has no ∑ rₖ∇²rₖ term, so the full cover collapses onto the GN cover.
        @test nls.h ≈ nls.h_gn
        H = ForwardDiff.hessian(nls_loss(rmap), x0)
        @test cover_ratio(H, nls.h) <= GATE
    end

    @testset "hybrid parity and independent Ũ" begin
        rng = MersenneTwister(3)
        n, m = 4, 7
        W = randn(rng, m, n)
        x0 = 0.5 .* randn(rng, n)
        rmap(x) = [tanh(sum(W[k, j] * x[j] for j in 1:n)) for k in 1:m]
        nls = cover_nls(WithDiagnostics(), rmap, x0)

        # The full cover is exactly what the hybrid merge gives on the reconstructed partition.
        t, outidx, residx, sqidx = nls_loss_tape(rmap, x0)
        groups = nls_partition(nnodes(t), sqidx)
        @test cover_hybrid(t, outidx, groups) ≈ nls.h
        # Ũ formed by the O(m·L) channel sweep matches the O(n·L) forward-sensitivity path.
        u = sensitivities(t)
        Ualt = reduce(vcat, permutedims(u[residx[k]]) for k in 1:m)
        @test nls.Utilde ≈ Ualt
    end

    @testset "coarser residual grouping stays valid and looser" begin
        rng = MersenneTwister(4)
        n, m = 3, 12
        W = randn(rng, m, n)
        tk = randn(rng, m)
        x0 = 0.4 .* randn(rng, n)
        rmap(x) = [sin(sum(W[k, j] * x[j] for j in 1:n)) - tk[k] for k in 1:m]
        fine = cover_nls(WithDiagnostics(), rmap, x0)
        coarse = cover_nls(WithDiagnostics(), rmap, x0; ngroups = 3)
        H = ForwardDiff.hessian(nls_loss(rmap), x0)
        J = ForwardDiff.jacobian(rmap, x0)
        @test cover_ratio(H, coarse.h) <= GATE
        # Summing residual atoms within a group before quadrature can only inflate either cover.
        @test all(coarse.h .>= fine.h .- 1.0e-12)
        @test all(coarse.h_gn .>= fine.h_gn .- 1.0e-12)
        # Both covers come from the same channels, so grouping costs the Gauss-Newton cover too,
        # and it still covers JᵀJ.
        @test coarse.h_gn != fine.h_gn
        @test cover_ratio(J' * J, coarse.h_gn) <= GATE
        @test all(coarse.h_gn .<= coarse.h .* GATE)
        # `ngroups` beyond the residual count is the per-residual partition.
        @test cover_nls(rmap, x0; ngroups = m).h_gn == fine.h_gn
        @test cover_nls(rmap, x0; ngroups = 2m).h_gn == fine.h_gn
        # One channel per residual is the surrogate's column quadrature; grouping sums first.
        @test fine.h_gn ≈ sqrt.(vec(sum(abs2, fine.Utilde; dims = 1)))
        gsum = zeros(3, n)
        for k in 1:m
            gsum[1 + (k - 1) % 3, :] .+= coarse.Utilde[k, :]
        end
        @test coarse.h_gn ≈ sqrt.(vec(sum(abs2, gsum; dims = 1)))
    end

    @testset "the residual channels are the Gauss-Newton part of the full cover" begin
        # `h` and `h_gn` are read off one sweep: the residual channels merge to `h_gn` and one
        # further channel carries the residual curvature, so h² - h_gn² is that channel's square
        # and `ngroups` leaves it untouched.
        rng = MersenneTwister(6)
        n, m = 3, 5
        W = randn(rng, m, n)
        x0 = 0.4 .* randn(rng, n)
        rmap(x) = [exp(sum(W[k, j] * x[j] for j in 1:n)) - 1 for k in 1:m]
        covers = [cover_nls(rmap, x0; ngroups) for ngroups in (nothing, 1, 2, 5)]
        curv = [c.h .^ 2 .- c.h_gn .^ 2 for c in covers]
        @test all(all(>=(0), v) for v in curv)
        @test all(v -> v ≈ curv[1], curv)
        # The full cover of the linear-residual case has no curvature channel at all.
        lin = cover_nls(x -> [x[1] - 2 * x[2], x[2] + 1], [0.3, -0.7]; ngroups = 1)
        @test lin.h ≈ lin.h_gn
    end

    @testset "gn_ab is a valid Jacobian cover" begin
        rng = MersenneTwister(5)
        n, m = 4, 8
        W = randn(rng, m, n)
        x0 = 0.5 .* randn(rng, n)
        rmap(x) = [tanh(sum(W[k, j] * x[j] for j in 1:n)) for k in 1:m]
        nls = cover_nls(WithDiagnostics(), rmap, x0)
        J = ForwardDiff.jacobian(rmap, x0)
        a, b = gn_ab(nls)
        @test length(a) == m
        @test length(b) == n
        @test all(a[k] * b[j] >= abs(J[k, j]) - 1.0e-12 for k in 1:m, j in 1:n)
    end

    @testset "residual that is an input node" begin
        # r = (x₁, x₂ - 1): the first residual is the raw input node.
        rmap(x) = [x[1], x[2] - 1.0]
        x0 = [2.0, 3.0]
        nls = cover_nls(WithDiagnostics(), rmap, x0)
        @test nls.h_gn ≈ [1.0, 1.0]
        @test nls.Utilde ≈ [1.0 0.0; 0.0 1.0]
    end

    # A unit residual is how `genrose`, `rosenbrock` and `chainwoo` write the additive constant of
    # their objectives: it contributes ½c² to the loss and a zero row to J.
    @testset "constant residual is a loss offset" begin
        rmap(x) = [one(eltype(x)), 10 * (x[2] - x[1]^2), x[1] - 1]
        x0 = [0.4, 0.7]
        floss = y -> 0.5 * sum(abs2, [1.0, 10 * (y[2] - y[1]^2), y[1] - 1])
        nls = cover_nls(WithDiagnostics(), rmap, x0)
        t, outidx, residx, sqidx = DerivativeCovers.nls_loss_tape(rmap, x0)

        @test t.vals[outidx] ≈ floss(x0)          # the offset reaches the loss
        @test nls.grad ≈ ForwardDiff.gradient(floss, x0)
        @test length(sqidx) == 3 && allunique(sqidx)
        @test all(iszero, nls.Utilde[1, :])
        H = ForwardDiff.hessian(floss, x0)
        @test maximum(abs.(H) ./ (nls.h * nls.h')) <= GATE
        @test all(nls.h_gn .<= nls.h .+ 1.0e-12)
    end

    @testset "input validation" begin
        x0 = [1.0, 2.0]
        @test_throws ArgumentError cover_nls(x -> x[1] + x[2], x0)          # scalar, not a vector
        @test_throws ArgumentError cover_nls(x -> [1.0, 2.0], x0)           # plain numbers, no tape
        @test_throws ArgumentError cover_nls(x -> TVar[], x0)               # no residuals
        @test_throws "every residual is a constant" cover_nls(x -> [one(eltype(x))], x0)
    end
end
