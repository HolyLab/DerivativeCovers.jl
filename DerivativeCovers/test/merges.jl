# The sum, quadrature, and hybrid merges combine the same per-atom contributions in different
# ways. Writing those contributions out explicitly gives an independent reference for each merge
# and for the inequalities that relate them.

@testset "merges.jl" begin

    # Atom `α` of node `k` contributes `s_α = sum_{i in α} rho_k[i] * utilde[p_i]` to the cover,
    # where `rho_k` is the node's local cover and `utilde` the absolute input sensitivities. The
    # sum merge adds these contributions, the quadrature merge adds their squares, and the hybrid
    # merge adds within groups and squares across them. Returns the contributions and the node
    # each belongs to.
    function atom_contributions(t, outidx)
        n = t.ninputs
        rhos, _ = local_covers(t, adjoints(t, outidx))
        u = sensitivities(t)
        seeds = Vector{Float64}[]
        nodes = Int[]
        for k in (n + 1):nnodes(t)
            isempty(rhos[k]) && continue
            ps = t.parents[k]
            for atom in DerivativeCovers.atom_slots(t.Cs[k], length(ps))
                s = zeros(n)
                for i in atom
                    s .+= rhos[k][i] .* u[ps[i]]
                end
                push!(seeds, s)
                push!(nodes, k)
            end
        end
        return seeds, nodes
    end

    @testset "merges and the participation ratio — $name" for (name, f, x0) in PROBLEMS
        t, outidx = tape_function(f, x0)
        res = cover_hessian(WithDiagnostics(), t, outidx)
        seeds, _ = atom_contributions(t, outidx)
        @test sum(seeds) ≈ res.h_sum rtol = 1.0e-12
        @test sqrt.(sum(s -> s .^ 2, seeds)) ≈ res.h_quad rtol = 1.0e-12

        # The participation ratio `P = (h_sum/h_quad)^2` lies between 1 and the number of atoms
        # contributing at that coordinate (Cauchy–Schwarz).
        for i in 1:t.ninputs
            res.h_quad[i] > 0 || continue
            P = (res.h_sum[i] / res.h_quad[i])^2
            m = count(s -> s[i] > 0, seeds)
            @test 1 - 1.0e-12 <= P <= m * (1 + 1.0e-12)
        end
    end

    @testset "closed forms on the all-ones quadratic" begin
        # f = sum_ij x[i]*x[j] / 2, every term a binary product, at x = ones(n): H = ones(n, n).
        # Coordinate i receives a contribution 1/sqrt(2) from each of the 2(n-1) cross products
        # and sqrt(2) from x[i]*x[i], so h_sum = sqrt(2)*n, h_quad = sqrt(n+1), and B = ones(n, n).
        function allones(x)
            s = x[1] * x[1]
            for i in eachindex(x), j in eachindex(x)
                (i == firstindex(x) && j == firstindex(x)) && continue
                s = s + x[i] * x[j]
            end
            return 0.5 * s
        end
        for n in (3, 8)
            x0 = ones(n)
            @test ForwardDiff.hessian(allones, x0) == ones(n, n)
            res = cover_hessian(WithDiagnostics(), allones, x0)
            @test res.h_sum ≈ fill(sqrt(2) * n, n) rtol = 1.0e-12
            @test res.h_quad ≈ fill(sqrt(n + 1), n) rtol = 1.0e-12
            @test res.B ≈ ones(n, n) rtol = 1.0e-12
            @test (res.h_sum ./ res.h_quad) .^ 2 ≈ fill(2n^2 / (n + 1), n) rtol = 1.0e-12
        end
    end

    @testset "partition bounds on the hybrid merge — $name" for (name, f, x0) in PROBLEMS
        t, outidx = tape_function(f, x0)
        N, n = nnodes(t), t.ninputs
        res = cover_hessian(WithDiagnostics(), t, outidx)
        seeds, nodes = atom_contributions(t, outidx)
        rng = MersenneTwister(hash(name))
        for (G, groups) in (
                (3, cyclic_partition(t, 3)), (4, level_partition(t, 4)),
                (5, rand(rng, 1:5, N)),
            )
            h = cover_hybrid(t, outidx, groups)
            for i in 1:n
                # Within group g, t_g = sqrt(P_g) * (root sum of squares over g), so the hybrid
                # merge is at most the largest per-group participation times the quadrature merge.
                tg = zeros(G)
                qg = zeros(G)
                for (s, k) in zip(seeds, nodes)
                    tg[groups[k]] += s[i]
                    qg[groups[k]] += s[i]^2
                end
                maxP = maximum((tg[g]^2 / qg[g] for g in 1:G if qg[g] > 0); init = 0.0)
                @test h[i] <= sqrt(maxP) * res.h_quad[i] * GATE
                # Cauchy–Schwarz over the G group totals.
                @test h[i] >= res.h_sum[i] / sqrt(G) / GATE
            end
        end
    end

    @testset "expected hybrid merge under a uniform random partition" begin
        # Averaging over all G^m assignments of the m atoms to G groups is exact:
        # E[h_G^2] = M_ii + (h_sum^2 - M_ii)/G with M_ii = h_quad^2.
        f = x -> x[1] * x[2] + x[2] * x[3] + x[1] * x[3] + exp(x[1])
        x0 = [0.4, -0.9, 1.3]
        t, outidx = tape_function(f, x0)
        res = cover_hessian(WithDiagnostics(), t, outidx)
        seeds, nodes = atom_contributions(t, outidx)
        @test allunique(nodes)                  # one atom per node, so nodes can stand for atoms
        m = length(nodes)
        for G in (2, 3)
            acc = zeros(t.ninputs)
            nassign = 0
            for assignment in Iterators.product(fill(1:G, m)...)
                groups = ones(Int, nnodes(t))
                groups[nodes] .= assignment
                acc .+= cover_hybrid(t, outidx, groups) .^ 2
                nassign += 1
            end
            M = res.h_quad .^ 2
            @test acc ./ nassign ≈ M .+ (res.h_sum .^ 2 .- M) ./ G rtol = 1.0e-12
        end
    end

end
