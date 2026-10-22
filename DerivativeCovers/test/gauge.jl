# Covariance of the Hessian cover pipeline under a gauge of the tape: node `k`'s value is
# multiplied by `lam[k]`, which multiplies first partials by `lam[k]/lam[p]` and second partials
# by `lam[k]/(lam[p]*lam[q])`. Scaling the inputs by `d` and the output by `c` is the case
# `lam[1:n] = d`, `lam[outidx] = c`; an interior gauge leaves both at one and rescales only the
# intermediates, which represents the same function. The cover sweeps read nothing but these
# tape entries, so the gradient, the bound and every merge must transform by the rescaling
# factors alone. Power-of-two factors make the transformation exact in floating point.
#
# The zero-value guard is where the split stops reading parent values; the later testsets check
# that both of its scale references transform with the gauge and that it fires only where a
# parent value actually vanishes.

@testset "gauge.jl" begin

    # A gauge applies to the curvature forms that store their entries. An inner-product node's
    # fixed unit pattern cannot carry gauged entries, and a declared sub-map records only a bound.
    function gauge_tape(t::Tape{T}, lam::AbstractVector) where {T}
        N = nnodes(t)
        length(lam) == N || throw(DimensionMismatch("lam must have one entry per tape node"))
        ds = [T[t.ds[k][i] * lam[k] / lam[p] for (i, p) in enumerate(t.parents[k])] for k in 1:N]
        Cs = Vector{Curvature}(undef, N)
        for k in 1:N
            ps, C = t.parents[k], t.Cs[k]
            g(i, j) = lam[k] / (lam[ps[i]] * lam[ps[j]])
            Cs[k] = if C isa NoCurvature
                C
            elseif C isa UnaryCurvature
                UnaryCurvature(C.c * g(1, 1))
            elseif C isa BinaryCurvature
                BinaryCurvature(C.c11 * g(1, 1), C.c12 * g(1, 2), C.c22 * g(2, 2))
            elseif C isa DenseCurvature
                DenseCurvature([C.C[i, j] * g(i, j) for i in eachindex(ps), j in eachindex(ps)])
            else
                error("cannot gauge a node carrying $(nameof(typeof(C)))")
            end
        end
        return Tape{T}(T[t.vals[k] * lam[k] for k in 1:N], [copy(p) for p in t.parents], ds, Cs, t.ninputs)
    end

    # The Hessian a tape represents, one Hessian-vector product per column; a gauged tape has no
    # program to hand to ForwardDiff.
    function tape_hessian(t, outidx)
        n = t.ninputs
        H = zeros(n, n)
        for i in 1:n
            e = zeros(n)
            e[i] = 1
            H[:, i] = hvp(t, outidx, e)
        end
        return H
    end

    # Every reported quantity of the pipeline. The hybrid merge deals atoms round-robin, which a
    # gauge does not reorder.
    function pipeline(t, outidx)
        res = cover_hessian(WithDiagnostics(), t, outidx)
        vc = cover_hessian(t, outidx)
        return (;
            res.grad, res.B, res.h_sum, res.h_quad, res.h_sum_rev, vc.h_var, severity = severity(vc),
            h_hyb = cover_hybrid(t, outidx, 4),
            lambda = scaled_spectral_radius(tape_hessian(t, outidx), res.h_sum),
        )
    end

    # The transformation law for scaling the inputs by `d` and the output by `c`.
    transformed(p, d, c) = (;
        grad = c .* p.grad ./ d, B = c .* p.B ./ (d * d'),
        h_sum = sqrt(c) .* p.h_sum ./ d, h_quad = sqrt(c) .* p.h_quad ./ d,
        h_sum_rev = sqrt(c) .* p.h_sum_rev ./ d, h_var = sqrt(c) .* p.h_var ./ d,
        p.severity, h_hyb = sqrt(c) .* p.h_hyb ./ d, p.lambda,
    )

    PROGS = [
        ("rosenbrock", rosenbrock, [-1.2, 1.0, -0.8, 1.3, 0.9, -0.5]),
        ("logsumexp", logsumexp, [0.3, -1.1, 2.0, 0.5, -0.2, 1.4]),
        (
            let p = Problems.quadratic(6; form = :entrywise, rng = MersenneTwister(1))
                ("quadratic entrywise", p.f, p.x0)
            end
        ),
        (
            let p = Problems.deep_composition(6; width = 2, rng = MersenneTwister(2))
                ("deep composition", p.f, p.x0)
            end
        ),
        (
            let p = Problems.random_dag(8; sharing = 0.4, rng = MersenneTwister(3))
                ("random dag", p.f, p.x0)
            end
        ),
    ]

    function full_gauge(t, outidx, rng)
        n, N = t.ninputs, nnodes(t)
        lam = 2.0 .^ rand(rng, -3:3, N)
        lam[1:n] = 2.0 .^ rand(rng, -4:4, n)
        lam[outidx] = 4.0
        return lam
    end

    @testset "an interior gauge leaves the pipeline unchanged — $name" for (name, f, x0) in PROGS
        t, outidx = tape_function(f, x0)
        n, N = t.ninputs, nnodes(t)
        p = pipeline(t, outidx)
        interior = setdiff((n + 1):N, outidx)

        lam = ones(N)
        lam[interior] = 2.0 .^ rand(MersenneTwister(11), -3:3, length(interior))
        @test any(!=(1), lam)
        @test pipeline(gauge_tape(t, lam), outidx) == p

        lam = ones(N)
        lam[interior] = exp.(randn(MersenneTwister(12), length(interior)))
        pg = pipeline(gauge_tape(t, lam), outidx)
        for k in keys(p)
            @test getproperty(pg, k) ≈ getproperty(p, k) rtol = 1.0e-12
        end
    end

    @testset "a full gauge transforms the pipeline exactly — $name" for (name, f, x0) in PROGS
        t, outidx = tape_function(f, x0)
        n = t.ninputs
        p = pipeline(t, outidx)
        lam = full_gauge(t, outidx, MersenneTwister(21))
        d, c = lam[1:n], lam[outidx]
        tg = gauge_tape(t, lam)
        @test pipeline(tg, outidx) == transformed(p, d, c)

        # The Hessian-vector product transforms as H -> c * D^-1 H D^-1.
        rng = MersenneTwister(22)
        for _ in 1:3
            w = randn(rng, n)
            @test hvp(tg, outidx, w) == c .* hvp(t, outidx, w ./ d) ./ d
        end

        # The matrix-free spectral radius sees the same scaled Hessian at every power iterate, so
        # with the same random start it returns the same estimate.
        if all(>(0), p.h_sum)
            hg = sqrt(c) .* p.h_sum ./ d
            l0 = scaled_spectral_radius(t, outidx, p.h_sum; rng = MersenneTwister(0))
            lg = scaled_spectral_radius(tg, outidx, hg; rng = MersenneTwister(0))
            @test lg == l0
        end
    end

    @testset "power-of-two input and objective scaling is exact — $name" for (name, f, x0) in PROGS
        n = length(x0)
        p = pipeline(tape_function(f, x0)...)
        # Rescaling inside the program adds nodes but, by a power of two, leaves every recorded
        # number exact. The hybrid merge's round-robin dealing depends on the tape's shape, so it
        # is compared between two wrappers of the same shape.
        d = 2.0 .^ round.(Int, range(-20, 15; length = n))
        pd = pipeline(tape_function(xt -> f(xt ./ d), d .* x0)...)
        pe = pipeline(tape_function(xt -> f(xt ./ ones(n)), x0)...)
        expected = transformed(p, d, 1.0)
        for k in (:grad, :B, :h_sum, :h_quad, :h_sum_rev, :h_var, :severity)
            @test getproperty(pd, k) == getproperty(expected, k)
        end
        @test pd.h_hyb == pe.h_hyb ./ d

        c = 4.0^3
        pc = pipeline(tape_function(x -> c * f(x), x0)...)
        expected = transformed(p, ones(n), c)
        for k in (:grad, :B, :h_sum, :h_quad, :h_var, :severity)
            @test getproperty(pc, k) == getproperty(expected, k)
        end
    end

    @testset "the guard's scale references transform with the gauge" begin
        # (x[1] - 1) * x[2] at x[1] == 1: the product's first parent value is exactly zero while
        # its input sensitivity is not, so the guard reads the declared scale. Gauging that parent
        # scales its value and its sensitivity alike, and the pipeline is unchanged at any
        # threshold.
        f = x -> (x[1] - 1.0) * x[2]
        t, outidx = tape_function(f, [1.0, 2.0])
        k0 = only(k for k in (t.ninputs + 1):nnodes(t) if t.vals[k] == 0 && t.parents[k] == [1])
        lam = ones(nnodes(t))
        lam[k0] = 4.0
        tg = gauge_tape(t, lam)
        @test pipeline(tg, outidx) == pipeline(t, outidx)
        for ratio in (nothing, 0.0)
            @test cover_hessian(tg, outidx; ratio).h_sum == cover_hessian(t, outidx; ratio).h_sum
        end
        # Away from the degeneracy the same gauge is equally inert.
        t2, outidx2 = tape_function(f, [1.3, 2.0])
        @test pipeline(gauge_tape(t2, lam), outidx2) == pipeline(t2, outidx2)

        # x[1]^2 * x[2] at x[1] == 0: the square has both a zero value and a zero sensitivity, so
        # its declared scale vanishes too. The product's cross term is then absent from the
        # Hessian, the product contributes no seed, and the cover of x[2] is zero. A gauge on the
        # dead parent changes nothing.
        fdead = x -> x[1]^2 * x[2]
        xd = [0.0, 2.0]
        td, outd = tape_function(fdead, xd)
        kd = only(k for k in (td.ninputs + 1):nnodes(td) if td.vals[k] == 0 && td.parents[k] == [1])
        sc = DerivativeCovers.split_scales(nothing, td)
        @test sc[kd].value == 0
        @test DerivativeCovers.declared_scale(sc[kd]) == 0
        @test all(iszero, td.ds[kd])
        pd = pipeline(td, outd)
        @test pd.h_sum[2] == 0
        @test pd.h_sum[1] > 0
        Hd = ForwardDiff.hessian(fdead, xd)
        for l in (4.0, 16.0)
            lamd = ones(nnodes(td))
            lamd[kd] = l
            pg = pipeline(gauge_tape(td, lamd), outd)
            for k in (:grad, :B, :h_sum, :h_quad, :h_var)
                @test getproperty(pg, k) == getproperty(pd, k)
            end
            @test cover_ratio(Hd, pg.h_sum) <= GATE
        end
    end

    @testset "the split's tie-break as a parent value approaches zero" begin
        # (x[1] - a) * x[2] at x0 = [1, 2] has H = [0 1; 1 0] for every a. With z1 = |1 - a|, the
        # value-balanced split gives h_sum[1] = sqrt(2/z1): valid at every z1 but without bound
        # as z1 -> 0. At z1 = 0 the guard reads the declared scales instead, and the cover drops
        # to a finite value.
        x0 = [1.0, 2.0]
        H = [0.0 1.0; 1.0 0.0]
        h1 = Float64[]
        for z1 in 4.0 .^ (0:-1:-8)
            f = x -> (x[1] - (1.0 - z1)) * x[2]
            @test ForwardDiff.hessian(f, x0) == H
            h = cover_hessian(f, x0).h_sum
            @test h[1] ≈ sqrt(2 / z1) rtol = 1.0e-12
            @test cover_ratio(H, h) <= GATE
            push!(h1, h[1])
        end
        h0 = cover_hessian(x -> (x[1] - 1.0) * x[2], x0).h_sum
        @test all(isfinite, h0)
        @test h0[1] < last(h1)
        @test cover_ratio(H, h0) <= GATE
    end

    @testset "the guard fires at a structural zero and nowhere else" begin
        # Whether any block read a declared scale during one pass of local covers.
        function reads_declared_scales(t, outidx)
            sc = DerivativeCovers.split_scales(nothing, t)
            r = DerivativeCovers.split_ratio(nothing, eltype(t))
            vbar = adjoints(t, outidx)
            for k in (t.ninputs + 1):nnodes(t)
                DerivativeCovers.local_cover_slots(r, t, k, abs(vbar[k]), sc)
            end
            return !isempty(sc.store.sigmas)
        end

        # (x[1] - x[1]) * x[2] has an exactly zero parent at every point, and H == 0.
        fzero = x -> (x[1] - x[1]) * x[2]
        rng = MersenneTwister(51)
        for _ in 1:5
            x0 = randn(rng, 2)
            t, outidx = tape_function(fzero, x0)
            @test reads_declared_scales(t, outidx)
            h = cover_hessian(t, outidx).h_sum
            @test all(isfinite, h) && all(>=(0), h)
            @test iszero(ForwardDiff.hessian(fzero, x0))
        end

        # Smooth programs at generic points never reach the guard.
        rng = MersenneTwister(52)
        @testset "$name" for (name, f, x0) in PROGS
            @test !reads_declared_scales(tape_function(f, x0)...)
            for _ in 1:3
                xp = x0 .+ 0.01 .* randn(rng, length(x0))
                @test !reads_declared_scales(tape_function(f, xp)...)
            end
        end
    end

end
