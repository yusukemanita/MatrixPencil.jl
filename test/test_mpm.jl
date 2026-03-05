@testset "matrix_pencil_method :basic" begin
    # Synthetic signal: two decaying sinusoids with known frequencies
    dt   = 0.1
    N    = 200
    t    = (0:N-1) .* dt
    ω1   = 0.8 - 0.05im   # mode 1
    ω2   = 1.2 - 0.12im   # mode 2
    A1   = 3.0 + 1.0im
    A2   = 1.5 - 0.5im
    y    = @. A1 * exp(-im * ω1 * t) + A2 * exp(-im * ω2 * t)

    L = N ÷ 2
    omegas, amps = matrix_pencil_method(y, L, dt, 2; method = :basic)

    # Sort by real part for comparison
    idx = sortperm(real.(omegas))
    ω_found = omegas[idx]

    @test abs(ω_found[1] - ω1) < 1e-6
    @test abs(ω_found[2] - ω2) < 1e-6
end

@testset "matrix_pencil_method :fb" begin
    # Same signal with mild additive noise
    dt  = 0.1
    N   = 200
    t   = (0:N-1) .* dt
    ω1  = 0.8 - 0.05im
    A1  = 2.0 + 0.0im
    y   = @. A1 * exp(-im * ω1 * t)
    y  .+= 1e-3 .* (randn(N) .+ im .* randn(N))

    L = N ÷ 2
    omegas, _ = matrix_pencil_method(y, L, dt, 4)   # :fb is the default

    # All returned poles must have Im(ω) < 0
    @test all(imag.(omegas) .< 0)

    # The dominant pole should be close to ω1
    best = argmin(abs.(omegas .- ω1))
    @test abs(omegas[best] - ω1) < 1e-3
end

@testset "stabilization_data" begin
    dt  = 0.1
    N   = 300
    t   = (0:N-1) .* dt
    ω1  = 0.87 - 0.089im
    y   = @. exp(-im * ω1 * t)

    data = stabilization_data(y, dt, 10:5:30;
               im_lim=(-0.5, 0.0), re_lim=(0.0, 2.0))

    @test length(data.re) > 0
    @test all(data.im .< 0)
    @test any(data.stable)
end

@testset "classify_modes poly_groups" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt

    ω_A  = 1.5 - 0.05im
    ω_p  = 0.8 - 0.08im    # primary ("prim")
    ω_s  = 0.805 - 0.082im  # secondary ("sec"), very close to ω_p

    A_A  = 2.0 + 0.0im
    A_p  = 1.0 + 0.5im
    A_s  = 0.3 - 0.1im

    signal = @. A_A * exp(-im * ω_A * t) +
                A_p * exp(-im * ω_p * t) +
                A_s * exp(-im * ω_s * t)

    theory = Dict("A" => ω_A, "prim" => ω_p, "sec" => ω_s)

    # --- standard fit (no poly_groups): three separate modes ---
    data     = stabilization_data(signal, dt, 10:5:40;
                   im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))
    clusters = cluster_poles(data, signal, dt;
                   ω_known=[ω_A, ω_p, ω_s], ε_complex=0.003, ε_assign=0.05,
                   min_count=3, min_M_span=5)
    modes_std = classify_modes(clusters, theory, signal, dt)

    @test all(m -> abs(m.amplitude_B) == 0, modes_std)

    # --- poly fit: merge "prim" + "sec" into (A+Bt)exp(-i ω_p t) ---
    modes_poly = classify_modes(clusters, theory, signal, dt;
                     poly_groups=[["prim", "sec"]])

    # "sec" must not appear in output
    @test all(m -> m.label != "sec", modes_poly)

    # "prim" must appear with nonzero B
    idx = findfirst(m -> m.label == "prim", modes_poly)
    @test idx !== nothing
    @test abs(modes_poly[idx].amplitude_B) > 0

    # "A" must be present and unchanged (B == 0)
    idx_A = findfirst(m -> m.label == "A", modes_poly)
    @test idx_A !== nothing
    @test abs(modes_poly[idx_A].amplitude_B) == 0

    # print_mode_table must not throw
    @test_nowarn print_mode_table(modes_poly)
end
