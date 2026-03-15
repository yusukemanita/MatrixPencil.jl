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

@testset "matrix_pencil_method: amplitude recovery" begin
    dt = 0.1
    N  = 200
    t  = (0:N-1) .* dt
    ω1 = 0.8 - 0.05im
    ω2 = 1.2 - 0.12im
    A1 = 3.0 + 1.0im
    A2 = 1.5 - 0.5im
    y  = @. A1 * exp(-im * ω1 * t) + A2 * exp(-im * ω2 * t)

    L = N ÷ 2
    omegas, amps = matrix_pencil_method(y, L, dt, 2; method = :basic)

    # Reconstruct signal from recovered poles and amplitudes
    poles = exp.(-im .* omegas .* dt)
    y_rec = zeros(ComplexF64, N)
    for n in 1:N
        for (k, p) in enumerate(poles)
            y_rec[n] += amps[k] * p^(n - 1)
        end
    end

    @test norm(y_rec - y) / norm(y) < 1e-6
end

@testset "matrix_pencil_method :fb: high noise" begin
    dt  = 0.1
    N   = 300
    t   = (0:N-1) .* dt
    ω1  = 0.8 - 0.05im
    A1  = 2.0 + 0.0im
    y   = @. A1 * exp(-im * ω1 * t)

    # SNR ≈ 10 dB: noise amplitude ~0.3 × signal amplitude
    rng = MersenneTwister(42)
    y  .+= 0.3 .* (randn(rng, N) .+ im .* randn(rng, N))

    L = N ÷ 2
    omegas, _ = matrix_pencil_method(y, L, dt, 6)

    @test all(imag.(omegas) .< 0)
    best = argmin(abs.(omegas .- ω1))
    @test abs(real(omegas[best]) - real(ω1)) / abs(real(ω1)) < 0.05
    @test abs(imag(omegas[best]) - imag(ω1)) / abs(imag(ω1)) < 0.05
end

@testset "matrix_pencil_method: invalid method throws" begin
    y = ones(ComplexF64, 50)
    @test_throws ErrorException matrix_pencil_method(y, 25, 0.1, 2; method = :bogus)
end

@testset "matrix_pencil_method :basic returns growing modes" begin
    dt  = 0.1
    N   = 200
    t   = (0:N-1) .* dt
    ω_growing = 0.5 + 0.05im   # Im > 0 → growing
    y   = @. exp(-im * ω_growing * t)

    L = N ÷ 2
    omegas_basic, _ = matrix_pencil_method(y, L, dt, 2; method = :basic)
    omegas_fb, _    = matrix_pencil_method(y, L, dt, 2; method = :fb)

    # :basic can return growing poles; :fb must not
    @test any(imag.(omegas_basic) .> 0)
    @test all(imag.(omegas_fb)   .< 0)
end

@testset "matrix_pencil_method: many modes" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt
    ωs = [0.5 - 0.02im, 1.0 - 0.05im, 1.5 - 0.08im, 2.0 - 0.03im, 2.5 - 0.10im]
    y  = sum(exp.(-im .* ω .* t) for ω in ωs)

    L = N ÷ 2
    omegas, _ = matrix_pencil_method(y, L, dt, length(ωs); method = :basic)

    # Sort by real part and compare
    found_re  = sort(real.(omegas))
    true_re   = sort(real.(ωs))
    for (f, r) in zip(found_re, true_re)
        @test abs(f - r) < 0.01
    end
end

@testset "classify_modes: empty theory_dict gives unknown labels" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt
    ω1 = 0.9 - 0.05im
    y  = @. 2.0 * exp(-im * ω1 * t)

    data     = stabilization_data(y, dt, 10:5:40;
                   im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))
    clusters = cluster_poles(data, y, dt;
                   ε_complex=0.05, min_count=3, min_M_span=5)
    modes    = classify_modes(clusters, Dict{String, ComplexF64}(), y, dt)

    @test !isempty(modes)
    @test all(m -> m.label == "unknown", modes)
end

@testset "classify_modes: tol threshold excludes distant reference" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt
    ω1 = 0.9 - 0.05im
    y  = @. 2.0 * exp(-im * ω1 * t)

    data     = stabilization_data(y, dt, 10:5:40;
                   im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))
    clusters = cluster_poles(data, y, dt;
                   ε_complex=0.05, min_count=3, min_M_span=5)

    # Reference far from the actual cluster
    far_theory = Dict("far_mode" => complex(5.0, -0.05))
    modes = classify_modes(clusters, far_theory, y, dt; tol=0.1)

    @test all(m -> m.label == "unknown", modes)
end

@testset "classify_modes: deduplication keeps larger cluster" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt
    ω1 = 0.9 - 0.05im
    y  = @. 2.0 * exp(-im * ω1 * t)

    data     = stabilization_data(y, dt, 10:5:40;
                   im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))
    clusters = cluster_poles(data, y, dt;
                   ε_complex=0.05, min_count=3, min_M_span=5)

    # Two reference frequencies very close to each other and to the cluster;
    # both within tol → one should be deduplicated
    theory = Dict("close_a" => complex(0.9, -0.05),
                  "close_b" => complex(0.91, -0.05))
    modes = classify_modes(clusters, theory, y, dt; tol=0.1)

    labels = [m.label for m in modes if m.label != "unknown"]
    # No duplicates: each reference label appears at most once
    @test length(labels) == length(unique(labels))
end

@testset "classify_modes: use_ref_freq=false" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt
    ω1 = 0.9 - 0.05im
    y  = @. 2.0 * exp(-im * ω1 * t)

    data     = stabilization_data(y, dt, 10:5:40;
                   im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))
    clusters = cluster_poles(data, y, dt;
                   ε_complex=0.05, min_count=3, min_M_span=5)
    theory   = Dict("m1" => ω1)

    # Should not throw, and amplitude should be finite
    modes = classify_modes(clusters, theory, y, dt; use_ref_freq=false)
    for m in modes
        @test isfinite(abs(m.amplitude))
    end
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
