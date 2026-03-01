@testset "matrix_pencil_method" begin
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
    omegas, amps = matrix_pencil_method(y, L, dt, 2)

    # Sort by real part for comparison
    idx = sortperm(real.(omegas))
    ω_found = omegas[idx]

    @test abs(ω_found[1] - ω1) < 1e-6
    @test abs(ω_found[2] - ω2) < 1e-6
end

@testset "matrix_pencil_method_fb" begin
    # Same signal with mild additive noise
    dt  = 0.1
    N   = 200
    t   = (0:N-1) .* dt
    ω1  = 0.8 - 0.05im
    A1  = 2.0 + 0.0im
    y   = @. A1 * exp(-im * ω1 * t)
    y  .+= 1e-3 .* (randn(N) .+ im .* randn(N))

    L = N ÷ 2
    omegas, _ = matrix_pencil_method_fb(y, L, dt, 4)

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
