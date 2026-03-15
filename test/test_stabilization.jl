@testset "stabilization_data: stable flag correctness" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt
    ω1 = 0.87 - 0.089im
    y  = @. exp(-im * ω1 * t)

    data = stabilization_data(y, dt, 10:5:40;
               im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))

    # Stable poles should exist
    @test any(data.stable)

    # Stable poles should be close to ω1
    stable_re = data.re[data.stable]
    stable_im = data.im[data.stable]
    @test any(r -> abs(r - real(ω1)) < 0.05, stable_re)
    @test any(i -> abs(i - imag(ω1)) < 0.05, stable_im)
end

@testset "stabilization_data: im_lim filtering" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt
    ω1 = 0.87 - 0.089im
    y  = @. exp(-im * ω1 * t)

    # Narrow im_lim that excludes ω1 (Im(ω1)=-0.089 ∉ (-0.05, 0.0))
    data = stabilization_data(y, dt, 10:5:30;
               im_lim=(-0.05, 0.0), re_lim=(0.0, 3.0))

    # No poles should survive the filter
    @test all(i -> -0.05 < i < 0.0, data.im)
end

@testset "stabilization_data: re_lim filtering" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt
    ω1 = 0.87 - 0.089im
    y  = @. exp(-im * ω1 * t)

    # re_lim that excludes ω1 (Re(ω1)=0.87 ∉ (1.5, 3.0))
    data = stabilization_data(y, dt, 10:5:30;
               im_lim=(-0.5, 0.0), re_lim=(1.5, 3.0))

    @test all(r -> 1.5 < r < 3.0, data.re)
end

@testset "stabilization_data: tight δ_re/δ_im yields fewer stable poles" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt
    ω1 = 0.87 - 0.089im
    y  = @. exp(-im * ω1 * t)

    data_loose = stabilization_data(y, dt, 10:5:40;
                     δ_re=0.1, δ_im=0.1,
                     im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))
    data_tight = stabilization_data(y, dt, 10:5:40;
                     δ_re=1e-6, δ_im=1e-6,
                     im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))

    n_stable_loose = count(data_loose.stable)
    n_stable_tight = count(data_tight.stable)

    @test n_stable_loose >= n_stable_tight
end

@testset "stabilization_data: M >= L entries skipped" begin
    dt = 0.01
    N  = 100        # L = 50
    t  = (0:N-1) .* dt
    y  = @. exp(-im * (0.5 - 0.05im) * t)

    # M_range starts at 55 > L=50, so all M are skipped → empty output
    data = stabilization_data(y, dt, 55:5:70;
               im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))

    @test isempty(data.M)
    @test isempty(data.re)
    @test isempty(data.stable)
end

@testset "stabilization_data: method=:basic" begin
    dt = 0.01
    N  = 300
    t  = (0:N-1) .* dt
    ω1 = 0.87 - 0.089im
    y  = @. exp(-im * ω1 * t)

    # :basic should work without error and return data
    data = stabilization_data(y, dt, 10:5:30; method=:basic,
               im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))

    @test length(data.re) > 0
end
