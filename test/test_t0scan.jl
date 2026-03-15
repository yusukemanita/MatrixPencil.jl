@testset "_central_differences" begin
    using MatrixPencil: _central_differences

    # Known derivatives of f(x) = x^2: f'(x) = 2x
    # Positions: 0, 1, 2, 3, 4  →  values: 0, 1, 4, 9, 16
    positions = Float64[0, 1, 2, 3, 4]
    vals      = Float64[0, 1, 4, 9, 16]
    deriv     = _central_differences(vals, positions)

    # Interior: central difference f'(x) ≈ (f(x+1)-f(x-1))/2
    #   j=2: (4-0)/2 = 2  → f'(1)=2 ✓
    #   j=3: (9-1)/2 = 4  → f'(2)=4 ✓
    #   j=4: (16-4)/2 = 6 → f'(3)=6 ✓
    @test deriv[2] ≈ 2.0
    @test deriv[3] ≈ 4.0
    @test deriv[4] ≈ 6.0

    # Boundaries: one-sided differences
    @test deriv[1] ≈ (vals[2] - vals[1]) / (positions[2] - positions[1])
    @test deriv[5] ≈ (vals[5] - vals[4]) / (positions[5] - positions[4])

    # Single element → [NaN]
    d1 = _central_differences([3.0], [0.0])
    @test length(d1) == 1
    @test isnan(d1[1])

    # Two elements → both use forward/backward difference
    d2 = _central_differences([0.0, 2.0], [0.0, 1.0])
    @test length(d2) == 2
    @test d2[1] ≈ 2.0
    @test d2[2] ≈ 2.0
end

@testset "time_window smoke test" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt

    ω1 = 0.9 - 0.05im
    ω2 = 1.5 - 0.10im
    signal = @. 2.0 * exp(-im * ω1 * t) + 1.0 * exp(-im * ω2 * t)

    theory = Dict("m1" => ω1, "m2" => ω2)

    modes = time_window(t, signal, theory, 0.0, t[end];
                M_RANGE    = 10:5:40,
                ε_complex  = 0.05,
                ε_assign   = 0.05,
                min_count  = 3,
                min_M_span = 5,
                im_rel_tol = 0.15,
                tol        = 0.1,
                im_lim     = (-0.5, 0.0),
                re_lim     = (0.0, 3.0))

    # Should return a vector of LabeledMode (may be empty if MPM struggles on
    # a short window, but must not throw)
    @test modes isa Vector{LabeledMode}

    # All returned modes must have Im(ω) < 0 (decaying)
    for m in modes
        @test imag(m.omega_mpm) < 0
    end
end

@testset "t0_scan basic test" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt

    ω1 = 0.9 - 0.06im
    A1 = 2.0 + 0.5im
    signal = @. A1 * exp(-im * ω1 * t)

    theory = Dict("m1" => ω1)

    # Scan only a few start times to keep the test fast
    t0_range = 0.0:0.5:1.5
    te_fixed = t[end]

    result = t0_scan(t, signal, theory, t0_range, te_fixed;
                 M_RANGE    = 10:5:30,
                 ε_complex  = 0.05,
                 ε_assign   = 0.05,
                 min_count  = 3,
                 min_M_span = 5,
                 im_rel_tol = 0.15,
                 tol        = 0.1,
                 im_lim     = (-0.5, 0.0),
                 re_lim     = (0.0, 3.0))

    @test result isa T0ScanResult
    @test length(result.t0_values) == length(collect(t0_range))
    @test length(result.raw_results) == length(collect(t0_range))

    # If the mode was detected enough times it should appear in result.modes
    if !isempty(result.modes)
        m = result.modes[1]
        @test m isa T0ModeResult
        @test m.label == "m1"
        @test isfinite(m.best_t0)
        @test isfinite(abs(m.amplitude))
        @test length(m.t0_vals) == length(m.amp_vs_t0)
        @test length(m.t0_vals) == length(m.omega_vs_t0)
    end
end

@testset "t0_scan min_detections cutoff" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt

    ω1 = 0.9 - 0.06im
    signal = @. 2.0 * exp(-im * ω1 * t)

    theory = Dict("m1" => ω1)

    # Use 4 start times, require at least 5 detections → mode must be excluded
    t0_range = 0.0:0.5:1.5   # 4 values
    te_fixed = t[end]

    result = t0_scan(t, signal, theory, t0_range, te_fixed;
                 min_detections = 5,
                 M_RANGE    = 10:5:30,
                 ε_complex  = 0.05,
                 ε_assign   = 0.05,
                 min_count  = 3,
                 min_M_span = 5,
                 im_rel_tol = 0.15,
                 tol        = 0.1,
                 im_lim     = (-0.5, 0.0),
                 re_lim     = (0.0, 3.0))

    # With only 4 t0 values the mode can appear at most 4 times, so it must
    # be filtered out when min_detections=5.
    @test isempty(result.modes)
end

@testset "t0_scan length mismatch error" begin
    dt = 0.01
    N  = 256
    t  = (0:N-1) .* dt
    signal = ones(ComplexF64, N)
    theory = Dict("x" => complex(1.0, -0.05))

    # Passing te as a vector of different length should throw
    @test_throws ErrorException t0_scan(t, signal, theory, [0.0, 0.5], [1.0, 1.5, 2.0])
end

@testset "print_t0_scan_table smoke test" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt

    ω1 = 0.9 - 0.06im
    signal = @. 2.0 * exp(-im * ω1 * t)

    theory = Dict("m1" => ω1)

    result = t0_scan(t, signal, theory, 0.0:0.5:1.5, t[end];
                 M_RANGE    = 10:5:30,
                 ε_complex  = 0.05,
                 ε_assign   = 0.05,
                 min_count  = 3,
                 min_M_span = 5,
                 im_rel_tol = 0.15,
                 tol        = 0.1,
                 im_lim     = (-0.5, 0.0),
                 re_lim     = (0.0, 3.0))

    @test_nowarn print_t0_scan_table(result)

    # Smoke test with empty modes as well
    empty_result = T0ScanResult(T0ModeResult[], Vector{LabeledMode}[], Float64[])
    @test_nowarn print_t0_scan_table(empty_result)
end
