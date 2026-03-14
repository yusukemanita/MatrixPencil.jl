# t0 scan: sweep start time and find stable amplitudes

"""
    T0ModeResult

Per-mode result from a t0 scan.

# Fields
- `label`       : mode label from `theory_dict`
- `best_t0`     : t0 where `|d|A|/dt0|` is minimized
- `amplitude`   : phase-corrected complex amplitude at `best_t0`
- `omega_mpm`   : MPM frequency at `best_t0`
- `omega_ref`   : reference frequency from `theory_dict`
- `abs_amp`     : `|amplitude|` at `best_t0`
- `t0_vals`     : t0 values where this mode was detected
- `amp_vs_t0`   : phase-corrected amplitude at each `t0_vals` entry
- `omega_vs_t0` : `omega_mpm` at each `t0_vals` entry
"""
struct T0ModeResult
    label       :: String
    best_t0     :: Float64
    amplitude   :: ComplexF64
    omega_mpm   :: ComplexF64
    omega_ref   :: ComplexF64
    abs_amp     :: Float64
    t0_vals     :: Vector{Float64}
    amp_vs_t0   :: Vector{ComplexF64}
    omega_vs_t0 :: Vector{ComplexF64}
end

"""
    T0ScanResult

Container for the full t0 scan output.

# Fields
- `modes`       : `Vector{T0ModeResult}`, one per accepted mode
- `raw_results` : per-t0 raw `classify_modes` output
- `t0_values`   : all scanned t0 values
"""
struct T0ScanResult
    modes       :: Vector{T0ModeResult}
    raw_results :: Vector{Vector{LabeledMode}}
    t0_values   :: Vector{Float64}
end

"""
    time_window(t, signal, theory_dict, ti, te; kw...) -> Vector{LabeledMode}

Run the full MPM pipeline on a time window `[ti, te]`.

Extracts the signal segment nearest to `[ti, te]`, then runs
`stabilization_data` → `cluster_poles` → `classify_modes`.

# Arguments
- `t`           : time array
- `signal`      : signal data
- `theory_dict` : `Dict{String, ComplexF64}` of reference frequencies
- `ti`          : window start time
- `te`          : window end time

# Keyword arguments
All keyword arguments are forwarded to the corresponding pipeline functions.
See `stabilization_data`, `cluster_poles`, and `classify_modes` for details.
"""
function time_window(t, signal, theory_dict, ti, te;
        dt          = t[2] - t[1],
        ω_known     = collect(values(theory_dict)),
        M_RANGE     = 20:20:200,
        ε_complex   = 0.05,
        ε_assign    = 0.05,
        min_count   = 3,
        min_M_span  = 10,
        im_rel_tol  = 0.05,
        tol         = 0.05,
        poly_groups = Vector{String}[],
        use_ref_freq = true,
        δ_re        = 0.01,
        δ_im        = 0.01,
        im_lim      = (-1.0, 0.0),
        re_lim      = (-3.0, 3.0))

    ib = argmin(abs.(t .- ti))
    ie = argmin(abs.(t .- te))
    signal_tw = signal[ib:ie]

    data = stabilization_data(signal_tw, dt, M_RANGE;
            δ_re = δ_re, δ_im = δ_im, im_lim = im_lim, re_lim = re_lim)

    clusters = cluster_poles(data, signal_tw, dt;
        ω_known    = ω_known,
        ε_complex  = ε_complex,
        ε_assign   = ε_assign,
        min_count  = min_count,
        min_M_span = min_M_span,
        im_rel_tol = im_rel_tol)

    modes = classify_modes(clusters, theory_dict, signal_tw, dt;
        tol          = tol,
        poly_groups  = poly_groups,
        use_ref_freq = use_ref_freq)
    # print_mode_table(modes)

    return modes
end

"""
    t0_scan(t, signal, theory_dict, t0_range, te; min_detections=3, kw...) -> T0ScanResult

Sweep window start time `t0` and find the most stable amplitude for each mode.

For each `t0` in `t0_range`, runs `time_window(t, signal, theory_dict, t0, te; kw...)`,
phase-corrects the amplitudes back to `t = 0`, and selects the `t0` where
`|d|A(t0)|/dt0|` is minimized.

# Arguments
- `t0_range`       : range or vector of start times to scan
- `te`             : window end time — a scalar (fixed for all t0) or a
                     range/vector of the same length as `t0_range`
- `min_detections` : minimum number of t0 values where a mode must appear (default 3)
- `t_ref`          : reference time for phase correction (default `0.0`).
                     Amplitudes are corrected to `t = t_ref` via `A * exp(iω(t0 - t_ref))`.
- All other keyword arguments are forwarded to `time_window`.
  In particular, `use_ref_freq` (default `true`) controls both the Vandermonde
  amplitude fit in `classify_modes` and the phase correction in `t0_scan`.
"""
function t0_scan(t, signal, theory_dict, t0_range, te;
        min_detections::Int = 3,
        t_ref::Real = 0.0,
        kw...)

    t0_vec = Float64.(collect(t0_range))
    te_vec = te isa Number ? fill(Float64(te), length(t0_vec)) : Float64.(collect(te))
    length(t0_vec) == length(te_vec) || error("t0_range and te must have the same length")
    n_t0   = length(t0_vec)

    # Pass 1: run time_window for each t0
    raw_results = Vector{Vector{LabeledMode}}(undef, n_t0)
    all_labels  = Set{String}()

    for (i, t0) in enumerate(t0_vec)
        raw_results[i] = time_window(t, signal, theory_dict, t0, te_vec[i]; kw...)
        for m in raw_results[i]
            m.label != "unknown" && push!(all_labels, m.label)
        end
    end

    sorted_labels = sort(collect(all_labels))

    # Extract use_ref_freq from forwarded kwargs (default true)
    use_ref_freq = get(kw, :use_ref_freq, true)

    # Pass 2: phase-correct amplitudes and collect frequencies
    amps_all = Dict(lbl => Vector{Union{ComplexF64, Missing}}(missing, n_t0) for lbl in sorted_labels)
    oms_all  = Dict(lbl => Vector{Union{ComplexF64, Missing}}(missing, n_t0) for lbl in sorted_labels)

    for (i, t0) in enumerate(t0_vec)
        for m in raw_results[i]
            m.label == "unknown" && continue
            haskey(theory_dict, m.label) || continue
            ω_phase = use_ref_freq ? theory_dict[m.label] : m.omega_mpm
            amps_all[m.label][i] = m.amplitude * exp(im * ω_phase * (t0 - t_ref))
            oms_all[m.label][i]  = m.omega_mpm
        end
    end

    # Pass 3: find best t0 per mode via derivative minimization
    mode_results = T0ModeResult[]

    for lbl in sorted_labels
        valid_idx = findall(!ismissing, amps_all[lbl])
        length(valid_idx) < min_detections && continue

        a_vals  = ComplexF64[amps_all[lbl][i] for i in valid_idx]
        a_abs   = abs.(a_vals)
        t0_vals = [t0_vec[i] for i in valid_idx]
        o_vals  = ComplexF64[oms_all[lbl][i] for i in valid_idx]

        deriv = _central_differences(a_abs, t0_vals)

        best_j   = argmin(abs.(deriv))
        best_i   = valid_idx[best_j]
        best_t0  = t0_vec[best_i]
        best_amp = a_vals[best_j]
        best_om  = oms_all[lbl][best_i]
        ω_ref    = theory_dict[lbl]

        push!(mode_results, T0ModeResult(
            lbl, best_t0, best_amp, best_om, ω_ref, abs(best_amp),
            t0_vals, a_vals, o_vals))
    end

    sort!(mode_results, by = r -> real(r.omega_ref), rev=true)

    return T0ScanResult(mode_results, raw_results, t0_vec)
end

"""
    print_t0_scan_table(result::T0ScanResult)

Print a formatted table of t0 scan results.
"""
function print_t0_scan_table(result::T0ScanResult)
    modes = result.modes
    isempty(modes) && return

    @printf("%-22s  %7s  %12s  %12s  %12s\n",
            "Label", "best t0", "|A(t0_best)|", "Re(ω_mpm)", "Im(ω_mpm)")
    println("─"^75)
    for m in modes
        @printf("%-22s  %7.1f  %12.4e  %12.5f  %12.5f\n",
                m.label, m.best_t0, m.abs_amp,
                real(m.omega_mpm), imag(m.omega_mpm))
    end
end

# ── internal ──

function _central_differences(vals::Vector{Float64}, positions::Vector{Float64})
    n = length(vals)
    n == 1 && return [NaN]
    deriv = similar(vals)
    for j in 1:n
        if j == 1
            deriv[j] = (vals[2] - vals[1]) / (positions[2] - positions[1])
        elseif j == n
            deriv[j] = (vals[n] - vals[n-1]) / (positions[n] - positions[n-1])
        else
            deriv[j] = (vals[j+1] - vals[j-1]) / (positions[j+1] - positions[j-1])
        end
    end
    return deriv
end
