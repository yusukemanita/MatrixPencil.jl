# Stabilization diagram data collection
#
# Runs FB-MPM across a range of model orders M and tracks pole stability:
# a pole is "stable" if it reappears within (δ_re, δ_im) of a pole from the
# previous model order.

"""
    stabilization_data(signal, dt, M_range; method, δ_re, δ_im, im_lim, re_lim)

Run MPM for each model order in `M_range` and collect poles with
stability flags.

A pole at order M is marked **stable** if there exists a pole at order M-1
within absolute distance `δ_re` in Re(ω) and `δ_im` in Im(ω).

# Arguments
- `method` : `:fb` (default) for Forward-Backward MPM, or `:basic` for standard MPM

# Returns
`NamedTuple` with fields:
- `M`      : model order for each pole
- `re`     : Re(ω)
- `im`     : Im(ω)
- `amp`    : |amplitude|
- `stable` : Bool stability flag
"""
function stabilization_data(signal, dt, M_range;
                             method = :fb,
                             δ_re   = 0.01,
                             δ_im   = 0.01,
                             im_lim = (-0.5, 0.0),
                             re_lim = (-2.0, 2.0))
    all_M      = Int[]
    all_re     = Float64[]
    all_im     = Float64[]
    all_amp    = Float64[]
    all_stable = Bool[]
    prev_poles = ComplexF64[]

    for M in M_range
        L = length(signal) ÷ 2
        poles, amps = matrix_pencil_method(signal, L, dt, M; method)

        mask  = (im_lim[1] .< imag.(poles) .< im_lim[2]) .&
                (re_lim[1] .< real.(poles) .< re_lim[2])
        poles = poles[mask]
        amps  = amps[mask]

        stable = falses(length(poles))
        if !isempty(prev_poles)
            for (j, p) in enumerate(poles)
                idx = argmin(abs.(prev_poles .- p))
                pn  = prev_poles[idx]
                if abs(real(pn) - real(p)) < δ_re &&
                   abs(imag(pn) - imag(p)) < δ_im
                    stable[j] = true
                end
            end
        end

        append!(all_M,      fill(M, length(poles)))
        append!(all_re,     real.(poles))
        append!(all_im,     imag.(poles))
        append!(all_amp,    abs.(amps))
        append!(all_stable, stable)
        prev_poles = poles
    end

    return (M=all_M, re=all_re, im=all_im, amp=all_amp, stable=all_stable)
end

# Map log-amplitude to marker sizes in [ms_min, ms_max] for plotting
function _amp_to_ms(amp; ms_min = 2.0, ms_max = 10.0)
    la      = log10.(amp .+ 1e-30)
    lo, hi  = extrema(la)
    hi ≈ lo && return fill((ms_min + ms_max) / 2, length(amp))
    return ms_min .+ (ms_max - ms_min) .* (la .- lo) ./ (hi - lo)
end
