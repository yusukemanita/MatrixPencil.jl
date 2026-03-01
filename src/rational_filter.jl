# Rational filter for signal preprocessing
#
# Removes specific complex-frequency components from a signal using
# an all-pass rational filter in the frequency domain.

"""
    rational_filter(h, fs, ω0s) -> h_filtered

Apply a rational filter that removes the modes at complex frequencies `ω0s`
from the signal `h`.

The filter transfer function is:
    F(ω) = ∏ₖ (ω - ω0ₖ) / (ω - conj(ω0ₖ))

# Arguments
- `h`   : input signal (AbstractVector{ComplexF64})
- `fs`  : sampling frequency (Hz)
- `ω0s` : complex frequencies to remove (Vector{ComplexF64} or scalar ComplexF64)
"""
function rational_filter(h, fs, ω0s::Vector{ComplexF64})
    dt = 1 / fs
    N  = length(h)

    H = _physics_fft(h, dt)

    f = fftfreq(N, fs)
    ω = 2π .* f
    filt = ones(ComplexF64, N)
    for ω0 in ω0s
        filt .*= (ω .- ω0) ./ (ω .- conj(ω0))
    end

    return _physics_ifft(filt .* H, dt)
end

function rational_filter(h, fs, ω0::ComplexF64)
    dt = 1 / fs
    N  = length(h)

    H = _physics_fft(h, dt)

    f = fftfreq(N, fs)
    ω = 2π .* f
    filt = (ω .- ω0) ./ (ω .- conj(ω0))

    return _physics_ifft(filt .* H, dt)
end

"""
    signal_cleaning(h, fs, ω0s) -> h_filtered

Apply `rational_filter` with zero-padding to suppress edge artifacts.
The signal is padded by 10× its length on each side before filtering.
"""
function signal_cleaning(h, fs, ω0s)
    N_z    = 10 * length(h)
    padded = [zeros(ComplexF64, N_z); reverse(h); h; zeros(ComplexF64, N_z)]
    out    = rational_filter(padded, fs, ω0s)
    return out[N_z + length(h) + 1 : N_z + 2*length(h)]
end

# ── Internal helpers ──────────────────────────────────────────────────────────

function _physics_fft(h, dt::Float64)
    # Convention: H(ω) = (dt/√2π) ∫ h(t) e^{iωt} dt
    return (dt / sqrt(2π)) .* conj.(fft(conj.(h)))
end

function _physics_ifft(H, dt::Float64)
    N  = length(H)
    dω = 2π / (N * dt)
    return (dω / sqrt(2π)) .* fft(H)
end
