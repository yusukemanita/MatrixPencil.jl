# Matrix Pencil Method (MPM) and Forward-Backward MPM
#
# Reference: Generalized Pencil-of-Function method
# https://en.wikipedia.org/wiki/Generalized_pencil-of-function_method
#
# Forward-Backward averaging reference: Hua & Sarkar (1990)

"""
    matrix_pencil_method(y, L, dt, M; method=:fb) -> (omegas, amplitudes)

Matrix Pencil Method (MPM) for estimating complex frequencies and amplitudes
from a sampled signal `y`.

# Arguments
- `y`      : sampled signal (AbstractVector)
- `L`      : pencil parameter (typically `length(y) ÷ 2`)
- `dt`     : sampling interval
- `M`      : model order (number of poles to extract)
- `method` : `:fb` (default) for Forward-Backward MPM, or `:basic` for standard MPM

# Methods
- `:fb`    — Forward-Backward MPM (Hua & Sarkar 1990). Combines forward and
             backward Hankel matrices before SVD. More noise-robust; returns
             only physically decaying modes (`Im(ω) < 0`).
- `:basic` — Standard MPM. Returns all `M` poles including growing modes.

# Returns
- `omegas`     : complex frequencies ω = Re(ω) + i·Im(ω)
- `amplitudes` : complex amplitudes (least-squares fit)
"""
function matrix_pencil_method(y, L, dt, M; method = :fb)
    if method === :fb
        return _mpm_fb(y, L, dt, M)
    elseif method === :basic
        return _mpm_basic(y, L, dt, M)
    else
        error("Unknown method: $method. Use :fb or :basic")
    end
end

# ── Internal implementations ───────────────────────────────────────────────────

function _mpm_basic(y, L, dt, M)
    N = length(y)
    H = [y[i+j] for i in 1:N-L, j in 0:L]

    U, _, _ = svd(H)
    Us = U[:, 1:M]

    # TLS-MPM
    Y1 = Us[1:end-1, :]
    Y2 = Us[2:end, :]

    _, _, V_tls = svd([Y1 Y2])

    V12 = V_tls[1:M, M+1:2M]
    V22 = V_tls[M+1:2M, M+1:2M]

    Z_mat = -V12 / V22
    poles = eigvals(Z_mat)

    # Compute physical complex frequencies
    omegas = im .* log.(poles) ./ dt

    # Compute complex amplitudes (least squares using Vandermonde matrix)
    V_mat = zeros(ComplexF64, N, M)
    for j in 1:M, n in 1:N
        V_mat[n, j] = poles[j]^(n - 1)
    end
    amplitudes = V_mat \ y

    return omegas, amplitudes
end

function _mpm_fb(y, L, dt, M)
    N = length(y)

    # Forward Hankel matrix: H_f[i,j] = y[i+j], size (N-L) × (L+1)
    H_f = [y[i+j]         for i in 1:N-L, j in 0:L]

    # Backward Hankel matrix: H_b[i,j] = conj(y[N+1-i-j]), size (N-L) × (L+1)
    H_b = [conj(y[N+1-i-j]) for i in 1:N-L, j in 0:L]

    # Total (Forward + Backward) data matrix
    H = vcat(H_f, H_b)

    U, _, _ = svd(H)

    # Split signal subspace into forward and backward parts
    U_f = U[1:N-L,     1:M]
    U_b = U[N-L+1:end, 1:M]

    # Combined TLS-MPM pencil matrices from both subspaces
    Y1 = vcat(U_f[1:end-1, :], U_b[1:end-1, :])
    Y2 = vcat(U_f[2:end,   :], U_b[2:end,   :])

    _, _, V_tls = svd([Y1 Y2])

    V12 = V_tls[1:M, M+1:2M]
    V22 = V_tls[M+1:2M, M+1:2M]

    Z_mat = -V12 / V22
    poles = eigvals(Z_mat)

    # Compute physical complex frequencies
    omegas = im .* log.(poles) ./ dt

    # Filter to physically decaying modes (Im(ω) < 0)
    mask        = imag.(omegas) .< 0
    omegas_filt = omegas[mask]
    poles_filt  = poles[mask]
    M_filt      = length(poles_filt)

    # Compute complex amplitudes using only the filtered poles
    V_mat = zeros(ComplexF64, N, M_filt)
    for j in 1:M_filt, n in 1:N
        V_mat[n, j] = poles_filt[j]^(n - 1)
    end
    amplitudes = V_mat \ y

    return omegas_filt, amplitudes
end
