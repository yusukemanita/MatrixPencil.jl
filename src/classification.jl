# QNM classification and mode labeling

"""
    ClusterResult

Statistical summary of a single pole cluster from `cluster_poles`.

# Fields
- `re_mean`, `im_mean` : trimmed-mean Re(ω) and Im(ω)
- `re_std`,  `im_std`  : trimmed standard deviations (uncertainty estimate)
- `im_rel_std`         : `im_std / |im_mean|` (Im stability indicator)
- `N`                  : number of poles in cluster
- `M_min`, `M_max`, `M_span` : model-order range
- `accepted`           : true if all acceptance criteria are met
- `amplitude`          : complex amplitude from Vandermonde least squares (zero if rejected)
- `re_vals`, `im_vals`, `M_vals` : raw pole data (for plotting)
"""
struct ClusterResult
    re_mean    :: Float64
    im_mean    :: Float64
    re_std     :: Float64
    im_std     :: Float64
    im_rel_std :: Float64
    N          :: Int
    M_min      :: Int
    M_max      :: Int
    M_span     :: Int
    accepted   :: Bool
    amplitude  :: ComplexF64
    re_vals    :: Vector{Float64}
    im_vals    :: Vector{Float64}
    M_vals     :: Vector{Int}
end

"""
    LabeledMode

Cluster matched to a reference frequency from a theory dictionary.

# Fields
- `label`     : string key from `theory_dict`, or `"unknown"`
- `omega_mpm` : MPM estimate `re_mean + i·im_mean`
- `omega_ref` : reference frequency (equals `omega_mpm` for `"unknown"`)
- `amplitude` : complex amplitude (re-fitted simultaneously with all modes)
- `re_std`, `im_std` : statistical uncertainties
- `cluster`   : the originating `ClusterResult`
"""
struct LabeledMode
    label      :: String
    omega_mpm  :: ComplexF64
    omega_ref  :: ComplexF64
    amplitude  :: ComplexF64
    re_std     :: Float64
    im_std     :: Float64
    cluster    :: ClusterResult
end

"""
    classify_modes(clusters, theory_dict, signal, dt; tol=0.02) -> Vector{LabeledMode}

Match accepted clusters to reference frequencies in `theory_dict` and
re-fit amplitudes for all matched modes simultaneously.

# Steps
1. Assign to each accepted cluster the nearest `theory_dict` entry within `tol`.
2. Remove duplicate labels (keep the cluster with the most poles).
3. Re-fit amplitudes via Vandermonde least squares on the full `signal`.

# Arguments
- `clusters`     : output of `cluster_poles`
- `theory_dict`  : `Dict{String, ComplexF64}` of reference frequencies
- `signal`       : original time series
- `dt`           : sampling interval
- `tol`          : maximum distance to assign a label (default 0.02)

The result is sorted by `Re(ω)` in descending order.
"""
function classify_modes(clusters::Vector{ClusterResult},
                        theory_dict::Dict{String, ComplexF64},
                        signal, dt::Float64;
                        tol::Float64 = 0.02)
    # Step 1: label assignment
    candidates = LabeledMode[]
    for c in clusters
        c.accepted || continue
        ω_mpm = complex(c.re_mean, c.im_mean)

        best_label = "unknown"
        best_omega = ω_mpm
        min_dist   = Inf
        for (label, ω_th) in theory_dict
            d = abs(ω_mpm - ω_th)
            if d < min_dist
                min_dist   = d
                best_label = d < tol ? label : "unknown"
                best_omega = d < tol ? ω_th  : ω_mpm
            end
        end
        push!(candidates, LabeledMode(best_label, ω_mpm, best_omega,
                                      zero(ComplexF64), c.re_std, c.im_std, c))
    end

    # Step 2: deduplicate (keep cluster with most poles per label)
    deduped = LabeledMode[]
    seen    = Dict{String, Int}()
    for r in candidates
        if r.label == "unknown"
            push!(deduped, r)
        elseif haskey(seen, r.label)
            j = seen[r.label]
            if r.cluster.N > deduped[j].cluster.N
                deduped[j] = r
            end
        else
            push!(deduped, r)
            seen[r.label] = length(deduped)
        end
    end

    # Step 3: simultaneous amplitude re-fit
    sig   = ComplexF64.(signal)
    N_sig = length(sig)
    K     = length(deduped)
    if K > 0
        V_mat = zeros(ComplexF64, N_sig, K)
        for (k, r) in enumerate(deduped), n in 1:N_sig
            V_mat[n, k] = exp(-im * r.omega_mpm * dt)^(n - 1)
        end
        amps = V_mat \ sig
        deduped = [LabeledMode(r.label, r.omega_mpm, r.omega_ref,
                               amps[k], r.re_std, r.im_std, r.cluster)
                   for (k, r) in enumerate(deduped)]
    end

    sort!(deduped, by = r -> real(r.omega_mpm), rev=true)
    return deduped
end

"""
    print_mode_table(modes)

Print a formatted table of `LabeledMode` results to stdout.
"""
function print_mode_table(modes::Vector{LabeledMode})
    @printf("%-22s  %9s  %9s  %9s  %9s  %12s\n",
            "Label", "Re(ω)", "Im(ω)", "δRe", "δIm", "|A|")
    println("-"^76)
    for m in modes
        @printf("%-22s  %9.5f  %9.5f  %9.5f  %9.5f  %12.4e\n",
                m.label,
                real(m.omega_mpm), imag(m.omega_mpm),
                m.re_std, m.im_std,
                abs(m.amplitude))
    end
end
