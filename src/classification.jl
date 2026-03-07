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
- `label`       : string key from `theory_dict`, or `"unknown"`
- `omega_mpm`   : MPM estimate `re_mean + i·im_mean`
- `omega_ref`   : reference frequency (equals `omega_mpm` for `"unknown"`)
- `amplitude`   : complex amplitude A (re-fitted simultaneously with all modes)
- `amplitude_B` : linear coefficient; non-zero only for poly-group primaries
- `re_std`, `im_std` : statistical uncertainties
- `cluster`     : the originating `ClusterResult`
"""
struct LabeledMode
    label       :: String
    omega_mpm   :: ComplexF64
    omega_ref   :: ComplexF64
    amplitude   :: ComplexF64    # A: constant term
    amplitude_B :: ComplexF64    # B: linear-in-t coefficient (zero for standard modes)
    re_std      :: Float64
    im_std      :: Float64
    cluster     :: ClusterResult
end

"""
    classify_modes(clusters, theory_dict, signal, dt; tol=0.02, poly_groups=[], use_ref_freq=true) -> Vector{LabeledMode}

Match accepted clusters to reference frequencies in `theory_dict` and
re-fit amplitudes for all matched modes simultaneously.

# Steps
1. Assign to each accepted cluster the nearest `theory_dict` entry within `tol`.
2. Remove duplicate labels (keep the cluster with the most poles).
3. Re-fit amplitudes via Vandermonde least squares on the full `signal`.

# Arguments
- `clusters`      : output of `cluster_poles`
- `theory_dict`   : `Dict{String, ComplexF64}` of reference frequencies
- `signal`        : original time series
- `dt`            : sampling interval
- `tol`           : maximum distance to assign a label (default 0.02)
- `poly_groups`   : list of label groups for polynomial fitting. Each group is a
                    `Vector{String}` where the first element is the primary label.
                    Secondary labels (positions 2+) are excluded from the output
                    and absorbed into `(A + B·t)·exp(-i·ω_primary·t)`. A single-
                    element group adds only the B column (no secondary suppression).
                    Default: `[]` (all modes use standard exponential form).
- `use_ref_freq`  : if `true` (default), labeled modes use `omega_ref` in the
                    Vandermonde matrix; unknown modes always use `omega_mpm`.
                    Set to `false` to use `omega_mpm` for all modes (exploratory
                    analysis where MPM estimates are trusted over theory values).

The result is sorted by `Re(ω)` in descending order.
"""
function classify_modes(clusters::Vector{ClusterResult},
                        theory_dict::Dict{String, ComplexF64},
                        signal, dt::Float64;
                        tol::Float64 = 0.02,
                        poly_groups::Vector{Vector{String}} = Vector{String}[],
                        use_ref_freq::Bool = true)
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
                                      zero(ComplexF64), zero(ComplexF64),
                                      c.re_std, c.im_std, c))
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

    # Step 3: simultaneous amplitude re-fit (poly-group aware)
    sig   = ComplexF64.(signal)
    N_sig = length(sig)

    # Build secondary-label suppression set and poly-primary set
    secondary_labels = Set{String}()
    poly_primary_set = Set{String}()
    label_to_idx     = Dict(r.label => k for (k, r) in enumerate(deduped))

    for grp in poly_groups
        isempty(grp)                 && continue
        haskey(label_to_idx, grp[1]) || continue  # skip if primary not found
        push!(poly_primary_set, grp[1])
        for sec in grp[2:end]
            push!(secondary_labels, sec)
        end
    end

    # Remove secondaries; track which remaining entries are poly primaries
    kept    = [r for r in deduped if r.label ∉ secondary_labels]
    is_poly = [r.label ∈ poly_primary_set for r in kept]

    # Build column spec: (kept_index, :A) or (kept_index, :B)
    col_map = Tuple{Int, Symbol}[]
    for (k, _) in enumerate(kept)
        push!(col_map, (k, :A))
        is_poly[k] && push!(col_map, (k, :B))
    end

    if !isempty(col_map)
        V_mat = zeros(ComplexF64, N_sig, length(col_map))
        for (col, (k, kind)) in enumerate(col_map)
            r = kept[k]
            ω = (use_ref_freq && r.label != "unknown") ? r.omega_ref : r.omega_mpm
            z = exp(-im * ω * dt)
            for n in 1:N_sig
                zn = z^(n - 1)
                V_mat[n, col] = kind === :A ? zn : (n - 1) * dt * zn
            end
        end
        amps = V_mat \ sig

        amp_A = zeros(ComplexF64, length(kept))
        amp_B = zeros(ComplexF64, length(kept))
        for (col, (k, kind)) in enumerate(col_map)
            kind === :A ? (amp_A[k] = amps[col]) : (amp_B[k] = amps[col])
        end

        kept = [LabeledMode(r.label, r.omega_mpm, r.omega_ref,
                            amp_A[k], amp_B[k], r.re_std, r.im_std, r.cluster)
                for (k, r) in enumerate(kept)]
    end

    deduped = kept

    sort!(deduped, by = r -> real(r.omega_mpm), rev=true)
    return deduped
end

"""
    print_mode_table(modes)

Print a formatted table of `LabeledMode` results to stdout.
"""
function print_mode_table(modes::Vector{LabeledMode})
    has_poly = any(m -> abs(m.amplitude_B) > 0, modes)
    if has_poly
        @printf("%-22s  %9s  %9s  %9s  %9s  %12s  %12s\n",
                "Label", "Re(ω)", "Im(ω)", "δRe", "δIm", "|A|", "|B|")
        println("-"^92)
        for m in modes
            @printf("%-22s  %9.5f  %9.5f  %9.5f  %9.5f  %12.4e  %12.4e\n",
                    m.label, real(m.omega_mpm), imag(m.omega_mpm),
                    m.re_std, m.im_std, abs(m.amplitude), abs(m.amplitude_B))
        end
    else
        @printf("%-22s  %9s  %9s  %9s  %9s  %12s\n",
                "Label", "Re(ω)", "Im(ω)", "δRe", "δIm", "|A|")
        println("-"^76)
        for m in modes
            @printf("%-22s  %9.5f  %9.5f  %9.5f  %9.5f  %12.4e\n",
                    m.label, real(m.omega_mpm), imag(m.omega_mpm),
                    m.re_std, m.im_std, abs(m.amplitude))
        end
    end
end
