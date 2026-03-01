# Pole clustering via Union-Find
#
# Two strategies:
#   - Simple (ω_known=nothing) : merge any two stable poles within ε_complex
#   - Tagged (ω_known provided) : same, but poles tagged to different known
#     frequencies are never merged (prevents close-mode contamination)

"""
    cluster_poles(data, signal, dt; ω_known, ε_complex, ε_assign,
                  min_count, min_M_span, im_rel_tol, trim_frac)
        -> Vector{ClusterResult}

Cluster stable poles from a stabilization diagram and compute complex
amplitudes for accepted clusters via Vandermonde least squares.

# Strategies
- **Untagged** (`ω_known = nothing`, default): generic Union-Find in the
  complex plane. Works without prior knowledge of mode frequencies.
- **Tagged** (`ω_known = [ω₁, ω₂, …]`): each pole is first assigned to
  the nearest known frequency within `ε_assign`. Poles tagged to *different*
  known frequencies are never merged. Recommended when two modes are closer
  than `2 × ε_complex` (e.g. the "221" / "220+220-220*" near-resonant pair).

# Acceptance criteria (all must be satisfied)
| Parameter    | Default | Meaning                              |
|:-------------|:--------|:-------------------------------------|
| `min_count`  | 3       | minimum poles per cluster            |
| `min_M_span` | 5       | minimum range of model orders M      |
| `im_rel_tol` | 0.15    | std(Im) / |mean(Im)| upper bound      |

# Warning
Set `ε_complex` smaller than half the minimum separation between any two
distinct physical modes you want to resolve. For the "221"/"220+220-220*"
pair (separation ≈ 0.012), use `ε_complex ≤ 0.005` or switch to tagged mode.
"""
function cluster_poles(data, signal, dt;
                       ω_known    = nothing,
                       ε_complex  = 0.005,
                       ε_assign   = 0.005,
                       min_count  = 3,
                       min_M_span = 5,
                       im_rel_tol = 0.15,
                       trim_frac  = 0.25)
    sm   = data.stable
    re_s = data.re[sm]
    im_s = data.im[sm]
    M_s  = data.M[sm]

    isempty(re_s) && return ClusterResult[]

    poles = complex.(re_s, im_s)

    # Choose clustering strategy
    labels = if ω_known === nothing
        _cluster_2d_simple(poles, ε_complex)
    else
        tags = _assign_tags(poles, ω_known, ε_assign)
        n_tagged = count(!=(0), tags)
        @info "Tagged $(n_tagged) / $(length(poles)) poles  (untagged: $(length(poles) - n_tagged))"
        _cluster_2d_tagged(poles, tags, ε_complex)
    end

    # Pass 1: cluster statistics and acceptance decision
    raw = []
    for lbl in unique(labels)
        idx                     = findall(==(lbl), labels)
        re_c, im_c, M_c         = re_s[idx], im_s[idx], M_s[idx]
        N                       = length(idx)
        re_mean, re_std         = _trimmed_stats(re_c; trim_frac)
        im_mean, im_std         = _trimmed_stats(im_c; trim_frac)
        im_rel_std              = im_std / abs(im_mean)
        M_min, M_max            = minimum(M_c), maximum(M_c)
        M_span                  = M_max - M_min
        accepted = (N >= min_count) && (M_span >= min_M_span) &&
                   (im_rel_std <= im_rel_tol)
        push!(raw, (re_mean, im_mean, re_std, im_std, im_rel_std,
                    N, M_min, M_max, M_span, accepted, re_c, im_c, M_c))
    end

    # Pass 2: batch least-squares amplitude for accepted clusters
    acc_idx        = findall(r -> r[10], raw)
    N_sig          = length(signal)
    amplitudes_acc = zeros(ComplexF64, length(acc_idx))
    if !isempty(acc_idx)
        z_acc = [exp(-im * complex(raw[i][1], raw[i][2]) * dt) for i in acc_idx]
        V_mat = zeros(ComplexF64, N_sig, length(z_acc))
        for (k, z) in enumerate(z_acc), n in 1:N_sig
            V_mat[n, k] = z^(n - 1)
        end
        amplitudes_acc = V_mat \ ComplexF64.(signal)
    end

    # Pass 3: assemble ClusterResult vector
    clusters    = ClusterResult[]
    acc_counter = 0
    for r in raw
        re_mean, im_mean, re_std, im_std, im_rel_std,
            N, M_min, M_max, M_span, accepted, re_c, im_c, M_c = r
        amp = if accepted
            acc_counter += 1
            amplitudes_acc[acc_counter]
        else
            zero(ComplexF64)
        end
        push!(clusters, ClusterResult(re_mean, im_mean, re_std, im_std,
                                      im_rel_std, N, M_min, M_max, M_span,
                                      accepted, amp, re_c, im_c, M_c))
    end
    return clusters
end

# ── Internal helpers ──────────────────────────────────────────────────────────

# Simple Union-Find: merge any two poles within ε
function _cluster_2d_simple(poles::Vector{ComplexF64}, ε::Float64)
    n      = length(poles)
    parent = collect(1:n)
    find(x) = begin
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        x
    end
    for i in 1:n, j in i+1:n
        if abs(poles[i] - poles[j]) < ε
            ri, rj = find(i), find(j)
            ri != rj && (parent[ri] = rj)
        end
    end
    roots = find.(1:n)
    uq    = unique(roots)
    lmap  = Dict(r => k for (k, r) in enumerate(uq))
    return [lmap[r] for r in roots]
end

# Tagged Union-Find: merge poles only if they share the same tag (or are untagged)
function _cluster_2d_tagged(poles::Vector{ComplexF64}, tags::Vector{Int}, ε::Float64)
    n      = length(poles)
    parent = collect(1:n)
    find(x) = begin
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        x
    end
    for i in 1:n, j in i+1:n
        ti, tj = tags[i], tags[j]
        ti != 0 && tj != 0 && ti != tj && continue   # different tags → no merge
        if abs(poles[i] - poles[j]) < ε
            ri, rj = find(i), find(j)
            ri != rj && (parent[ri] = rj)
        end
    end
    roots = find.(1:n)
    uq    = unique(roots)
    lmap  = Dict(r => k for (k, r) in enumerate(uq))
    return [lmap[r] for r in roots]
end

# Assign each pole to the nearest known frequency within ε_assign (0 = untagged)
function _assign_tags(poles::Vector{ComplexF64},
                      ω_known::AbstractVector{<:Complex},
                      ε_assign::Float64)
    tags = zeros(Int, length(poles))
    for (i, p) in enumerate(poles)
        dists        = abs.(ω_known .- p)
        min_d, min_k = findmin(dists)
        min_d < ε_assign && (tags[i] = min_k)
    end
    return tags
end

# Trimmed mean and std (remove both ends by trim_frac fraction)
function _trimmed_stats(x::Vector{Float64}; trim_frac::Float64 = 0.25)
    n       = length(x)
    k       = round(Int, trim_frac * n)
    hi      = max(k + 1, n - k)
    trimmed = sort(x)[k+1:hi]
    isempty(trimmed) && (trimmed = x)
    return mean(trimmed), (length(trimmed) > 1 ? std(trimmed) : 0.0)
end
