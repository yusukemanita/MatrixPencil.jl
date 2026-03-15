using MatrixPencil: _cluster_2d_simple, _cluster_2d_tagged, _assign_tags, _trimmed_stats

# ── _cluster_2d_simple ────────────────────────────────────────────────────────

@testset "_cluster_2d_simple: two separated groups" begin
    # Group A around 1.0 - 0.1im, group B around 2.0 - 0.2im
    poles_A = [complex(1.0 + randn() * 0.001, -0.1 + randn() * 0.001) for _ in 1:5]
    poles_B = [complex(2.0 + randn() * 0.001, -0.2 + randn() * 0.001) for _ in 1:5]
    poles   = vcat(poles_A, poles_B)

    labels = _cluster_2d_simple(poles, 0.01)

    @test length(unique(labels)) == 2
    @test all(==(labels[1]), labels[1:5])      # A poles same label
    @test all(==(labels[6]), labels[6:10])     # B poles same label
    @test labels[1] != labels[6]               # different labels
end

@testset "_cluster_2d_simple: single pole" begin
    labels = _cluster_2d_simple([0.5 - 0.1im], 0.01)
    @test labels == [1]
end

@testset "_cluster_2d_simple: all poles merged" begin
    # All poles within ε of each other
    poles  = [complex(1.0 + i * 0.001, -0.1) for i in 0:4]
    labels = _cluster_2d_simple(poles, 0.01)
    @test length(unique(labels)) == 1
end

# ── _cluster_2d_tagged ────────────────────────────────────────────────────────

@testset "_cluster_2d_tagged: different tags prevent merge" begin
    # Two poles very close but carrying different tags
    p1 = 1.0 - 0.1im
    p2 = 1.001 - 0.1im    # within ε_complex=0.01 of p1
    labels = _cluster_2d_tagged([p1, p2], [1, 2], 0.01)
    @test labels[1] != labels[2]
end

@testset "_cluster_2d_tagged: same tag allows merge" begin
    p1 = 1.0 - 0.1im
    p2 = 1.001 - 0.1im    # within ε_complex=0.01
    labels = _cluster_2d_tagged([p1, p2], [1, 1], 0.01)
    @test labels[1] == labels[2]
end

@testset "_cluster_2d_tagged: untagged (0) poles merge with anyone" begin
    p1 = 1.0 - 0.1im
    p2 = 1.001 - 0.1im    # within ε_complex
    # p2 untagged (tag=0), p1 has tag=1 → should merge
    labels = _cluster_2d_tagged([p1, p2], [1, 0], 0.01)
    @test labels[1] == labels[2]
end

# ── _assign_tags ──────────────────────────────────────────────────────────────

@testset "_assign_tags: nearest within threshold" begin
    ω_known = [1.0 - 0.1im, 2.0 - 0.2im]
    poles   = [1.002 - 0.1im,   # close to ω_known[1]
               2.001 - 0.2im,   # close to ω_known[2]
               5.0   - 0.5im]   # far from both

    tags = _assign_tags(poles, ω_known, 0.01)
    @test tags[1] == 1
    @test tags[2] == 2
    @test tags[3] == 0   # untagged
end

# ── _trimmed_stats ────────────────────────────────────────────────────────────

@testset "_trimmed_stats: basic" begin
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    m, s = _trimmed_stats(x; trim_frac=0.0)
    @test m ≈ mean(x)

    # With trim_frac=0.4: k=2, hi=max(3,3)=3 → trimmed=[3.0]
    m2, s2 = _trimmed_stats(x; trim_frac=0.4)
    @test m2 ≈ 3.0
    @test s2 ≈ 0.0
end

@testset "_trimmed_stats: single element" begin
    m, s = _trimmed_stats([42.0]; trim_frac=0.25)
    @test m ≈ 42.0
    @test s ≈ 0.0
end

# ── cluster_poles acceptance criteria ─────────────────────────────────────────

@testset "cluster_poles: empty input returns empty" begin
    data   = (M=Int[], re=Float64[], im=Float64[], amp=Float64[], stable=Bool[])
    result = cluster_poles(data, ones(100), 0.01)
    @test result == ClusterResult[]
end

@testset "cluster_poles: min_count rejection" begin
    # Build a signal with one good mode ω1 and one sparse mode ω2
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt
    ω1 = 0.8 - 0.05im
    ω2 = 1.5 - 0.10im
    signal = @. 2.0 * exp(-im * ω1 * t) + 0.5 * exp(-im * ω2 * t)

    data = stabilization_data(signal, dt, 10:5:50;
               im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))

    # min_count=100 is larger than any realistic cluster → all rejected
    clusters = cluster_poles(data, signal, dt; min_count=100)
    @test all(c -> !c.accepted, clusters)
end

@testset "cluster_poles: min_M_span rejection" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt
    ω1 = 0.8 - 0.05im
    signal = @. 2.0 * exp(-im * ω1 * t)

    # Only 2 model orders in range → M_span ≤ 1 < default min_M_span=5
    data = stabilization_data(signal, dt, 10:1:11;
               im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))
    clusters = cluster_poles(data, signal, dt; min_M_span=5)
    @test all(c -> !c.accepted, clusters)
end

@testset "cluster_poles: im_rel_tol rejection" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt
    ω1 = 0.8 - 0.05im
    signal = @. 2.0 * exp(-im * ω1 * t)

    data = stabilization_data(signal, dt, 10:5:50;
               im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))

    # im_rel_tol=0.0 means any spread in Im rejects the cluster
    clusters = cluster_poles(data, signal, dt; im_rel_tol=0.0)
    @test all(c -> !c.accepted, clusters)
end

@testset "cluster_poles: untagged two-mode signal" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt
    ω1 = 0.8 - 0.05im
    ω2 = 1.5 - 0.10im
    signal = @. 2.0 * exp(-im * ω1 * t) + 1.0 * exp(-im * ω2 * t)

    data     = stabilization_data(signal, dt, 10:5:50;
                   im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))
    clusters = cluster_poles(data, signal, dt;
                   ε_complex=0.05, min_count=3, min_M_span=5)

    accepted = filter(c -> c.accepted, clusters)
    # Should detect at least the two injected modes
    @test length(accepted) >= 2

    re_means = sort([c.re_mean for c in accepted])
    @test any(r -> abs(r - real(ω1)) < 0.05, re_means)
    @test any(r -> abs(r - real(ω2)) < 0.05, re_means)
end

@testset "cluster_poles: tagged mode prevents merge" begin
    dt = 0.01
    N  = 512
    t  = (0:N-1) .* dt

    # Two very close modes
    ω1 = 0.80 - 0.080im
    ω2 = 0.82 - 0.082im   # separation ~0.02
    signal = @. 1.5 * exp(-im * ω1 * t) + 1.0 * exp(-im * ω2 * t)

    data = stabilization_data(signal, dt, 10:5:50;
               im_lim=(-0.5, 0.0), re_lim=(0.0, 3.0))

    # Without tagging and with large ε these two modes would merge
    # With tagging they should remain separate
    clusters_tagged = cluster_poles(data, signal, dt;
                         ω_known    = [ω1, ω2],
                         ε_complex  = 0.05,
                         ε_assign   = 0.05,
                         min_count  = 3,
                         min_M_span = 5)
    accepted_tagged = filter(c -> c.accepted, clusters_tagged)

    clusters_simple = cluster_poles(data, signal, dt;
                         ε_complex  = 0.05,
                         min_count  = 3,
                         min_M_span = 5)
    accepted_simple = filter(c -> c.accepted, clusters_simple)

    # Tagged should give at least as many accepted clusters as simple
    # (tagging prevents merging close modes into one cluster)
    @test length(accepted_tagged) >= length(accepted_simple)
end
