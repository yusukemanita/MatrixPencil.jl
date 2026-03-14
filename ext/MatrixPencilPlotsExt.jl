module MatrixPencilPlotsExt

using MatrixPencil
import MatrixPencil: plot_stabilization, plot_complex_plane
using Plots
using LaTeXStrings

# ── Stabilization diagram: Re(ω) vs model order M ────────────────────────────

"""
    plot_stabilization(data; omegas_ref, re_lim, im_lim, title_str, ms)

Stabilization diagram: x = Re(ω), y = model order M.

Stable poles are coloured by Im(ω) (viridis). Unstable poles are shown in
grey. Optional `omegas_ref` adds faint vertical reference lines.
"""
function plot_stabilization(data;
                            omegas_ref = ComplexF64[],
                            re_lim     = (-2.0, 2.0),
                            im_lim     = (-0.5, 0.0),
                            title      = "",
                            dpi        = 100,
                            ms         = 3,
                            size       = (700, 500),
                            fontfamily = "Times New Roman",
                            legend     = :topright,)
    nm, sm = .!data.stable, data.stable

    p = scatter(data.re[nm], data.M[nm];
        ms = ms, color = :gray, alpha = 0.2, markerstrokewidth = 0,
        label = "",
        xlabel = L"\mathrm{Re}(\omega)",
        ylabel = L"M",
        title  = title,
        frame  = :box, xlim = re_lim,
        legend = legend,
        tickfontsize   = 11,
        guidefontsize  = 13,
        legendfontsize = 10,
        titlefontsize  = 13,
        fontofamily    = fontfamily,
        size           = size,
        dpi            = dpi,
        left_margin    = 5Plots.mm,
        bottom_margin  = 5Plots.mm,
        top_margin     = 5Plots.mm,
        right_margin   = 15Plots.mm,
        legend_background_color = RGBA(1, 1, 1, 0.7))

    if any(sm)
        idx_s = sortperm(data.im[sm], rev=true)
        scatter!(p, data.re[sm][idx_s], data.M[sm][idx_s];
            zcolor = data.im[sm][idx_s], c = :viridis, clims = im_lim,
            ms = ms + 1, markerstrokewidth = 0,
            label = "Stable", colorbar_title = "")
    end

    for ω_ref in omegas_ref
        vline!(p, [real(ω_ref)]; color = :gray, alpha = 0.15, lw = 0.5, label = "")
    end

    return p
end

# ── Complex-plane overview: Re(ω) vs Im(ω) ───────────────────────────────────

"""
    plot_complex_plane(data, modes; theory_dict, re_lim, im_lim, title_str)

Complex ω-plane plot with four layers:
1. Unstable poles (grey)
2. Stable poles coloured by log₁₀|A| (viridis)
3. Reference frequencies from `theory_dict` (red ×)
4. Accepted cluster means with labels (black ●)
"""
function plot_complex_plane(data, modes;
                            theory_dict = Dict{String, ComplexF64}(),
                            re_lim      = (-1.5, 1.5),
                            im_lim      = (-0.5, 0.0),
                            title      = "",
                            dpi        = 100,
                            ms         = 3,
                            size       = (700, 500),
                            fontfamily = "Times New Roman",
                            legend     = :topright,)
    nm     = .!data.stable
    sm     =  data.stable
    ms_all = MatrixPencil._amp_to_ms(data.amp)

    # Layer 1: unstable poles
    p = scatter(data.re[nm], data.im[nm];
        ms = ms_all[nm], color = :gray, alpha = 0.15, markerstrokewidth = 0,
        label = "",
        xlabel = L"\mathrm{Re}(\omega)",
        ylabel = L"\mathrm{Im}(\omega)",
        title  = title,
        frame  = :box, xlim = re_lim, ylim = im_lim,
        legend = legend,
        fontfamily     = fontfamily,
        tickfontsize   = 11,
        guidefontsize  = 13,
        legendfontsize = 10,
        titlefontsize  = 13,
        size           = size,
        dpi            = dpi,
        left_margin    = 5Plots.mm,
        bottom_margin  = 5Plots.mm,
        top_margin     = 5Plots.mm,
        right_margin   = 15Plots.mm,
        legend_background_color = RGBA(1, 1, 1, 0.7))

    # Layer 2: stable poles (coloured by log amplitude)
    if any(sm)
        scatter!(p, data.re[sm], data.im[sm];
            ms = ms_all[sm],
            zcolor = log10.(data.amp[sm] .+ 1e-30),
            c = :viridis, markerstrokewidth = 0,
            label = "", colorbar_title = "")
    end

    # Layer 3: reference frequencies (within plot limits)
    if !isempty(theory_dict)
        ref_re = Float64[real(ω) for ω in values(theory_dict)]
        ref_im = Float64[imag(ω) for ω in values(theory_dict)]
        mask   = (re_lim[1] .< ref_re .< re_lim[2]) .&
                 (im_lim[1] .< ref_im .< im_lim[2])
        if any(mask)
            scatter!(p, ref_re[mask], ref_im[mask];
                marker = :xcross, ms = 6, color = :red, alpha = 0.7,
                markerstrokewidth = 1.5, label = "Theory")
        end
    end

    # Layer 4: accepted cluster means with labels
    dy = (im_lim[2] - im_lim[1]) * 0.05
    for m in modes
        ω = m.omega_mpm
        re_lim[1] < real(ω) < re_lim[2] || continue
        im_lim[1] < imag(ω) < im_lim[2] || continue
        scatter!(p, [real(ω)], [imag(ω)];
            ms = 6, color = :black, markerstrokewidth = 0,
            marker = :circle, label = "")
        annotate!(p, real(ω), imag(ω) - dy,
                  text(m.label, 9, :black, :center, "Times New Roman"))
    end

    return p
end

end # module MatrixPencilPlotsExt
