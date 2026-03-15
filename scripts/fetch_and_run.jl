"""
fetch_and_run.jl

Download one gravitational-wave waveform from Zenodo record 18953179
and analyse it with the MatrixPencil.jl pipeline.

Data source
-----------
Kubota & Motohashi (2026), "Numerical solutions to the Teukolsky equation
sourced by a delta-function source", Zenodo 10.5281/zenodo.18953179.
arXiv: 2509.06411

The archive (10.3 GB zip) is probed with HTTP Range requests so that only
the single chosen file (~1.2 MB compressed, ~3.6 MB uncompressed) is
downloaded.

Usage
-----
    julia --project=. scripts/fetch_and_run.jl

The script analyses the l=2, m=2, a=0 (Schwarzschild) waveform h(t) and
prints the recovered QNM table.  The dominant mode should match the known
fundamental Schwarzschild frequency

    Re(ω₂₂₀) M ≈ 0.3737,   Im(ω₂₂₀) M ≈ −0.0890.
"""

# ── Dependencies ───────────────────────────────────────────────────────────────
import Pkg
for pkg in ["CodecZlib", "Downloads", "CSV", "DataFrames"]
    if !haskey(Pkg.project().dependencies, pkg)
        Pkg.add(pkg; io=devnull)
    end
end

using MatrixPencil
using CodecZlib, Downloads
using CSV, DataFrames
using Printf

# ── Constants ──────────────────────────────────────────────────────────────────
const ZIP_URL = "https://zenodo.org/records/18953179/files/" *
                "delta_func_waveform.zip?download=1"

# Pre-computed byte offsets for h_waveform_l2m2a0.000.csv inside the zip.
# (Local file header at LOCAL_OFF; compressed data starts LOCAL_OFF + HEADER_SIZE)
const LOCAL_OFF   = 7_184_198_662
const HEADER_SIZE = 123          # 30-byte LFH + 61-byte filename + 32-byte extra
const COMP_SIZE   = 1_172_564    # bytes of deflate-compressed data

# ── Download one file from the zip via HTTP Range ──────────────────────────────
function download_zip_entry(url, local_offset, header_size, comp_size)
    data_start = local_offset + header_size
    data_end   = data_start + comp_size - 1
    buf = IOBuffer()
    Downloads.download(url, buf;
        headers = ["Range" => "bytes=$data_start-$data_end"])
    return take!(buf)
end

println("Downloading h_waveform_l2m2a0.000.csv from Zenodo...")
compressed = download_zip_entry(ZIP_URL, LOCAL_OFF, HEADER_SIZE, COMP_SIZE)
println("  Downloaded $(length(compressed)) compressed bytes")

raw_csv = transcode(DeflateDecompressor, compressed)
println("  Decompressed to $(length(raw_csv)) bytes")

# ── Parse CSV ─────────────────────────────────────────────────────────────────
df = CSV.read(IOBuffer(raw_csv), DataFrame)
# Columns: "t/M", "Re(h)", "Im(h)"
rename!(df, Symbol("t/M") => :t, Symbol("Re(h)") => :reh, Symbol("Im(h)") => :imh)

t_all = df.t
h_all = complex.(df.reh, df.imh)
dt    = t_all[2] - t_all[1]

println("\nWaveform loaded: $(length(h_all)) samples, dt = $dt M")
println("Time range: [$(t_all[1]), $(t_all[end])] M")

# ── Select ringdown window ─────────────────────────────────────────────────────
# Use t/M ∈ [20, 120] where the QNM ringdown dominates.
# Downsample by 10 (dt_eff = 0.1 M) to keep matrices manageable.
t_start, t_end = 20.0, 120.0
step           = 10
mask           = (t_all .>= t_start) .& (t_all .<= t_end)
t_win          = t_all[mask][1:step:end]
h_win          = h_all[mask][1:step:end]
dt_win         = t_win[2] - t_win[1]
N              = length(h_win)

println("\nRingdown window: t/M ∈ [$(t_win[1]), $(t_win[end])]  " *
        "N=$(N)  dt=$(dt_win) M")

# ── Run MPM pipeline ───────────────────────────────────────────────────────────
println("\nRunning stabilization diagram (M_range = 10:5:50)...")
data_stab = stabilization_data(h_win, dt_win, 10:5:50;
                im_lim = (-0.8, 0.0),
                re_lim = (-1.0, 1.0))

println("  Collected $(length(data_stab.re)) poles, " *
        "$(count(data_stab.stable)) stable")

println("Clustering stable poles...")
clusters = cluster_poles(data_stab, h_win, dt_win;
               ε_complex  = 0.02,
               ε_assign   = 0.02,
               min_count  = 3,
               min_M_span = 5,
               im_rel_tol = 0.15)

n_acc = count(c -> c.accepted, clusters)
println("  $(length(clusters)) clusters found, $n_acc accepted")

# ── Known Schwarzschild l=2,m=2 QNM reference (Berti et al. 2009) ─────────────
theory = Dict{String, ComplexF64}(
    "220"  =>  0.37367168 - 0.08896232im,
    "221"  =>  0.34671099 - 0.27391488im,
    "22-0" => -0.37367168 - 0.08896232im,
    "22-1" => -0.34671099 - 0.27391488im,
)

println("Classifying modes...")
modes = classify_modes(clusters, theory, h_win, dt_win; tol = 0.05)

# ── Print results ──────────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("  QNM analysis:  h_waveform_l2m2a0.000.csv  (Schwarzschild, a=0)")
println("=" ^ 70)
print_mode_table(modes)

println("\nReference frequencies (Berti et al. 2009):")
for (lbl, ω) in sort(collect(theory), by = p -> -real(p[2]))
    @printf("  %-6s  Re(ω)M = %8.5f   Im(ω)M = %8.5f\n",
            lbl, real(ω), imag(ω))
end
