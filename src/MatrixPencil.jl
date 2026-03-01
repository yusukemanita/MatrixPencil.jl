"""
    MatrixPencil

Matrix Pencil Method (MPM) for identifying complex frequencies from
damped sinusoidal signals, with stabilization diagram analysis and
optional QNM mode classification.

## Typical workflow

```julia
using MatrixPencil

# 1. Collect poles across model orders
data = stabilization_data(signal, dt, 50:10:200)

# 2a. Cluster without prior knowledge (generic)
clusters = cluster_poles(data, signal, dt)

# 2b. Cluster with known frequencies (prevents close-mode merging)
clusters = cluster_poles(data, signal, dt; ω_known = theory_freqs)

# 3. Match to reference frequencies (optional)
modes = classify_modes(clusters, theory_dict, signal, dt)
print_mode_table(modes)

# 4. Plotting (requires Plots.jl)
using Plots, LaTeXStrings
plot_stabilization(data)            # stabilization diagram
plot_complex_plane(data, modes)     # complex-plane overview
```
"""
module MatrixPencil

using LinearAlgebra
using Statistics
using Printf
using FFTW

include("mpm.jl")
include("rational_filter.jl")
include("stabilization.jl")
include("classification.jl")
include("clustering.jl")

export matrix_pencil_method, matrix_pencil_method_fb
export rational_filter, signal_cleaning
export stabilization_data
export ClusterResult, LabeledMode
export cluster_poles
export classify_modes, print_mode_table

# Plotting stubs — implemented in MatrixPencilPlotsExt when Plots.jl is loaded
function plot_stabilization end
function plot_complex_plane  end
export plot_stabilization, plot_complex_plane

end # module MatrixPencil
