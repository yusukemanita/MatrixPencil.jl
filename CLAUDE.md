# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`MatrixPencil.jl` is a Julia package implementing the Matrix Pencil Method (MPM) for extracting complex resonance frequencies from time-domain signals. Primary use case is quasi-normal mode (QNM) analysis in gravitational wave physics.

## Commands

```bash
# Run all tests
julia --project=. test/runtests.jl

# Run tests via Pkg
julia --project=. -e 'using Pkg; Pkg.test()'

# Load the package interactively
julia --project=. -e 'using MatrixPencil'
```

## Architecture

The pipeline runs: `stabilization_data` → `cluster_poles` → `classify_modes` → `print_mode_table`.

### Source files (`src/`)

| File | Purpose |
|------|---------|
| `MatrixPencil.jl` | Module entry point; exports and plotting stubs |
| `mpm.jl` | Core algorithm: `_mpm_basic` and `_mpm_fb` (Forward-Backward variant) |
| `stabilization.jl` | `stabilization_data` — sweeps model orders M, marks stable poles |
| `clustering.jl` | `cluster_poles` — Union-Find in the complex plane; simple and tagged variants |
| `classification.jl` | `ClusterResult`, `LabeledMode` structs; `classify_modes`, `print_mode_table` |
| `rational_filter.jl` | `rational_filter`, `signal_cleaning` — frequency-domain mode removal |

### Extension (`ext/`)

`MatrixPencilPlotsExt.jl` implements `plot_stabilization` and `plot_complex_plane` as a weak dependency on `Plots.jl`. The stubs are declared in the main module; implementations activate only when `using Plots` is called.

### Key design decisions

- **Forward-Backward MPM (`:fb`, default)**: averages forward and time-reversed Hankel matrices before SVD; filters to `Im(ω) < 0` (decaying modes only). Use `:basic` when you need all poles including growing ones.
- **Pencil parameter `L`**: typically `length(signal) ÷ 2`; passed explicitly to `matrix_pencil_method`.
- **Tagged clustering** (`ω_known` kwarg): prevents close modes (separation < `2ε_complex`) from merging. Assign each known frequency as a tag; poles with different tags are never merged.
- **Amplitude re-fitting**: `classify_modes` re-solves a joint Vandermonde least-squares problem over all matched clusters, replacing per-cluster amplitudes from `cluster_poles`.
- **FFT convention in `rational_filter`**: uses physics convention `H(ω) = (dt/√2π) ∫ h e^{iωt} dt` via `conj(fft(conj(h)))`.

### Acceptance criteria for clusters (defaults)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `min_count` | 3 | Minimum poles per cluster |
| `min_M_span` | 5 | Model-order range `M_max − M_min` |
| `im_rel_tol` | 0.15 | `std(Im)/|mean(Im)|` upper bound |

### Signal convention

Signals are modeled as sums of `exp(-i ω t)` with `Im(ω) < 0` for decaying modes. This matches the QNM convention in gravitational wave physics.
