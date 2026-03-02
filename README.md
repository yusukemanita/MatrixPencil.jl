# MatrixPencil.jl

A Julia package for extracting complex resonance frequencies from time-domain signals using the **Matrix Pencil Method (MPM)**. Designed for quasi-normal mode (QNM) analysis in gravitational wave physics, but applicable to any system with damped sinusoidal components.

## Features

- **Matrix Pencil Method** — Hankel matrix SVD + Total Least Squares eigenvalue decomposition
- **Forward-Backward MPM** — Improved noise rejection via combined forward/backward Hankel matrices (Hua & Sarkar 1990)
- **Stabilization diagrams** — Track pole persistence across model orders to distinguish physical modes from numerical artifacts
- **Union-Find clustering** — Group stable poles in the complex plane; supports a tagged variant for near-degenerate modes
- **Theory matching** — Match extracted clusters to reference frequencies and re-fit amplitudes via Vandermonde least squares
- **Rational filtering** — FFT-based preprocessing to remove known frequency components before extraction
- **Optional plotting** — Stabilization diagram and complex-plane visualization via a weak Plots.jl dependency

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/yourusername/MatrixPencil.jl")
```

Requires Julia ≥ 1.9.

## Quick Start

```julia
using MatrixPencil

# Synthetic signal: two damped sinusoids
dt = 0.1
t  = 0:dt:100
ω1 = 1.0 - 0.1im
ω2 = 2.0 - 0.2im
y  = exp.(im .* ω1 .* t) .+ 0.5 .* exp.(im .* ω2 .* t)

# 1. Extract poles across model orders
data = stabilization_data(y, dt, 5:2:40)

# 2. Cluster stable poles
clusters = cluster_poles(data, y, dt)

# 3. (Optional) match clusters to known frequencies
theory = Dict("mode_A" => ω1, "mode_B" => ω2)
modes  = classify_modes(clusters, theory, y, dt)
print_mode_table(modes)
```

## Typical Workflow

```
signal  →  stabilization_data()  →  cluster_poles()  →  classify_modes()
                                                              ↓
                                                      print_mode_table()
                                                      plot_stabilization()
                                                      plot_complex_plane()
```

## API Reference

### Core Algorithms

| Function | Description |
|----------|-------------|
| `matrix_pencil_method(y, L, dt, M)` | Basic MPM; returns all poles including growing modes |
| `matrix_pencil_method_fb(y, L, dt, M)` | Forward-Backward MPM; filters to decaying modes (`Im(ω) < 0`) |

### Signal Preprocessing

| Function | Description |
|----------|-------------|
| `rational_filter(h, fs, ω0s)` | Remove specific frequency components in the frequency domain |
| `signal_cleaning(h, fs, ω0s)` | Same as above with 10× zero-padding to suppress edge artifacts |

### Stabilization & Clustering

| Function | Description |
|----------|-------------|
| `stabilization_data(signal, dt, M_range; ...)` | Collect poles across model orders; returns `(M, re, im, amp, stable)` |
| `cluster_poles(data, signal, dt; ...)` | Union-Find clustering of stable poles; optional `ω_known` for tagged mode |

### Classification & Output

| Function | Description |
|----------|-------------|
| `classify_modes(clusters, theory_dict, signal, dt)` | Match clusters to reference frequencies; re-fit all amplitudes jointly |
| `print_mode_table(modes)` | Pretty-print a summary table of labeled modes |

### Plotting (requires `Plots.jl`)

| Function | Description |
|----------|-------------|
| `plot_stabilization(data; ...)` | Stabilization diagram: Re(ω) vs model order M |
| `plot_complex_plane(data, modes; ...)` | Complex ω-plane with clustered poles and labeled modes |

### Types

**`ClusterResult`** — Output of `cluster_poles`:
- `re_mean`, `im_mean`, `re_std`, `im_std`, `im_rel_std` — Pole statistics
- `N`, `M_min`, `M_max`, `M_span` — Pole count and model order range
- `accepted` — Whether the cluster passed acceptance criteria
- `amplitude` — Vandermonde least-squares amplitude
- `re_vals`, `im_vals`, `M_vals` — Raw pole data for plotting

**`LabeledMode`** — Output of `classify_modes`:
- `label` — Theory dictionary key or `"unknown"`
- `omega_mpm` — MPM frequency estimate
- `omega_ref` — Reference frequency from theory dictionary
- `amplitude` — Re-fitted amplitude
- `re_std`, `im_std` — Frequency uncertainties
- `cluster` — Originating `ClusterResult`

## Example: Gravitational Wave QNM Analysis

```julia
using MatrixPencil, Plots

# Load ringdown signal
signal = load_signal("ringdown.dat")
dt     = 0.1

# Collect stabilization data
data = stabilization_data(signal, dt, 4:2:50;
    L_frac = 0.5,
    tol_re  = 1e-3,
    tol_im  = 1e-3)

# Cluster with known 220 mode as anchor
ω220 = 0.7473 - 0.0890im
clusters = cluster_poles(data, signal, dt;
    ω_known   = [ω220],
    ε_complex = 0.02)

# Match to theory
theory = Dict("220" => ω220, "221" => 0.6934 - 0.2739im)
modes  = classify_modes(clusters, theory, signal, dt)
print_mode_table(modes)

# Visualize
plot_stabilization(data)
plot_complex_plane(data, modes)
```

## Algorithm Details

The Matrix Pencil Method decomposes a signal of the form

$$y(t) = \sum_{k=1}^{M} A_k \, e^{i\omega_k t}$$

by constructing a Hankel data matrix, computing its SVD to determine the model order, solving a Total Least Squares eigenvalue problem, and returning the complex frequencies $\omega_k$ and amplitudes $A_k$.

The Forward-Backward variant (Hua & Sarkar 1990) averages the forward and time-reversed Hankel matrices before the eigendecomposition, significantly improving noise robustness.

**Stabilization**: Physical poles persist as model order increases; spurious poles do not. A pole at order $M$ is marked stable if a pole within tolerance `(tol_re, tol_im)` exists at order $M-1$.

**Clustering**: Union-Find merges poles within complex-plane distance `ε_complex`. The tagged variant assigns each pole to the nearest known frequency first, then only merges poles sharing the same tag — preventing aliasing between near-degenerate modes.

## References

- Hua, Y., & Sarkar, T. K. (1990). *Matrix pencil method for estimating parameters of exponentially damped/undamped sinusoids in noise*. IEEE Transactions on Acoustics, Speech, and Signal Processing, 38(5), 814–824.
- Sarkar, T. K., & Pereira, O. (1995). *Using the matrix pencil method to estimate the parameters of a sum of complex exponentials*. IEEE Antennas and Propagation Magazine, 37(1), 48–55.

## License

MIT
