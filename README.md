# MatrixPencil.jl

A Julia package for extracting complex resonance frequencies from time-domain signals using the **Matrix Pencil Method (MPM)**. Designed for quasi-normal mode (QNM) analysis in gravitational wave physics, but applicable to any system with damped sinusoidal components.

## Features

- **Matrix Pencil Method** — Hankel matrix SVD + Total Least Squares eigenvalue decomposition
- **Forward-Backward MPM** — Improved noise rejection via combined forward/backward Hankel matrices (Hua & Sarkar 1990)
- **Stabilization diagrams** — Track pole persistence across model orders to distinguish physical modes from numerical artifacts
- **Union-Find clustering** — Group stable poles in the complex plane; supports a tagged variant for near-degenerate modes
- **Theory matching** — Match extracted clusters to reference frequencies and re-fit amplitudes via Vandermonde least squares
- **Polynomial amplitude fitting** — Fit nearly-degenerate modes as `(A + B·t)·exp(-i·ω·t)` (Jordan-block form) via `poly_groups` in `classify_modes`
- **Optional plotting** — Stabilization diagram and complex-plane visualization via a weak Plots.jl dependency

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
| `matrix_pencil_method(y, L, dt, M; method=:fb)` | MPM with selectable variant: `:fb` (default, Forward-Backward, filters to decaying modes) or `:basic` (returns all poles) |


### Stabilization & Clustering

| Function | Description |
|----------|-------------|
| `stabilization_data(signal, dt, M_range; ...)` | Collect poles across model orders; returns `(M, re, im, amp, stable)` |
| `cluster_poles(data, signal, dt; ...)` | Union-Find clustering of stable poles; optional `ω_known` for tagged mode |

### Classification & Output

| Function | Description |
|----------|-------------|
| `classify_modes(clusters, theory_dict, signal, dt; tol, poly_groups)` | Match clusters to reference frequencies; re-fit all amplitudes jointly; optionally fit degenerate mode groups as `(A + B·t)·exp(-i·ω·t)` |
| `print_mode_table(modes)` | Pretty-print a summary table; shows `\|B\|` column automatically when any poly-group mode is present |

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
- `amplitude` — Re-fitted amplitude (constant term A)
- `amplitude_B` — Linear-in-t coefficient B; non-zero only for poly-group primaries
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
    δ_re = 1e-3,
    δ_im = 1e-3)

# Cluster with known 220 mode as anchor
ω220 = 0.7473 - 0.0890im
clusters = cluster_poles(data, signal, dt;
    ω_known   = [ω220],
    ε_complex = 0.02)

# Match to theory (standard fit)
theory = Dict("220" => ω220, "221" => 0.6934 - 0.2739im)
modes  = classify_modes(clusters, theory, signal, dt)
print_mode_table(modes)

# If 221 and the nonlinear "220+220-220*" mode are nearly degenerate,
# fit them as a single Jordan-block term (A + B·t)·exp(-i·ω₂₂₁·t):
theory2 = Dict("220" => ω220, "221" => 0.6934 - 0.2739im,
               "nl"  => 0.6930 - 0.2741im)
modes2  = classify_modes(clusters, theory2, signal, dt;
              poly_groups = [["221", "nl"]])
print_mode_table(modes2)   # shows |A| and |B| columns

# Visualize
plot_stabilization(data)
plot_complex_plane(data, modes)
```

## Polynomial Amplitude Fitting (Jordan-Block Form)

When two physical modes are nearly degenerate — closer in the complex plane than the MPM frequency resolution — they cannot be separated as independent exponentials. Their combined contribution in the ringdown behaves like a Jordan-block term:

$$s(t) = (A + B \cdot t)\, e^{-i\omega t}$$

Pass a `poly_groups` list to `classify_modes` to activate this fitting:

```julia
# Each group: first element = primary label (kept), rest = secondaries (suppressed)
modes = classify_modes(clusters, theory, signal, dt;
            poly_groups = [["221", "nl_mode"]])
```

- The **primary** (`"221"`) appears in the output with both `amplitude` (A) and `amplitude_B` (B).
- The **secondaries** (`"nl_mode"`) are absorbed into the primary and excluded from the output.
- A **single-element group** `["221"]` adds the B column without suppressing any other mode.
- All non-grouped modes use the standard exponential form (B = 0).
- `print_mode_table` automatically adds a `|B|` column when any mode has a non-zero B.

## Algorithm Details

The Matrix Pencil Method decomposes a signal of the form

$$y(t) = \sum_{k=1}^{M} A_k e^{i\omega_k t}$$

by constructing a Hankel data matrix, computing its SVD to determine the model order, solving a Total Least Squares eigenvalue problem, and returning the complex frequencies $\omega_k$ and amplitudes $A_k$.

The Forward-Backward variant (Hua & Sarkar 1990) averages the forward and time-reversed Hankel matrices before the eigendecomposition, significantly improving noise robustness.

**Stabilization**: Physical poles persist as model order increases; spurious poles do not. A pole at order $M$ is marked stable if a pole within tolerance `(tol_re, tol_im)` exists at order $M-1$.

### Union-Find Clustering

The clustering step groups stable poles that represent the same physical mode across different model orders. Two strategies are available.

#### Simple mode (no prior knowledge)

All pairs of stable poles $(z_i, z_j)$ are tested. If their complex-plane distance satisfies

$$|z_i - z_j| < \varepsilon_{\text{complex}}$$

The two poles are merged into the same cluster. This is a standard Union-Find with path-compression: each pole starts in its own set, and union operations progressively merge sets until no further merges are possible. The result is a partition of all stable poles into disjoint clusters.

#### Tagged mode (`ω_known` provided)

Used when two physical modes lie closer together than $2\varepsilon_{\text{complex}}$ (e.g. the QNM pair 220 / 221 in black-hole ringdown, which can be separated by $\lesssim 0.01$ in the complex plane).

1. **Tag assignment** — Each stable pole is assigned to the nearest known frequency within `ε_assign`. Poles that fall outside `ε_assign` of every known frequency remain untagged (tag = 0).
2. **Conditional merge** — Two poles are merged only if they are within `ε_complex` **and** do not carry different non-zero tags. Concretely, a pair $(i, j)$ is skipped when both are tagged and their tags differ. Untagged poles can merge freely with any other pole.

This prevents poles belonging to distinct known modes from bleeding into the same cluster, while still grouping numerical scatter around each mode.

#### Acceptance criteria

After clustering, each cluster is accepted only if **all three** conditions hold:

| Criterion | Parameter | Default | Meaning |
|-----------|-----------|---------|---------|
| Minimum pole count | `min_count` | 3 | Each cluster must contain at least this many poles |
| Minimum model-order span | `min_M_span` | 5 | Poles must appear across at least this range of model orders $M_{\max} - M_{\min}$ |
| Relative Im stability | `im_rel_tol` | 0.15 | Trimmed standard deviation of the imaginary part divided by its absolute mean must be below this threshold |

The first two criteria reject isolated or short-lived artifacts. The third rejects clusters whose imaginary part (decay rate) is too scattered, indicating a numerical rather than physical pole.

#### Cluster statistics

Accepted cluster frequencies are reported as trimmed means and standard deviations (default: discard the outer 25% of values at each end) to suppress the influence of outlier poles at the edges of the model-order range.

## References

- Hua, Y., & Sarkar, T. K. (1990). *Matrix pencil method for estimating parameters of exponentially damped/undamped sinusoids in noise*. IEEE Transactions on Acoustics, Speech, and Signal Processing, 38(5), 814–824.
- Sarkar, T. K., & Pereira, O. (1995). *Using the matrix pencil method to estimate the parameters of a sum of complex exponentials*. IEEE Antennas and Propagation Magazine, 37(1), 48–55.

## License

MIT