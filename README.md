# KrylovComplexity.jl

A Julia package for computing Krylov complexity and Krylov entropy of various quantum systems.

Given a configuration, the package generates the Hamiltonian $H$ and an initial operator $O$ of the system. Then, it computes the Lanczos $b$-coefficients of the Liouvillian superoperator $\mathcal{L} = [H, \cdot]$, from which it derives the Krylov complexity $C_K(t)$ and Krylov entropy $S_K(t)$ [[Parker et al., 2019]](https://arxiv.org/abs/1812.02666).

Currently, the following systems are supported: Ising, XXZ, XYZ, cSYK, GOE and GUE matrices.

The algorithm uses a constraint eigenspace representation to remove degenerate eigenvalues of the Liouvillian operator accounting for the projection of the initial operator (implemented based on [[Eckseler et al., 2025]](https://arxiv.org/abs/2505.02670)).

---

## Installation

From the Julia package manager, with the repository cloned locally:

```julia
using Pkg
Pkg.develop(path="KrylovComplexity.jl")
```

Or activate the package environment directly:

```julia
using Pkg
Pkg.activate("KrylovComplexity.jl")
Pkg.instantiate()
```

---

## Quick start

The simplest entry point is `simulate()` with a YAML config. For more control, call `build_system()` and `lanczos()` directly. Examples are provided in `examples/`.

### From a YAML config file

```bash
julia --project=KrylovComplexity.jl -e '
    using KrylovComplexity
    result = simulate("examples/ising.yaml")
    println(result)
'
```

### From Julia code

```julia
using KrylovComplexity

# Build the system
H, O = build_system(Dict("model" => "ising", "N" => 6,
                          "hx" => 1.05, "hz" => 0.5,
                          "precision_build" => 1024))

# Run Lanczos
b, Kdim, reortho_count = lanczos(H, O; precision=:double, reortho=:pro)

# Compute K(t) and S(t)
t, K, S = krylov_complexity_entropy(b; time_max=1000.0)
```

### simulate() wrapper

```julia
result = simulate(Dict(
    "model"      => "csyk",
    "N"          => 8,
    "precision"  => "double",
    "reortho"    => "pro",
    "time_max"   => 3000.0,
    "output_dir" => "results/csyk_n8/",
))
```

YAML example configs for all supported models are in `examples/`.

---

## Output

Both `simulate()` and the lower-level functions return a `KCResult` struct:

| Field | Type | Description |
|---|---|---|
| `b` | `Vector{T}` | Lanczos $b$-coefficients at working precision `T` |
| `t` | `Vector{Float64}` | Time grid for $C_K(t)$ and $S_K(t)$ |
| `K` | `Vector{Float64}` | Krylov complexity $C_K(t)$ |
| `S` | `Vector{Float64}` | Krylov entropy $S_K(t)$ |
| `Kdim` | `Int` | Krylov dimension (`> 0` = converged; `-1` = `max_iter` reached) |
| `reortho_count` | `Int` | Total number of reorthogonalization passes |
| `wall_time` | `Float64` | Elapsed wall time in seconds |

When `save: true`, the full `KCResult` is written to `output_dir/result.jld2` and can be reloaded with `load_result(path)`. When `save_b: true`, only `b` is saved to `output_dir/b.jld2` (reloaded with `load_b(path)`), preserving the working precision including BigFloat.

---

## Models

### Ising chain

$$H = J \sum_{i=1}^{N-1} \sigma^z_i \sigma^z_{i+1} + \sum_i h_x  \sigma^x_i + h_z \sigma^z_i$$

$$O = \sigma^\alpha_{m} \quad (\alpha \in \{x, y, z\})$$

- Where $\sigma_i$ are the Pauli matrices and $m$ can be chosen but is one by default.
- Boundary conditions: `bc`: `pbc` (periodic) or `obc` (open)
- Integrable at $h_z = 0$; chaotic for generic $h_z \neq 0$
- No symmetry reduction is implemented; the full 2N2^N
2N-dimensional Hilbert space is used.

### XYZ chain

$$H = \sum_{i=1}^{N-1} J_x \sigma^x_i \sigma^x_{i+1} + J_y \sigma^y_i \sigma^y_{i+1} + J_z  \sigma^z_i \sigma^z_{i+1} + h_z \sum_i^N \sigma^z_i + c\, \sigma^z_{N/2}$$

$$O = \sigma^\alpha_{m}$$

- Where $m$ can be chosen but is by default $N/2$.
- `bc`: `pbc` or `obc`
- Integrable at $c = 0$; $c \neq 0$ breaks integrability
- No symmetry sectors implemented

### XXZ chain (symmetry-reduced)

$$H = -\frac{J}{2} \sum_{\langle i,j \rangle} \left[ \sigma^x_i \sigma^x_j + \sigma^y_i \sigma^y_j + J_{zz}\!\left(\sigma^z_i \sigma^z_j - d\right) \right] + c\, \sigma^z_{(N+1)/2}$$

$$O = \left(\sigma^z_{m} + \sigma^z_{N+1-m}\right) \text{ projected to symmetry sector}$$

Symmetries removed before Lanczos:
- Where $m$ can be chosen but is by default $N/2$.
- **Total Magnetization symmetry**: projection to total-$S^z$ sector with $M$ spin-up sites (default $M = N/2$)
- **Parity Symmetry**: projection to parity sector $p = \pm 1$ (default $p=+1$)

The operator is the symmetrized two-site combination and is made traceless, following [arXiv:2207.07701].

### GOE / GUE random matrices
Gaussian Orthogonal and Unitary Ensemble matrices are well-studied objects in Random Matrix Theory.
GOE: Symmetric matrices: 
$$H = (A^T + A)/2$$
where $A_{ij} \sim \mathcal{N}(0,1)$. $O$ follows the same distribution.

GUE: Hermitian matrices: 
$$H_{ii} \sim \mathcal{N}(0,1)$$
$$Re(H_{ij}) \sim \mathcal{N}(0,1/2)$$
$$Im(H_{ij}) \sim \mathcal{N}(0,1/2)$$

where $i\neq j$. $O$ follows the same distribution.

### Complex SYK (cSYK)


$$H = \sum_{ijkl} J_{ij;kl} c_i^\dagger c_j^\dagger c_k c_l$$
$$O = c_{N-1}^\dagger c_{N} + c_N^\dagger c_{N-1}$$

Where $i,j,k,l \in \{1,2,\ldots,N\}$ and $c_i$ are complex fermions. $J_{ij;kl} \sim \mathcal{N}(0, 6J^2/N^3)$ are random couplings that satisfy the following conditions 
$$J_{ij;kl} = - J_{ji;kl} = - J_{ij;lk} = J^*_{kl;ij}$$ 


- **Particle Number Symmetry**: we work in the symmetry sector defined by the particle number (default $N/2$)
- $J_{ijkl}$ are drawn from a complex Gaussian at build time — fix `seed` for reproducibility
- Without a fixed seed the Hamiltonian differs every run

---

## Precision

| Key | Type | Decimal digits | Recommended use |
|---|---|---|---|
| `f64` | Float64 | 15 | quick checks, small $N$ |
| `double` | Double64 | 31 | default for production runs |
| integer (e.g. `256`) | BigFloat | $\approx p / 3.32$ | high-precision validation |

`precision_build` controls the type used to build $H$ and $O$ (including diagonalization). Default on `f64`, to ensure reproducibility of the results between different runs.
`precision` controls the working precision of the Lanczos algorithm.

## Reorthogonalization

| Symbol | Method | Extra Cost |
|---|---|---|
| `:none` | none | $-$ |
| `:fro` | full single-pass Gram–Schmidt | $O(K^3)$ |
| `:dro` | double-pass Gram–Schmidt | $2 \times O(2K^3)$ |
| `:pro` | partial adaptive (W-matrix recurrence) | typically $\ll$ `:fro` |
| `:dpro` | double-pass PRO | typically $\ll$ `:fro` |

`:pro` is the recommended default.

---

## Parameters

All parameters can be set in a YAML file or passed as keys in a `Dict` to `simulate()` or `build_system()`. Let $\varepsilon_M$ denote the machine precision.

### Global parameters

| Key | Default | Description |
|---|---|---|
| `model` | — | Model name: `ising`, `xyz`, `xxz`, `goe`, `gue`, `csyk` |
| `N` | — | System size (number of sites / modes) |
| `precision_build` | `"f64"` | Precision for building $H$ and $O$: `f64`, `double`, `bigfloat`, or integer bits |
| `precision` | `"f64"` | Lanczos working precision: `f64`, `double`, `bigfloat`, or integer bits |
| `reortho` | `"pro"` | Reorthogonalization method: `none`, `fro`, `dro`, `pro`, `dpro` |
| `remove_degeneracies` | `true` | Collapse degenerate Liouvillian frequency blocks and drop zero-weight projection of the operator on Liouvillian eigenspace |
| `max_iter` | $K_\text{eff} + 10$ | Maximum number of Lanczos iterations |
| `eps_factor` | $\times 10^{-4}$ | Convergence threshold factor; stop when $b_n <$ eps\_factor $\times \sqrt{\varepsilon_M}$ |
| `tol_factor` | $\times 10^{-4}$ | Degeneracy tolerance factor for frequency block collapsing: $\texttt{tol\_factor} \times \sqrt{\varepsilon_M}$ |
| `eps_reortho_factor` | $\times 1.0$ | PRO trigger; reorthogonalize when $\max{\|W_{mn}\|} >$ eps\_reortho\_factor $\times \sqrt{\varepsilon_M}$ |
| `time_max` | `1000.0` | Maximum integration time for $C_K(t)$ |
| `dt_factor` | `0.1` | Initial RK4 step $\Delta t = \texttt{dt\_factor} / \max(b)$ |
| `kc_out_resolution` | `2000` | Number of output time points in $C_K(t)$ and $S_K(t)$ |
| `output_dir` | `"results/"` | Directory for output files |
| `save` | `false` | Save full `KCResult` (b, t, K, S, metadata) to `output_dir/result.jld2` |
| `save_b` | `false` | Save only $b$ to `output_dir/b.jld2` (ignored when `save: true`) |

### Ising specific parameters

| Key | Default | Description |
|---|---|---|
| `J` | `1.0` | Ising coupling $J$ |
| `hx` | `1.0` | Transverse field $h_x$ |
| `hz` | `0.0` | Longitudinal field $h_z$; nonzero breaks integrability |
| `bc` | `"pbc"` | Boundary conditions: `pbc` or `obc` |
| `oper` | `"Sz"` | Initial operator: `Sx`, `Sy`, `Sz` |
| `oper_site` | `1` | Site index of the initial operator (1-based) |

### XYZ specific parameters

| Key | Default | Description |
|---|---|---|
| `Jx` | `1.0` | $x$-coupling |
| `Jy` | `1.0` | $y$-coupling |
| `Jz` | `1.0` | $z$-coupling; `Jx=Jy=Jz` gives the Heisenberg model |
| `hz` | `0.0` | Uniform longitudinal field |
| `c` | `0.0` | Single-site symmetry-breaking field at site $N/2$; nonzero breaks integrability |
| `bc` | `"pbc"` | Boundary conditions: `pbc` or `obc` |
| `oper` | `"Sz"` | Initial operator: `Sx`, `Sy`, `Sz` |
| `oper_site` | `1` | Site index of the initial operator (1-based) |

### XXZ specific parameters

| Key | Default | Description |
|---|---|---|
| `J` | `1.0` | Heisenberg coupling $J$ |
| `Jzz` | `1.0` | Anisotropy $J_{zz}$ |
| `d` | `1.0` | Offset $d$ in the $J_{zz}(\sigma^z_i \sigma^z_j - d)$ term |
| `c` | `0.0` | Single-site integrability-breaking field at the middle site |
| `magnetization` | `-1` | Magnetization sector $M$ (number of spin-up); `-1` uses $N/2$ |
| `parity` | `1` | Parity sector: `1`, `-1`, or `0` (no parity projection) |
| `oper_site` | `-1` | Operator site index; `-1` uses $N/2$ |
| `bc` | `"obc"` | Boundary conditions: `pbc` or `obc` |

### GOE / GUE specific parameters

| Key | Default | Description |
|---|---|---|
| `seed` | `null` | RNG seed; `null` for a random draw each run |
| `sparsity` | `1.0` | Fraction of nonzero entries (1.0 = dense) |

### cSYK specific parameters

| Key | Default | Description |
|---|---|---|
| `J` | `1.0` | Coupling scale; $\mathrm{Var}(|J_{ijkl}|^2) = J^2 \cdot 3!/N^3$ |
| `Q` | `-1` | Charge sector; `-1` uses $N/2$ |
| `seed` | `null` | RNG seed for $J_{ijkl}$; `null` for a different draw each run |
| `site1` | `-1` | First site of the operator; `-1` uses $N-1$ |
| `site2` | `-1` | Second site of the operator; `-1` uses $N$ |

---

## Code structure

```
src/
  KrylovComplexity.jl   module entry point, exports
  types.jl              KCResult struct
  operators.jl          build_system and all Hamiltonian/operator constructors
  lanczos.jl            Lanczos algorithm, reorthogonalization, eigenbasis build
  complexity.jl         RK4 integration of chain Schrödinger equation
  config.jl             YAML/Dict parsing, simulate() wrapper
  io.jl                 save/load KCResult and b-coefficients (JLD2)

examples/
  ising.yaml                                  Ising chain example config
  xyz.yaml                                    XYZ chain example config
  xxz.yaml                                    XXZ chain example config
  gue.yaml                                    GUE random matrix example config
  csyk.yaml                                   cSYK example config
  viz_helper.jl                               CairoMakie plotting helpers (plot_b, plot_complexity)
  csyk_convergence_arxiv_250706286.jl         cSYK N=8: CER vs nCER across 3 precisions

test/
  runtests.jl
  test_operators.jl
  test_lanczos.jl
  test_pro.jl
  test_complexity.jl
  test_config_io.jl
  test_consistency_mathematica.jl
  test_consistency_xxz.jl
  test_consistency_csyk.jl
  data/                 reference b-coefficient files for consistency tests
```

---

## Running tests

```bash
cd KrylovComplexity.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Or from the REPL:

```julia
using Pkg
Pkg.activate("KrylovComplexity.jl")
Pkg.test()
```

### Test coverage and reference data

Several tests validate $b$-coefficients against independent reference values from
[Craps et al.](http://arxiv.org/abs/2409.15666) (Mathematica implementation)
and other validated simulations by the authors.

| Test file | What it validates |
|---|---|
| `test_operators.jl` | Matrix dimensions, Hermiticity, Frobenius norm, sector sizes |
| `test_lanczos.jl` | $b_n > 0$, Krylov dimension, tridiagonal eigenvalue inclusion, FRO/DRO consistency |
| `test_pro.jl` | PRO vs FRO agreement, reortho count ordering |
| `test_complexity.jl` | $C_K(0) = S_K(0) = 0$, physical bounds, plateau $C_K \approx (K_\text{dim}-1)/2$ |
| `test_config_io.jl` | YAML loading, `simulate()` parameter wiring, save/load round-trips |
| `test_consistency_mathematica.jl` | Ising and XYZ $b$-coefficients against Mathematica reference ([Craps et al., 2024](http://arxiv.org/abs/2409.15666))|
| `test_consistency_xxz.jl` | XXZ N=8 $b$-coefficients against validated research code |
| `test_consistency_csyk.jl` | cSYK N=8 against a stored reference; **may fail** for unlucky draws of random $J_{ijkl}$ when no `seed` is fixed. This is expected and does not indicate a code bug |


## Logging

The package emits `@debug` messages every 100 Lanczos steps (step index, current $b_n$, reortho count). They are suppressed by default. To enable them:

```bash
# from the terminal — enable debug output for this package only
JULIA_DEBUG=KrylovComplexity julia --project=. script.jl
```

```julia
# from the REPL
using Logging
global_logger(ConsoleLogger(stderr, Logging.Debug))
```

To re-suppress after enabling in the REPL:

```julia
global_logger(ConsoleLogger(stderr, Logging.Info))
```

---

## References

- Parker, Cao, Avdoshkin, Scaffidi & Altman (2019) — *A Universal Operator Growth Hypothesis*, Phys. Rev. X **9**, 041017. [DOI](http://dx.doi.org/10.1103/PhysRevX.9.041017) | [arXiv](https://arxiv.org/abs/1812.02666)

- Rabinovici, Sánchez-Garrido, Shir & Sonner (2022) — *Krylov complexity from integrability to chaos*, JHEP **07** (2022) 151. [DOI](http://dx.doi.org/10.1007/JHEP07(2022)151) | [arXiv](https://arxiv.org/abs/2207.07701)

- Eckseler, Pieper & Schnack (2025) — *Escaping the Krylov space during the finite-precision Lanczos algorithm*, Phys. Rev. E **112**, 025302. [DOI](http://dx.doi.org/10.1103/k94p-vls8) | [arXiv](https://arxiv.org/abs/2505.02670)

- Craps, Evnin & Pascuzzi (2025) — *Multiseed Krylov Complexity*, Phys. Rev. Lett. **134**, 050402. [DOI](http://dx.doi.org/10.1103/physrevlett.134.050402) | [arXiv](http://arxiv.org/abs/2409.15666)
