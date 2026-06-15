# KrylovComplexity.jl

A Julia package for computing Krylov complexity and Krylov entropy of various quantum systems.

Given a configuration, the package generates the Hamiltonian $H$ and an initial operator $O$ of the system. Then, it computes the Lanczos $b$-coefficients of the Liouvillian superoperator $\mathcal{L} = [H, \cdot]$, from which it derives the Krylov complexity $C_K(t)$ and Krylov entropy $S_K(t)$ [[Parker et al., 2019]](https://arxiv.org/abs/1812.02666).

Currently, the following systems are supported: Ising, XXZ, XYZ, cSYK, GOE and GUE matrices.

The algorithm uses a constraint eigenspace representation to remove degenerate eigenvalues of the Liouvillian operator accounting for the projection of the initial operator (implemented based on [[Eckseler et al., 2025]](https://arxiv.org/abs/2505.02670)).

## Quick start

```julia
using KrylovComplexity

H, O = build_system(Dict("model" => "ising", "N" => 6,
                          "hx" => 1.05, "hz" => 0.5,
                          "precision_build" => 1024))

b, Kdim, reortho_count = lanczos(H, O; precision=:double, reortho=:pro)

t, K, S = krylov_complexity_entropy(b; time_max=1000.0)
```

Or drive everything from a YAML config:

```julia
result = simulate("examples/ising.yaml")
```

## Supported models

| Key | Model |
|-----|-------|
| `ising` | Transverse-field Ising chain |
| `xyz` | XYZ spin chain |
| `xxz` | XXZ chain with U(1) + ℤ₂ symmetry reduction |
| `goe` | GOE random matrix |
| `gue` | GUE random matrix |
| `csyk` | Complex SYK (all-to-all random quartic fermions) |

See the [README](https://github.com/shanoether/KrylovComplexity.jl) for full model equations,
parameter tables, and precision/reorthogonalization options.

## API

```@contents
Pages = ["api.md"]
Depth = 2
```