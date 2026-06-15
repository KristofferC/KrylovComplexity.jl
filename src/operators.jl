# --------
# Const
# Keys to build the pipeline
const _GLOBAL_KEYS = Set{String}([
    "model", "N", "precision_build", "precision", "reortho",
    "use_eigenbasis", "remove_degeneracies", "time_max",
    "dt_factor", "kc_out_resolution", "output_dir", "save_b", "do_plot",
    "save",
    "eps_reortho_factor", "eps_factor", "tol_factor", "max_iter"
])

#---------
# helpers

"Return the 2x2 Pauli-X matrix as a sparse `Complex{T}` matrix."
pauli_x(::Type{T}) where T<:AbstractFloat = sparse(Complex{T}[0 1;1 0])
"Return the 2x2 Pauli-Y matrix as a sparse `Complex{T}` matrix."
pauli_y(::Type{T}) where T<:AbstractFloat = sparse(Complex{T}[0 -1im;1im 0])
"Return the 2x2 Pauli-Z matrix as a sparse `Complex{T}` matrix."
pauli_z(::Type{T}) where T<:AbstractFloat = sparse(Complex{T}[1 0; 0 -1])
"Return the 2x2 identity matrix as a sparse `Complex{T}` matrix."
id2(::Type{T})     where T<:AbstractFloat = sparse(Complex{T}[1 0; 0 1])

"Tensor product of a list of matrices: kron(ops[1], ops[2], ...)."
kron_list(ops) = reduce(kron, ops)

@inline frobenius_inner(A, B) = real(dot(A, B)) / size(A, 1)
@inline frobenius_norm(A)     = sqrt(real(dot(A, A)) / size(A, 1))

# function _rescale_H(H::AbstractMatrix)
#     lambda_max = maximum(abs, eigvals(Hermitian(Matrix(H))))
#     if lambda_max < 1e-12
#         error("Max lambda of Hamiltonian is zero - exiting")
#     end
#     return H / lambda_max
# end

"""
    _rescale_O(O) -> O / ||O||_F

Normalise operator O so that its Frobenius norm equals 1.
"""
function _rescale_O(O::AbstractMatrix)
    return O / frobenius_norm(Matrix(O))
end

"""
    build_site_operator(op, site, N) -> SparseMatrix

Embed a single-site operator `op` at position `site` in an N-site chain
by tensoring with identity matrices on all other sites.
"""
function build_site_operator(op::AbstractMatrix, site::Int, N::Int)
    T = real(eltype(op))
    ops = [i == site ? op : id2(T) for i in 1:N]
    return kron_list(ops)
end

_build_rng(seed::Integer) = MersenneTwister(seed)
_build_rng(::Nothing)     = Random.default_rng()

# ----------
# Ising

const _ISING_DEFAULTS = (
    J           = 1.0,
    hx          = 1.0,
    hz          = 0.0,
    bc          = "pbc",
    oper        = "Sz",
    oper_site   = 1
)

"""
    ising_hamiltonian(N; J, hx, hz, bc, T) -> SparseMatrix

Build the transverse-field Ising Hamiltonian on N sites:
  H = J sum_{<i,j>} sigma^z_i sigma^z_j  +  hx sum_i sigma^x_i  +  hz sum_i sigma^z_i
Boundary conditions: `:pbc` (periodic) or `:obc` (open).
"""
function ising_hamiltonian(N::Int; J::Real=1.0, hx::Real=1.0, hz::Real=0.0, bc::Symbol=:pbc, T::Type{<:AbstractFloat}=Float64)
    sz = [build_site_operator(pauli_z(T), i, N) for i in 1:N]
    sx = [build_site_operator(pauli_x(T), i, N) for i in 1:N]

    H = spzeros(Complex{T}, 2^N, 2^N)

    if bc==:obc
        for i in 1:N-1
            H +=  J*sz[i]*sz[i+1]
        end
    elseif bc==:pbc
        for i in 1:N
            j = (i % N) + 1
            H += J*sz[i]*sz[j]
        end
    end

    for i in 1:N
        H += hx*sx[i] + hz*sz[i]
    end
    @info "Ising H built" N=N dim=size(H,1) nnz=nnz(H) J=J hx=hx hz=hz bc=bc T=T
    return H
end


"""
    ising_operator(N; op_name, site, T) -> SparseMatrix

Single-site Pauli operator for the Ising chain.
`op_name` is one of `"Sx"`, `"Sy"`, `"Sz"` (case-insensitive).
"""
function ising_operator(N::Int; op_name::AbstractString="sz", site::Int=1, T::Type{<:AbstractFloat}=Float64)
    key = lowercase(replace(op_name, "_" => "")) # S_z -> sz
    if key == "sx"
        op = pauli_x(T)
    elseif key == "sy"
        op = pauli_y(T)
    elseif key == "sz"
        op = pauli_z(T)
    else
        error("Invalid operator symbol. Initial operator must be a pauli matrix, choose between sx, sy, sz")
    end
    O = build_site_operator(op, site, N)
    @info "Ising operator built" N=N oper=op_name oper_site=site T=T
    return O
end


# ------
# XYZ
const _XYZ_DEFAULTS = (
    Jx          = 1.0,
    Jy          = 1.0,
    Jz          = 1.0,
    hz          = 0.0,
    c           = 0.0,
    bc          = "pbc",
    oper        = "Sz",
    oper_site   = 1
)

"""
    xyz_hamiltonian(N; Jx, Jy, Jz, hz, c, bc, T) -> SparseMatrix

Build the anisotropic XYZ Heisenberg Hamiltonian:
  H = Jx sum sigma^x_i sigma^x_{i+1}
    + Jy sum sigma^y_i sigma^y_{i+1}
    + Jz sum sigma^z_i sigma^z_{i+1}
    + hz sum sigma^z_i
    + c sigma^z_{N/2}
The single-site field `c` at the middle site breaks integrability when nonzero.
Boundary conditions: `:pbc` or `:obc`.
"""
function xyz_hamiltonian(N::Int; Jx::Real=1.0, Jy::Real=1.0, Jz::Real=1.0, hz::Real=0.0, c::Real=0.0, bc::Symbol=:pbc, T::Type{<:AbstractFloat}=Float64)
    sx = [build_site_operator(pauli_x(T), i, N) for i in 1:N]
    sy = [build_site_operator(pauli_y(T), i, N) for i in 1:N]
    sz = [build_site_operator(pauli_z(T), i, N) for i in 1:N]

    H = spzeros(Complex{T}, 2^N, 2^N)

    bonds = bc==:obc ? N-1 : N
    for i in 1:bonds
        j = (i % N) + 1
        H += Jx*sx[i]*sx[j] + Jy*sy[i]*sy[j] + Jz*sz[i]*sz[j]
    end

    for i in 1:N
        H +=hz*sz[i]
    end
    H += c *sz[div(N,2)]
    @info "XYZ H built" N=N dim=size(H,1) nnz=nnz(H) Jx=Jx Jy=Jy Jz=Jz hz=hz c=c
    return H
end


"""
    xyz_operator(N; op_name, site, T) -> SparseMatrix

Single-site Pauli operator for the XYZ chain (delegates to `ising_operator`).
"""
function xyz_operator(N::Int; op_name::AbstractString="sz", site::Int=1, T::Type{<:AbstractFloat}=Float64)
    return ising_operator(N; op_name=op_name, site=site, T=T)
end

# ------
# GOE & GUE
const _RANDOM_DEFAULTS = (
    seed        = nothing,
    sparsity    = 1.0,
)

"""
    GOE_matrix(N, rng; sparsity, T) -> SparseMatrix

Draw a random real symmetric matrix from the Gaussian Orthogonal Ensemble.
  A_{ij} ~ N(0, 1/2),  A_{ii} ~ N(0, 1),  eigenvalues in [-2, 2] (Wigner semicircle).
"""
function GOE_matrix(N::Int, rng::AbstractRNG; sparsity=1.0, T::Type{<:AbstractFloat}=Float64)
    A = sprandn(rng, T,2^N,2^N, sparsity)
    return (A+A')/sqrt(2*sparsity*2^N) #rescaling to ahve the same order of magnitude between te eigenvalues for different system size    return (A+A')/sqrt(2*sparsity*D) #rescaling to have the same order of magnitude between te eigenvalues for different system size
end

"""
    GUE_matrix(N, rng; sparsity, T) -> SparseMatrix

Draw a random complex Hermitian matrix from the Gaussian Unitary Ensemble.
  Re/Im(A_{ij}) ~ N(0, 1/2),  A_{ii} ~ N(0, 2),  eigenvalues in [-2, 2] (Wigner semicircle).
"""
function GUE_matrix(N::Int, rng::AbstractRNG; sparsity=1.0, T::Type{<:AbstractFloat}=Float64)
    A = sprandn(rng, T,2^N,2^N, sparsity) + 1im*sprandn(rng, T,2^N,2^N, sparsity)
    return (A+A')/sqrt(2*sparsity*2^N) #rescaling to ahve the same order of magnitude between te eigenvalues for different system size
end

# ------
# XXZ with symmetries

"""
    m is the number of spin up
    Returns the indieces of base states with M up spin that corresponds with M 1
    Shift of one because indexing in julia starts with one
"""
## Helper
function _block_magnetization(N::Int, m::Int)
    D = 2^N
    base = [s+1 for s in 0:D-1 if count_ones(s) == m]
    @info("m=$m: magnetization sector: $D => $(length(base))")
    return base #length is binomial(N,M)
end

function _reverse_bits(x::Int, N::Int)
    y = 0
    for _ in 1:N
        y = (y << 1) | (x & 1)     # grab LSB of x, append to y
        x >>= 1                    # discard that bit from x
    end
    return y
end

function _parity_basis(N::Int, sector_indices::Vector{Int}, T::Type{<:AbstractFloat})
    D = 2^N
    visited = falses(D)
    idx = Dict(val => i for (i, val) in enumerate(sector_indices))

    # Triplets for sparse construction: (row, col, value)
    # P is for Positive and N for negative
    rP = Int[]; cP = Int[]; vP = Complex{T}[]   # Vplus
    rM = Int[]; cM = Int[]; vM = Complex{T}[]   # Vminus

    colP = 0
    colM = 0
    a = one(T) / sqrt(T(2))

    for s in sector_indices
        visited[s] && continue
        r = _reverse_bits(s-1, N)+1 #because of 0 indexing try reverse_bits(7-1,4)
        !haskey(idx, r) && continue

        if r == s # reverse_bits(s, N) == s
            # Palindrome: |s> is already even parity
            visited[s] = true
            colP += 1 # move to a new column
            push!(rP, idx[s]); push!(cP, colP); push!(vP, Complex{T}(1, 0)) # basically add a column of 0 except in i where there is a one, because |s> is an eigenvalue of parity +1
        else
            # Pair: build symmetric (+) and antisymmetric (-)
            visited[s] = true
            visited[r] = true

            # add new column of eigenvector of parity +1 = (|s> + |r>)/sqrt(2)
            colP += 1
            push!(rP, idx[s]); push!(cP, colP); push!(vP, Complex{T}(a, 0))
            push!(rP, idx[r]); push!(cP, colP); push!(vP, Complex{T}(a, 0))

            # add new column of eigenvector of parity -1 = (|s> - |r>)/sqrt(2)
            colM += 1
            push!(rM, idx[s]); push!(cM, colM); push!(vM, Complex{T}(a, 0))
            push!(rM, idx[r]); push!(cM, colM); push!(vM, Complex{T}(-a, 0))
        end
    end

    Vplus  = sparse(rP, cP, vP, length(sector_indices), colP)
    Vminus = sparse(rM, cM, vM, length(sector_indices), colM)
    return Vplus, Vminus
end

function _project_to_sector(A, N::Int, M::Int, parity::Int, T::Type{<:AbstractFloat})
    I = _block_magnetization(N,M)
    A_m = A[I,I]
    parity == 0 && return A_m
    Vplus, Vminus = _parity_basis(N,I,T)
    parity == 1 && return Vplus' * A_m * Vplus
    parity == -1 && return Vminus' * A_m * Vminus
    error("parity must be 0, 1, or -1, got $parity")
end

## Const
const _XXZ_DEFAULTS = (
    J               = 1.0,
    Jzz             = 1.0,
    d               = 1.0,  # -d in the Jzz(sigma_i^z sigma_i+1^z -d)
    c               = 0.0,   # H += c * sz[div(N,2)+1]
    magnetization   = -1,   # -1 → use N÷2 at build time, cannot be negative as it is the number of |up>
    parity          = 1,
    oper_site       = -1,   # -1 → use N÷2 at build time, cannot be negative
    bc              = "obc"
)

## Hamiltonian

"""
    xxz_hamiltonian(N; J, Jzz, d, c, magnetization, parity, bc, T) -> SparseMatrix

Build the XXZ Hamiltonian projected to the (magnetization, parity) symmetry sector:
  H = -(J/2) sum_{<i,j>} (sigma^x_i sigma^x_j + sigma^y_i sigma^y_j
                          + Jzz (sigma^z_i sigma^z_j - d))
    + c sigma^z_{(N+1)/2}
Symmetries removed:
  - U(1) spin rotation: total-Sz sector with `magnetization` spin-up sites
  - Z2 spatial reflection (parity): `parity = 1` (symmetric) or `-1` (antisymmetric)
    Set `parity = 0` to skip parity projection.
Default boundary conditions: `:obc`.
"""
function xxz_hamiltonian(N::Int; J::Real=1.0, Jzz::AbstractFloat=1.0, d::AbstractFloat=1.0, c::AbstractFloat=0.0, magnetization::Int=-1, parity::Int=1, bc::Symbol=:obc, T::Type{<:AbstractFloat}=Float64)
    M = magnetization == -1 ? div(N,2) : magnetization
    D = 2^N
    sx = [build_site_operator(pauli_x(T), i, N) for i in 1:N]
    sy = [build_site_operator(pauli_y(T), i, N) for i in 1:N]
    sz = [build_site_operator(pauli_z(T), i, N) for i in 1:N]
    Id = spdiagm(ones(Complex{T}, D))
    H = spzeros(Complex{T}, D, D)
    bonds = bc == :obc ? (1:N-1) : (1:N)
    for i in bonds
        # j = bc == :obc ? i+1 : (i % N) + 1
        j = (i % N) + 1 # never close if obc
        H += -J/2*(sx[i]*sx[j] + sy[i]*sy[j] + Jzz*(sz[i]*sz[j] - d*Id))
    end
    H += c * sz[div(N+1,2)]
    @debug "XXZ H before projection" N=N M=M D=size(H,1)
    H_proj = _project_to_sector(H, N, M, parity, T)
    @info "XXZ H block built" N=N M=M parity=parity dim=size(H_proj,1) nnz=nnz(H_proj)
    return H_proj # ./M to match with  http://arxiv.org/abs/2207.07701
end

"""
    xxz_operator(N; oper_site, magnetization, parity, T) -> SparseMatrix

Build the XXZ initial operator projected to the (magnetization, parity) sector.
For parity != 0: O = sigma^z_{oper_site} + sigma^z_{N+1-oper_site} (symmetrised).
For parity == 0: O = sigma^z_{oper_site}.
The result is made traceless and projected to the symmetry sector.
"""
function xxz_operator(N::Int; oper_site::Int=-1, magnetization::Int=-1, parity::Int=1, T::Type{<:AbstractFloat}=Float64)
    oper_site = oper_site == -1 ? div(N,2) : oper_site
    M = magnetization == -1 ? div(N,2) : magnetization
    sz = [build_site_operator(pauli_z(T), i, N) for i in 1:N]
    O = parity == 0 ? sz[oper_site] : sz[oper_site] + sz[N+1-oper_site]
    O_proj = _project_to_sector(O, N, M, parity, T)

    O_proj -= (tr(O_proj) / size(O_proj,1)) * I    # Make it traceless like in http://arxiv.org/abs/2207.07701

    @info "XXZ operator built" N=N M=M parity=parity dim=size(O_proj,1)
    return O_proj
end

# ------
# cSYK

## Helper

_csyk_bit(x,m)          = (x >> (m-1)) & 1
_csyk_flip(x,m)         = x ⊻ (1 << (m-1))
_csyk_parity_left(x,m)  = count_ones(x & ((1 << (m-1)) - 1)) % 2 # careful left side in binary representation of the integer is on the right!


function _csyk_annihilate(x,m)
    _csyk_bit(x,m) == 0 && return nothing, 0 #no particule at m, nothing to annihilate
    return _csyk_flip(x,m), (-1)^_csyk_parity_left(x,m) 
end

function _csyk_create(x,m)
    _csyk_bit(x,m) == 1 && return nothing, 0 #already a particule at m, nothing to create
    return _csyk_flip(x,m), (-1)^_csyk_parity_left(x,m)  
end

function _csyk_basis(N,Q)
    if Q == 0 # convetion: full basis
        basis_state = collect(0:2^N-1)      # Q=0: full basis
    else
        basis_state = [i for i in 0:2^N-1 if count_ones(i) == Q] # Q>0: charge sector
    end
    @debug "Created basis state for cSYK with N=$N and Q=$Q: len=$(length(basis_state))"
    return basis_state
end

## const
const _CSYK_DEFAULTS = (
    J       = 1.0,
    Q       = -1,       # dynmaic build to N/4
    seed    = nothing,
    site1   = -1,  #dynamic build to N-1
    site2   = -1   #dynamic build to N
)

## building
"""
    csyk_hamiltonian(N; J, Q, seed, T) -> SparseMatrix

Build the complex SYK Hamiltonian in the Q-particle charge sector:
  H = sum_{i<j,k<l} J_{ijkl} c^+_i c^+_j c_k c_l  +  h.c.
where J_{ijkl} are complex Gaussian with variance J^2 * 3! / N^3.
`Q = N/2` by default.  Fix `seed` for reproducibility.
"""
function csyk_hamiltonian(N; J::Real=1.0, Q::Int=-1, seed=nothing, T::Type=Float64)
    # q=4 by default
    rng = _build_rng(seed) 
    Q = Q==-1 ? div(N,2) : Q
    state_basis = _csyk_basis(N,Q)
    index_state = Dict(b=>i for (i, b) in enumerate(state_basis))

    D = length(state_basis)
    H = spzeros(Complex{T}, D, D)

    pairs = [(i,j) for i in 1:N for j in i+1:N]

    sigma2_J = T(factorial(3)) * T(J)^2 / T(N)^3
    # J_ijkl = 
    # Original coupling
    # J_ijkl = Dict(((i,j),(k,l)) => sqrt(sigma2_J) * _to_T(randn(rng), T) * exp(Complex{T}(0, _to_T(2*pi*rand(rng, Uniform(0,1)), T))) for (i,j) in pairs for (k,l) in pairs)
    J_ijkl = Dict(((i,j),(k,l)) => sqrt(sigma2_J) * T(randn(rng)) * exp(Complex{T}(0, T(2π) * T(rand(rng)))) for (i,j) in pairs for (k,l) in pairs) # if (i,j) != (k,l))

    for (i,j) in pairs # creation
        for (k,l) in pairs # annihilition$
            #(i,j) == (k,l) && continue # to test
            for (col, x) in enumerate(state_basis)
                # apply the different annihiliation and creation operator form right to left
                x1, s1 = _csyk_annihilate(x,l) # c_l |x⟩
                isnothing(x1) && continue # cannot annihilate if no particle: this term is 0
                x2, s2 = _csyk_annihilate(x1,k) # c_k c_l |x⟩
                isnothing(x2) && continue
                x3, s3 = _csyk_create(x2,j) # c†_j c_k c_l |x⟩
                isnothing(x3) && continue
                x4, s4 = _csyk_create(x3,i) # c†_i c†_j c_k c_l |x⟩
                isnothing(x4) && continue
                row = get(index_state, x4, nothing)
                isnothing(row) && continue
                H[row, col] += J_ijkl[(i,j),(k,l)] * s1*s2*s3*s4
            end
        end
    end
    H += H'
    @info "cSYK H block built" N=N Q=Q J=J dim=size(H,1) nnz=nnz(H)
    return H
end

"""
    csyk_operator(N; Q, site1, site2, T) -> SparseMatrix

Build the cSYK initial operator in the Q-particle sector:
  O = c^+_{site1} c_{site2}  +  c_{site1} c^+_{site2}
Defaults: site1 = N-1, site2 = N.
"""
function csyk_operator(N::Int; Q::Int=-1, site1::Int=-1, site2::Int=-1, T::Type=Float64)
    Q = Q==-1 ? div(N,2) : Q
    site1 = site1==-1 ? N-1 : site1
    site2 = site2==-1 ? N : site2
    state_basis = _csyk_basis(N,Q)
    index_state = Dict(b=>i for (i, b) in enumerate(state_basis))

    D = length(state_basis)
    O = spzeros(Complex{T}, D, D)

    for (col, x) in enumerate(state_basis)
        # c^dagger L-1 c_L 
        x1, s1 = _csyk_annihilate(x,site2)
        if !isnothing(x1) 
            x2, s2 = _csyk_create(x1,site1)
            if !isnothing(x2)
                row = index_state[x2]
                O[row, col] += s1*s2
            end
        end

        # c_L-1 c^\dagger_L
        y1, t1 = _csyk_annihilate(x,site1)
        if !isnothing(y1) 
            y2, t2 = _csyk_create(y1,site2)
            if !isnothing(y2)
                row = index_state[y2]
                O[row, col] += t1*t2
            end
        end
    end
    
    @info "cSYK operator built" N=N Q=Q dim=size(O,1)
    return O
end

# ------
# XYZ


# ------
# Dispatch

"""
    _precision_type(p) -> (Type, Int)

Map a precision specifier to a Julia type and bit count.
Accepted: `:f64`/`"f64"`, `:double`/`"double"`, `:bigfloat`/`"bigfloat"`, or a positive integer.
"""
function _precision_type(p)
    (p == "f64"         || p == :f64)           && return (Float64, 53)
    (p == "double"      || p == :double)        && return (Double64, 106)
    (p == "bigfloat"    || p == :bigfloat)      && return (BigFloat, 256)
    if p isa Integer
        p > 0 || error("precision_build must be a positive integer, got $p")
        return (BigFloat, Int(p))
    end
    return (Float64, 53) #Float64 has 53 bits of precision
end

"""
    _resolve_params(defaults, cfg) -> NamedTuple

Merge model-specific config keys from `cfg` into `defaults`.
Raises an error if `cfg` contains any key not in `defaults` and not in `_GLOBAL_KEYS`.
"""
function _resolve_params(defaults::NamedTuple, cfg::Dict)
    model_keys = Set(string.(keys(defaults))) 
    # Check the keys are allowed
    for k in keys(cfg)
        k in model_keys || k in _GLOBAL_KEYS ||
            error("Unknow config key: '$k'")
    end
    # overwrite default keys with user settings
    return NamedTuple{keys(defaults)}(
        get(cfg, string(k), defaults[k]) for k in keys(defaults)
    )
end


"""
    build_system(cfg) -> (H, O)

Build the Hamiltonian H and initial operator O from a configuration dict.
Required keys: `"model"` and `"N"`.  Optional key `"precision_build"` sets
the floating-point type used for construction (default `:f64`).

The returned O is normalised so that its Frobenius norm equals 1.
See the model-specific `*_hamiltonian` / `*_operator` functions for
model-specific parameters.
"""
function build_system(cfg::Dict)
    haskey(cfg, "model") || error("Config must specify a 'model' key")
    haskey(cfg, "N")     || error("Config must specify a 'N' key (system size)")

    model = lowercase(cfg["model"])
    N = cfg["N"]

    T, bits = _precision_type(get(cfg, "precision_build", "f64"))
    
    _build() = begin
        if model == "ising"
            p = _resolve_params(_ISING_DEFAULTS, cfg)
            H = ising_hamiltonian(N; J=p.J, hx=p.hx, hz=p.hz, bc=Symbol(p.bc), T=T)
            O = ising_operator(N; op_name=p.oper, site=p.oper_site, T=T)
        elseif model == "xyz"
            p = _resolve_params(_XYZ_DEFAULTS, cfg)
            H = xyz_hamiltonian(N; Jx=p.Jx, Jy=p.Jy, Jz=p.Jz, hz=p.hz, c=p.c, bc=Symbol(p.bc), T=T)
            O = xyz_operator(N; op_name=p.oper, site=p.oper_site, T=T)
        elseif model == "goe"
            p = _resolve_params(_RANDOM_DEFAULTS, cfg)
            rng = _build_rng(p.seed)
            H = GOE_matrix(N, rng; sparsity=p.sparsity, T=T)
            O = GOE_matrix(N, rng; sparsity=p.sparsity, T=T)
        elseif model == "gue"
            p = _resolve_params(_RANDOM_DEFAULTS, cfg)
            rng = _build_rng(p.seed)
            H = GUE_matrix(N, rng; sparsity=p.sparsity, T=T)
            O = GUE_matrix(N, rng; sparsity=p.sparsity, T=T)
        elseif model == "xxz"
            p = _resolve_params(_XXZ_DEFAULTS, cfg)
            H = xxz_hamiltonian(N; J=p.J, Jzz=p.Jzz, d=p.d, c=p.c, magnetization=p.magnetization, parity=p.parity, bc=Symbol(p.bc), T=T)
            O = xxz_operator(N; oper_site=p.oper_site, magnetization=p.magnetization, parity=p.parity, T=T)
        elseif model == "csyk"
            p = _resolve_params(_CSYK_DEFAULTS, cfg)
            Q = p.Q == -1 ? div(N,2) : p.Q
            H = csyk_hamiltonian(N; J=p.J, Q=Q, seed=p.seed, T=T)
            O = csyk_operator(N; Q=Q, site1=p.site1, site2=p.site2, T=T)
        else
            error("Unknow model $model")
        end
        @info "System built" model=model N=N H_dim=size(H,1) H_nnz=nnz(H) O_dim=size(O,1) O_nnz=nnz(O) build_precision=bits
        return H, _rescale_O(O)
    end

    return T === BigFloat ? setprecision(_build, bits) : _build()
end