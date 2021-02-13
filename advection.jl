using DiscontinuousGherkins
using DiscontinuousGherkins: jacobi_gauss_lobatto, vandermonde_1d, Dmatrix1D

NODETOL = 1e-10

N = 8  # Order of polymomials used for approximation
K = 10 # Number of elements

domain = (0, 2)
grid = DiscontinuousGalerkin1DGrid(domain, K)

Np = N+1
Nfp = 1
Nfaces = 2

r = jacobi_gauss_lobatto(0, 0, N)

V = vandermonde_1d(N, r)
V⁻¹ = inv(V)

Dr = Dmatrix1D(N, r, V)

Emat = zeros(Np, Nfaces * Nfp)
Emat[1,  1] = 1
Emat[Np, 2] = 1
lift = V * (V' * Emat)

# build coordinates of all the nodes
VX = grid.node_coordinates
EToV = grid.element_to_node_connectivity
va = EToV[:, 1]'
vb = EToV[:, 2]'
x = ones(N+1, 1) * VX[Int.(va)] .+ 0.5 .* (r .+ 1) * (VX[Int.(vb)] - VX[Int.(va)])

# Compute the metric elements for the local mappings of the 1D elements.
xr = Dr * x
J = xr
rx = 1 ./ J

# Compute masks for edge nodes
Fmask = BitArray(undef, size(x)...)
Fmask .= 0
Fmask[1, 1] = 1
Fmask[end, end] = 1
Fx = x[Fmask]

# % Build surface normals and inverse metric at surface
nx = zeros(Nfp * Nfaces, K)
nx[1, :] .= -1
nx[2, :] .= +1

Fscale = 1 ./ J[Fmask]

## Build global connectivity arrays for 1D grid based on standard EToV input array from grid generator

Nfaces = 2

# Find number of elements and vertices
TotalFaces = Nfaces * K
Nv = K+1

# List of local face to local vertex connections
vn = Int[1, 2]

# Build global face to node sparse array
SpFToV = zeros(TotalFaces, Nv)
sk = 1;
for k in 1:K, face in 1:Nfaces
    SpFToV[sk, EToV[k, vn[face]] |> Int] = 1
end

# Build global face to global face sparse array
SpFToF = SpFToV * SpFToV' - I(TotalFaces)

% Find complete face to face connections
[faces1, faces2] = find(SpFToF==1);

% Convert face global number to element and face numbers
element1 = floor( (faces1-1)/Nfaces ) + 1;
face1 = mod( (faces1-1), Nfaces ) + 1;
element2 = floor( (faces2-1)/Nfaces ) + 1;
face2 = mod( (faces2-1), Nfaces ) + 1;

% Rearrange into Nelements x Nfaces sized arrays
ind = sub2ind([K, Nfaces], element1, face1);
EToE = (1:K)’*ones(1,Nfaces);
EToF = ones(K,1)*(1:Nfaces);
EToE(ind) = element2; EToF(ind) = face2;
return
