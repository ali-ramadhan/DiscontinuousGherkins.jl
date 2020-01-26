using LinearAlgebra
using SpecialFunctions
using OffsetArrays

const Γ = gamma

# Coefficients in the Jacobi polynomial recurrence relations.
a(α, β, n) = 2/(2n+α+β) * √(n * (n+α+β) * (n+α) * (n+β) / (2n+α+β-1) / (2n+α+β+1))
b(α, β, n) = -(α^2 - β^2) / (2n+α+β) / (2n+α+β+2)

" Evaluates the Jacobi polynomial Pₙᵅᵝ(x). "
function jacobi(x, α, β, n::Int)
    Pᵅᵝ = OffsetArray(zeros(max(2, n+1)), 0:n)
    Pᵅᵝ[0] = √(2.0^-(α+β+1) * Γ(α+β+2) / Γ(α+1) / Γ(β+1))
    Pᵅᵝ[1] = Pᵅᵝ[0]/2 * √((α+β+3) / (α+1) / (β+1)) * ((α+β+2)*x + α - β)
    for n′ in 1:n-1
        Pᵅᵝ[n′+1] = ((x - b(α,β,n′)) * Pᵅᵝ[n′] - a(α,β,n′) * Pᵅᵝ[n′-1]) / a(α, β, n′+1)
    end
    return Pᵅᵝ[n]
end

" Evaluates the derivative of the Jacobi polynomial Pₙᵅᵝ(x). "
∂jacobi(x, α, β, n::Int) = √(n * (n+α+β+1)) * jacobi(x, α+1, β+1, n-1)

" Guassian quadrature points and weights for the Jacobi polynomial Pₙᵅᵝ. "
function jacobi_gauss_quadrature(α, β, N)
    N == 0 && return [(α-β) / (α+β+2)], [2]

    # Form symmetric matrix from recurrence.
    dv = OffsetArray(zeros(N+1), 0:N)  # diagonal vector
    ev = OffsetArray(zeros(N+1), 0:N)  # sub/super-diagonal vector

    for n in 0:N
        dv[n] = b(α, β, n)
        ev[n] = a(α, β, n)
    end

    # Create full matrix combining the two.
    # Need to pass arrays that are not offset.
    J = SymTridiagonal(dv[0:N], ev[1:N])
    (α + β) ≈ 0 && (J[1, 1] = 0)

    # Compute quadrature points and weights by eigenvalue solve.
    x, V = eigen(J)
    w = @. V[1, :]^2 * 2^(α+β+1) / (α+β+1)
    @. w *= factorial(α) * factorial(β) / factorial(α+β)

    return x, w
end

" Guass-Labatto quadrature points for the Jacobi Polynomial Pₙᵅᵝ. "
function jacobi_gauss_lobatto(α, β, N)
    N == 0 && error("What are you doing? Gauss-Lobatto points only make sense if N >= 1.")
    N == 1 && return [-1, 1]
    
    x = zeros(N+1)
    x[1], x[N+1] = -1, 1

    x_GQ, w = jacobi_gauss_quadrature(α+1, β+1, N-2)
    x[2:N] .= x_GQ
    
    return x
end

