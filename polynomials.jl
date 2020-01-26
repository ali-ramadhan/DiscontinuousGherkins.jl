using SpecialFunctions
using OffsetArrays

const Γ = gamma

# Coefficients in the Jacobi polynomial recurrence relations.
a(α, β, n) = 2/(2n+α+β) * √(n * (n+α+β) * (n+α) * (n+β) / (2n+α+β-1) / (2n+α+β+1))
b(α, β, n) = -(α^2 - β^2) / (2n+α+β) / (2n+α+β+2)

" Evaluates the Jacobi polynomial P_n^(α,β) at the point x. "
function jacobi(x, α, β, n::Int)
    P₀ᵅᵝ = √(2.0^-(α+β+1) * Γ(α+β+2) / Γ(α+1) / Γ(β+1))
    n == 0 && return P₀ᵅᵝ

    P₁ᵅᵝ = P₀ᵅᵝ/2 * √((α+β+3) / (α+1) / (β+1)) * ((α+β+2)*x + α - β)
    n == 1 && return P₁ᵅᵝ

    Pᵅᵝ = OffsetArray(zeros(n+1), 0:n)
    Pᵅᵝ[0] = P₀ᵅᵝ
    Pᵅᵝ[1] = P₁ᵅᵝ
    for n′ in 1:n-1
        Pᵅᵝ[n′+1] = ((x - b(α,β,n′)) * Pᵅᵝ[n′] - a(α,β,n′) * Pᵅᵝ[n′-1]) / a(α, β, n′+1)
    end
    
    return Pᵅᵝ[n]
end


