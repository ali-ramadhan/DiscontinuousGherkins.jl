include("jacobi.jl")

function vandermonde_1d(N, r)
    V = zeros(length(r), N+1)
    for j in 1:N+1
        @. V[:,j] .= jacobi.(r, 0, 0, j-1)
    end
    return V
end
