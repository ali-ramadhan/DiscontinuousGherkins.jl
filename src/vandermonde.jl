function vandermonde_1d(N, r)
    V = zeros(length(r), N+1)
    for j in 1:N+1
        @. V[:, j] = jacobi(r, 0, 0, j-1)
    end
    return V
end

" Initialize the gradient of the modal basis (i) at (r) at order N. "
function ∂vandermonde_1d(N, r)
    DVr = zeros(length(r), N+1)
    for i in 0:N
        @. DVr[:, i+1] = ∂jacobi(r, 0, 0, i)
    end
    return DVr
end

" Initialize the (r) differentiation matrices on the interval, evaluated at (r) at order N. "
function Dmatrix1D(N, r, V)
    Vr = ∂vandermonde_1d(N, r)
    Dr = Vr / V
    return Dr
end
