struct DiscontinuousGalerkin1DGrid{N, C}
                node_coordinates :: N
    element_to_node_connectivity :: C
end

"""
Generate simple equidistant grid with N elements.
"""
function DiscontinuousGalerkin1DGrid(domain, N)
    xₗ, xᵣ = domain

    node_coordinates = [xₗ + (xᵣ - xₗ) * (i - 1) / N for i in 1:N+1]

    # read element to node connectivity
    element_to_node_connectivity = zeros(N, 2)
    for n in 1:N
        element_to_node_connectivity[n, 1] = n
        element_to_node_connectivity[n, 2] = n+1
    end

    return DiscontinuousGalerkin1DGrid{typeof(node_coordinates), typeof(element_to_node_connectivity)}(
        node_coordinates, element_to_node_connectivity)
end
