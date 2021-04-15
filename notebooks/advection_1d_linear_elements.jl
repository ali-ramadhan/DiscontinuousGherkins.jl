### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 29443685-beb0-4102-a46d-63c48b008742
begin
	using LinearAlgebra
	using Plots
	plotly()
end

# ╔═╡ bdeabaa3-8a91-4f9d-8cb7-4527a6385438
md"""
This notebook is a Julia translation of Josh Bevan's [intro to DG YouTube video](https://www.youtube.com/watch?v=sMfPHJUKfaI).
"""

# ╔═╡ dbbb3fea-9dea-11eb-1f7b-afb4a0c5114e
md"""
Here we attempt to solve the 1D scalar conservation eqn of the form:

$ \displaystyle \frac{\partial u}{\partial t} + \frac{\partial f(u)}{\partial x} = 0 $

where $f(u)$ is some flux function describing the "flow" of a conserved
quantity. In this simplified case $f(u) = u$ giving us a linear PDE
We will use periodic boundary conditions to examine numerical dissipation
effects
"""

# ╔═╡ 35708c02-2ef3-432f-963b-62b77f072b19
md"""
Let $u(x,t)$ be the exact solution on the domain $0 \leq x \leq 1$.

Let $u(x,0) = u_0(x) = \sin(2\pi x)$.

Let $V_h$ be the finite vector space of linear polynomials.

Let $u_h$ be the approximate numerical solution consisting of a linear 
combination of basis functions ($\psi_n$ for the nth basis) in $V_h$ with
scalar coefficients $a_n$.

Let $\phi_n$ be the nth test function in the same vector space as the basis
functions ($V_h$), in this simplified case N=2.
"""

# ╔═╡ 14c235cf-7b61-45b2-85a8-3d8af45b4edb
N = 2

# ╔═╡ 6be6e442-6983-4629-b734-e6487ac8001d
md"""
Discretize the domain into $K$ elements with $K+1$ nodes.
"""

# ╔═╡ ec7aeb56-230b-4d05-a561-632f71c09e02
struct GridElement{T}
	xl :: T  # Left endpoint
	xr :: T  # Right endpoint
end

# ╔═╡ 34a301b0-58b4-4617-ad47-bd7883b09d8b
struct Grid{T, E}
	K :: Int
	Δ :: T
	elements :: E
end

# ╔═╡ d7c33092-ca55-4e1f-92be-d62aef900393
begin
	K = 32
	Δx = 1/K
	elements = [GridElement(i*Δx, (i+1)*Δx) for i in 0:K-1]
	grid = Grid(K, Δx, elements)
end

# ╔═╡ 28c5d715-c14a-4573-81b5-1e835bc4926c
md"""
Compute and store the initial condition $u_0(x) = \sin(2\pi x)$.
"""

# ╔═╡ d75f11b0-e29b-41d6-87bb-108b0d3cb468
struct FieldElement{D}
	data :: D
end

# ╔═╡ 3f8100e7-dd79-44eb-864c-45ae08c6445a
begin	
	struct Field{G, E}
		grid :: G
		elements :: E
	end
	
	function Field(grid, f::Function)
		field_elements = []
		for grid_element in grid.elements
			field_data = [f(grid_element.xl), f(grid_element.xr)]
			field_element = FieldElement(field_data)
			push!(field_elements, field_element)
		end
		return Field(grid, field_elements)
	end
end

# ╔═╡ b52a8325-a0e0-4697-81b4-26abfbe70a98
begin
	u₀(x) = sin(2π*x)
	u = Field(grid, u₀)
end

# ╔═╡ 45016410-7e25-4f5b-82c5-aa665aaa3c7f
md"""
Let's plot ``u_0(x)`` now!
"""

# ╔═╡ 2f617585-62c1-4f6c-9088-aed874dc43e0
begin
	p = plot()
	for (ge, ue) in zip(grid.elements, u.elements)
		plot!([ge.xl, ge.xr], ue.data, linewidth=2, label="")
	end
	p
end

# ╔═╡ 4de78270-1f76-4f66-9271-34d0f148bc0c
md"""
According to Cockburn & Shu (2001, eq 2.2) let $u_h(x,0)$ be computed by

`` \displaystyle \int u_h(x,0) \phi(x) dx = \int u_0(x) \phi(x) dx ``

for each element (``x_{j-1/2} < x < x_{j+1/2}``).

We can explicitly define a formula for the value of the RHS 

`` \displaystyle \int u_0(x) \phi_N(x) dx ``
"""

# ╔═╡ 98e97ab0-6623-4a9e-a9b9-fb80ca182f53
md"""
For each element in the domain $a \le x \le b$, the cell coordinate is $0 \le \xi \le 1$ where

`` x = a + (b - a) \xi  = a + \Delta \xi ``

and

`` \displaystyle \xi = \frac{x - a}{\Delta} ``

so the basis ramp function can be written as

`` \displaystyle \phi_0(\xi) = 1 - \xi \implies \phi_0(x) = \frac{b - x}{\Delta} ``

`` \displaystyle \phi_1(\xi) = \xi \implies \phi_1(x) = \frac{x - a}{\Delta} ``
"""

# ╔═╡ c22c283e-615e-4964-9ff1-f9102f6862b2
md"""
For the left endpoint

``
\displaystyle
\int_a^b u_0(x) \phi_0(x) \; dx
= \int_a^b \sin(2\pi x) \frac{b-x}{\Delta} \; dx
= \frac{2\pi\Delta \cos(2\pi a) + \sin(2\pi a) - \sin(2\pi b)}{4\pi^2\Delta}
``
"""

# ╔═╡ 520fcb4f-bffc-4d53-97b1-703dfaf0b8ca
md"""
For the right endpoint

``
\displaystyle
\int_a^b u_0(x) \phi_1(x) \; dx
= \int_a^b \sin(2\pi x) \frac{x-a}{\Delta} \; dx
= \frac{-2\pi\Delta \cos(2\pi b) + \sin(2\pi b) - \sin(2\pi a)}{4\pi^2\Delta}
``
"""

# ╔═╡ 1096111c-9281-46bc-b5a4-4178a9855cad
begin
	∫u₀ϕ₀dx(a, b, Δ) = (2π*Δ*cos(2π*a) + sin(2π*a) - sin(2π*b)) / (4π^2*Δ)
	∫u₀ϕ₁dx(a, b, Δ) = (-2π*Δ*cos(2π*b) + sin(2π*b) - sin(2π*a)) / (4π^2*Δ)

	∫u₀ϕ_elements = []
	for grid_element in grid.elements
		xₗ = grid_element.xl
		xᵣ = grid_element.xr
		
		e0 = ∫u₀ϕ₀dx(xₗ, xᵣ, grid.Δ)
		e1 = ∫u₀ϕ₁dx(xₗ, xᵣ, grid.Δ)
		
		field_element = FieldElement([e0, e1])
		push!(∫u₀ϕ_elements, field_element)
	end
	
	∫u₀ϕ = Field(grid, ∫u₀ϕ_elements)
end

# ╔═╡ 16e1782b-3948-4d9b-8bfd-fb3134211977
md"""
To find the initial basis weights we need to solve for $a_i$ from

``
\displaystyle
\int_a^b u_0(x) \phi_j(x) \; dx
= \sum_{i=0}^1 a_i \int_a^b \phi_i(x) \phi_j(x) \; dx
``

or from the system of linear equations

``
\begin{pmatrix}
\int u_0 \phi_0
\int u_0 \phi_1
\end{pmatrix}
``
"""

# ╔═╡ 7bc02a02-947b-4526-8bed-49007f3fa8a4
mass_matrix = [1/3 1/6;
               1/6 1/3]

# ╔═╡ 92bdd228-617f-42da-b3d9-b9e9e18a7e4b
begin
	u_basis_elements = []
	for (k, grid_element) in enumerate(grid.elements)		
		e = mass_matrix \ ∫u₀ϕ.elements[k].data / grid.Δ
		u_basis_element = FieldElement(e)
		push!(u_basis_elements, u_basis_element)
	end
	u_basis = Field(grid, u_basis_elements)
end

# ╔═╡ 50142bc9-3dcd-4817-b9a6-3c4f37b292b1
begin
	p2 = plot()
	for (ge, ue) in zip(grid.elements, u_basis.elements)
		plot!([ge.xl, ge.xr], ue.data, linewidth=2, label="")
	end
	p2
end

# ╔═╡ 4a103139-e2f9-4182-89b0-9b0b3b4b2503
stiffness_matrix = [0 -1/2 -1/2;
                    0  1/2  1/2]

# ╔═╡ 5c7f8b83-1830-4466-82a2-dd2a6da4115b
upwind_flux = [1 0  0;
               0 0 -1]

# ╔═╡ 2d955948-bf0e-4f89-8bfa-2ffa6b9ac328
stencil = mass_matrix \ (stiffness_matrix + upwind_flux) 

# ╔═╡ aed204c6-d68d-449a-aee7-b5fcc537b81a
Δt = 1e-3

# ╔═╡ b7d972b4-82de-4c81-8a9a-80b98850763f
linear_operator = [0 1 0; 0 0 1] + Δt * stencil / grid.Δ

# ╔═╡ 741398fd-cfb6-46af-9bd0-ad047f8f4187
gr()

# ╔═╡ ca064da4-ab88-468e-a2cb-5b76debf6e95
begin
	iterations = 1000
	anim = @animate for i in 1:iterations
		@info "Iteration $i"
		for k in 1:length(grid.elements)
			left_index = k == 1 ? grid.K : k-1
			
			left_element = u_basis.elements[left_index]
			current_element = u_basis.elements[k]
	
			u_basis_local = [left_element.data[end], current_element.data...]
			current_element.data .= linear_operator * u_basis_local
		end
		
		pp = plot()
		for (ge, ue) in zip(grid.elements, u_basis.elements)
			plot!(pp, [ge.xl, ge.xr], ue.data, linewidth=2, label="")
		end
	end every 5
	mp4(anim)
end

# ╔═╡ Cell order:
# ╠═29443685-beb0-4102-a46d-63c48b008742
# ╟─bdeabaa3-8a91-4f9d-8cb7-4527a6385438
# ╟─dbbb3fea-9dea-11eb-1f7b-afb4a0c5114e
# ╟─35708c02-2ef3-432f-963b-62b77f072b19
# ╟─14c235cf-7b61-45b2-85a8-3d8af45b4edb
# ╟─6be6e442-6983-4629-b734-e6487ac8001d
# ╠═ec7aeb56-230b-4d05-a561-632f71c09e02
# ╠═34a301b0-58b4-4617-ad47-bd7883b09d8b
# ╠═d7c33092-ca55-4e1f-92be-d62aef900393
# ╟─28c5d715-c14a-4573-81b5-1e835bc4926c
# ╠═d75f11b0-e29b-41d6-87bb-108b0d3cb468
# ╠═3f8100e7-dd79-44eb-864c-45ae08c6445a
# ╠═b52a8325-a0e0-4697-81b4-26abfbe70a98
# ╟─45016410-7e25-4f5b-82c5-aa665aaa3c7f
# ╠═2f617585-62c1-4f6c-9088-aed874dc43e0
# ╟─4de78270-1f76-4f66-9271-34d0f148bc0c
# ╟─98e97ab0-6623-4a9e-a9b9-fb80ca182f53
# ╟─c22c283e-615e-4964-9ff1-f9102f6862b2
# ╟─520fcb4f-bffc-4d53-97b1-703dfaf0b8ca
# ╠═1096111c-9281-46bc-b5a4-4178a9855cad
# ╟─16e1782b-3948-4d9b-8bfd-fb3134211977
# ╠═7bc02a02-947b-4526-8bed-49007f3fa8a4
# ╠═92bdd228-617f-42da-b3d9-b9e9e18a7e4b
# ╠═50142bc9-3dcd-4817-b9a6-3c4f37b292b1
# ╠═4a103139-e2f9-4182-89b0-9b0b3b4b2503
# ╠═5c7f8b83-1830-4466-82a2-dd2a6da4115b
# ╠═2d955948-bf0e-4f89-8bfa-2ffa6b9ac328
# ╠═aed204c6-d68d-449a-aee7-b5fcc537b81a
# ╠═b7d972b4-82de-4c81-8a9a-80b98850763f
# ╠═741398fd-cfb6-46af-9bd0-ad047f8f4187
# ╠═ca064da4-ab88-468e-a2cb-5b76debf6e95
