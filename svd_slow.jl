# This file is a slight modification from that found in Julia v0.6.4

"""
    svdfact_slow!(A, thin::Bool=true) -> (u, s, vt) # SVD

`svdfact!` is the same as [`svdfact`](@ref), but saves space by
overwriting the input `A`, instead of creating a copy.
"""
function svdfact_slow!(A; thin::Bool = true)
    m, n = size(A)
    T = typeof(A)
    if m == 0 || n == 0
        u, s, vt = (eye(T, m, thin ? n : m), real(zeros(T, 0)), eye(T, n, n))
    else
        c = thin ? 'S' : 'A'
        u, s, vt = LAPACK.gesvd!(c, c, A)
    end
    return u, s, vt # SVD(u, s, vt)
end

# """
#     svdfact_slow(A; thin::Bool=true) -> SVD
#
# Compute the singular value decomposition (SVD) of `A` and return an `SVD` object.
#
# `U`, `S`, `V` and `Vt` can be obtained from the factorization `F` with `F[:U]`,
# `F[:S]`, `F[:V]` and `F[:Vt]`, such that `A = U*diagm(S)*Vt`.
# The algorithm produces `Vt` and hence `Vt` is more efficient to extract than `V`.
# The singular values in `S` are sorted in descending order.
#
# If `thin=true` (default), a thin SVD is returned. For a ``M \\times N`` matrix
# `A`, `U` is ``M \\times M`` for a full SVD (`thin=false`) and
# ``M \\times \\min(M, N)`` for a thin SVD.
#
# # Example
# ```jldoctest
# julia> A = [1. 0. 0. 0. 2.; 0. 0. 3. 0. 0.; 0. 0. 0. 0. 0.; 0. 2. 0. 0. 0.]
# 4×5 Array{Float64,2}:
#  1.0  0.0  0.0  0.0  2.0
#  0.0  0.0  3.0  0.0  0.0
#  0.0  0.0  0.0  0.0  0.0
#  0.0  2.0  0.0  0.0  0.0
#
# julia> F = svdfact(A)
# Base.LinAlg.SVD{Float64,Float64,Array{Float64,2}}([0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 -1.0; 0.0 0.0 1.0 0.0], [3.0, 2.23607, 2.0, 0.0], [-0.0 0.0 … -0.0 0.0; 0.447214 0.0 … 0.0 0.894427; -0.0 1.0 … -0.0 0.0; 0.0 0.0 … 1.0 0.0])
#
# julia> F[:U] * diagm(F[:S]) * F[:Vt]
# 4×5 Array{Float64,2}:
#  1.0  0.0  0.0  0.0  2.0
#  0.0  0.0  3.0  0.0  0.0
#  0.0  0.0  0.0  0.0  0.0
#  0.0  2.0  0.0  0.0  0.0
# ```
# """
# function svdfact_slow(A::StridedVecOrMat{T}; thin::Bool = true) where {T}
#     S = promote_type(Float32, typeof(one(T) / norm(one(T))))
#     u, s, vt = svdfact_slow!(copy_oftype(A, S), thin = thin)
#     return u, s, vt
# end
# svdfact_slow(x::Number; thin::Bool = true) = SVD(
#     x == 0 ? fill(one(x), 1, 1) : fill(x / abs(x), 1, 1),
#     [abs(x)],
#     fill(one(x), 1, 1),
# )
# svdfact_slow(x::Integer; thin::Bool = true) =
#     svdfact_slow(float(x), thin = thin)

"""
    svd_slow(A; thin::Bool=true) -> U, S, V

Computes the SVD of `A`, returning `U`, vector `S`, and `V` such that
`A == U*diagm(S)*V'`. The singular values in `S` are sorted in descending order. The gesvd! algorithm is used instead of gesdd!.

If `thin=true` (default), a thin SVD is returned. For a ``M \\times N`` matrix
`A`, `U` is ``M \\times M`` for a full SVD (`thin=false`) and
``M \\times \\min(M, N)`` for a thin SVD.

`svd` is a wrapper around [`svdfact`](@ref), extracting all parts
of the `SVD` factorization to a tuple. Direct use of `svdfact` is therefore more
efficient.

# Example

```jldoctest
julia> A = [1. 0. 0. 0. 2.; 0. 0. 3. 0. 0.; 0. 0. 0. 0. 0.; 0. 2. 0. 0. 0.]
4×5 Array{Float64,2}:
 1.0  0.0  0.0  0.0  2.0
 0.0  0.0  3.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  2.0  0.0  0.0  0.0

julia> U, S, V = svd(A)
([0.0 1.0 0.0 0.0; 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 -1.0; 0.0 0.0 1.0 0.0], [3.0, 2.23607, 2.0, 0.0], [-0.0 0.447214 -0.0 0.0; 0.0 0.0 1.0 0.0; … ; -0.0 0.0 -0.0 1.0; 0.0 0.894427 0.0 0.0])

julia> U*diagm(S)*V'
4×5 Array{Float64,2}:
 1.0  0.0  0.0  0.0  2.0
 0.0  0.0  3.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0
 0.0  2.0  0.0  0.0  0.0
```
"""
function svd_slow(M::Union{Number,AbstractArray}; thin::Bool = true)
    A = deepcopy(M)
    u, s, vt = svdfact_slow!(A, thin = thin)
    return u, s, vt' # F.U, F.S, F.Vt'
end

# function getindex(F::SVD, d::Symbol)
#     if d == :U
#         return F.U
#     elseif d == :S
#         return F.S
#     elseif d == :Vt
#         return F.Vt
#     elseif d == :V
#         return F.Vt'
#     else
#         throw(KeyError(d))
#     end
# end
#
# """
#     svdvals_slow!(A)
#
# Returns the singular values of `A`, saving space by overwriting the input.
# See also [`svdvals`](@ref).
# """
# svdvals_slow!(
#     A::StridedMatrix{T},
# ) where {T<:Union{Complex64,Complex128,Float32,Float64}} =
#     findfirst(size(A), 0) > 0 ? zeros(T, 0) : LAPACK.gesvd!('N', 'N', A)[2]
# svdvals_slow(A::AbstractMatrix{<:Union{Complex64,Complex128,Float32,Float64}}) = svdvals_slow!(copy(A))
#
# """
#     svdvals_slow(A)
#
# Returns the singular values of `A` in descending order.
#
# # Example
#
# ```jldoctest
# julia> A = [1. 0. 0. 0. 2.; 0. 0. 3. 0. 0.; 0. 0. 0. 0. 0.; 0. 2. 0. 0. 0.]
# 4×5 Array{Float64,2}:
#  1.0  0.0  0.0  0.0  2.0
#  0.0  0.0  3.0  0.0  0.0
#  0.0  0.0  0.0  0.0  0.0
#  0.0  2.0  0.0  0.0  0.0
#
# julia> svdvals(A)
# 4-element Array{Float64,1}:
#  3.0
#  2.23607
#  2.0
#  0.0
# ```
# """
# function svdvals_slow(A::AbstractMatrix{T}) where {T}
#     S = promote_type(Float32, typeof(one(T) / norm(one(T))))
#     svdvals_slow!(copy_oftype(A, S))
# end
# svdvals_slow(x::Number) = abs(x)
# svdvals_slow(S::SVD{<:Any,T}) where {T} = (S[:S])::Vector{T}
