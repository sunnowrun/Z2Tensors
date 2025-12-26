# abstracttensor.jl
#
# Abstract Tensor type
#----------------------

abstract type AbstractTensorMap{T<:Number,S<:IndexSpace,N₁,N₂} end

const AbstractTensor{T,S,N} = AbstractTensorMap{T,S,N,0}

Base.eltype(::Type{<:AbstractTensorMap{T}}) where {T} = T
spacetype(::Type{<:AbstractTensorMap{<:Any,S}}) where {S} = S
sectortype(::Type{TT}) where {TT<:AbstractTensorMap} = sectortype(spacetype(TT))

# function InnerProductStyle(::Type{TT}) where {TT<:AbstractTensorMap}
#     return InnerProductStyle(spacetype(TT))
# end

# field(::Type{TT}) where {TT<:AbstractTensorMap} = field(spacetype(TT))


similarstoragetype(TT::Type{<:AbstractTensorMap}) = similarstoragetype(TT, scalartype(TT))

function similarstoragetype(TT::Type{<:AbstractTensorMap}, ::Type{T}) where {T}
    return Core.Compiler.return_type(similar, Tuple{storagetype(TT),Type{T}})
end

# tensor characteristics: space and index information
#-----------------------------------------------------
space(t::AbstractTensorMap, i::Int) = space(t)[i]

codomain(t::AbstractTensorMap) = codomain(space(t))
codomain(t::AbstractTensorMap, i) = codomain(t)[i]
target(t::AbstractTensorMap) = codomain(t) # categorical terminology

domain(t::AbstractTensorMap) = domain(space(t))
domain(t::AbstractTensorMap, i) = domain(t)[i]
source(t::AbstractTensorMap) = domain(t) # categorical terminology

numout(::Type{<:AbstractTensorMap{T,S,N₁}}) where {T,S,N₁} = N₁
numin(::Type{<:AbstractTensorMap{T,S,N₁,N₂}}) where {T,S,N₁,N₂} = N₂
numind(::Type{TT}) where {TT<:AbstractTensorMap} = numin(TT) + numout(TT)
const order = numind

function codomainind(::Type{TT}) where {TT<:AbstractTensorMap}
    return ntuple(identity, numout(TT))
end
codomainind(t::AbstractTensorMap) = codomainind(typeof(t))

function domainind(::Type{TT}) where {TT<:AbstractTensorMap}
    return ntuple(n -> numout(TT) + n, numin(TT))
end
domainind(t::AbstractTensorMap) = domainind(typeof(t))

function allind(::Type{TT}) where {TT<:AbstractTensorMap}
    return ntuple(identity, numind(TT))
end
allind(t::AbstractTensorMap) = allind(typeof(t))

function adjointtensorindex(t::AbstractTensorMap, i)
    return ifelse(i <= numout(t), numin(t) + i, i - numout(t))
end

function adjointtensorindices(t::AbstractTensorMap, indices::IndexTuple)
    return map(i -> adjointtensorindex(t, i), indices)
end

function adjointtensorindices(t::AbstractTensorMap, p::Index2Tuple)
    return (adjointtensorindices(t, p[1]), adjointtensorindices(t, p[2]))
end

# tensor characteristics: work on instances and pass to type
#------------------------------------------------------------
spacetype(t::AbstractTensorMap) = spacetype(typeof(t))
sectortype(t::AbstractTensorMap) = sectortype(typeof(t))
# InnerProductStyle(t::AbstractTensorMap) = InnerProductStyle(typeof(t))
# field(t::AbstractTensorMap) = field(typeof(t))
storagetype(t::AbstractTensorMap) = storagetype(typeof(t))
blocktype(t::AbstractTensorMap) = blocktype(typeof(t))
similarstoragetype(t::AbstractTensorMap, T=scalartype(t)) = similarstoragetype(typeof(t), T)

numout(t::AbstractTensorMap) = numout(typeof(t))
numin(t::AbstractTensorMap) = numin(typeof(t))
numind(t::AbstractTensorMap) = numind(typeof(t))

# tensor characteristics: data structure and properties
#------------------------------------------------------
fusionblockstructure(t::AbstractTensorMap) = fusionblockstructure(space(t))

dim(t::AbstractTensorMap) = fusionblockstructure(t).totaldim

blocksectors(t::AbstractTensorMap) = keys(fusionblockstructure(t).blockstructure)

hasblock(t::AbstractTensorMap, c::Sector) = c ∈ blocksectors(t)

fusiontrees(t::AbstractTensorMap) = fusionblockstructure(t).fusiontreelist

# tensor data: block access
#---------------------------
function blocks(t::AbstractTensorMap)
    iter = Base.Iterators.map(blocksectors(t)) do c
        return c => block(t, c)
    end
    return iter
end

function blocktype(::Type{T}) where {T<:AbstractTensorMap}
    return Core.Compiler.return_type(block, Tuple{T,sectortype(T)})
end

# Derived indexing behavior for tensors with trivial symmetry
#-------------------------------------------------------------
# using TensorKit.Strided: SliceIndex
using Strided: SliceIndex

# Similar
#---------
# The implementation is written for similar(t, TorA, V::TensorMapSpace) -> TensorMap
# and all other methods are just filling in default arguments
# 4 arguments
@doc """
    similar(t::AbstractTensorMap, [AorT=storagetype(t)], [V=space(t)])
    similar(t::AbstractTensorMap, [AorT=storagetype(t)], codomain, domain)

Creates an uninitialized mutable tensor with the given scalar or storagetype `AorT` and
structure `V` or `codomain ← domain`, based on the source tensormap. The second and third
arguments are both optional, defaulting to the given tensor's `storagetype` and `space`.
The structure may be specified either as a single `HomSpace` argument or as `codomain` and
`domain`.

By default, this will result in `TensorMap{T}(undef, V)` when custom objects do not
specialize this method.
""" Base.similar(::AbstractTensorMap, args...)

function Base.similar(t::AbstractTensorMap, ::Type{T}, codomain::TensorSpace{S},
                      domain::TensorSpace{S}) where {T,S}
    return similar(t, T, codomain ← domain)
end
# 3 arguments
function Base.similar(t::AbstractTensorMap, codomain::TensorSpace{S},
                      domain::TensorSpace{S}) where {S}
    return similar(t, similarstoragetype(t), codomain ← domain)
end
function Base.similar(t::AbstractTensorMap, ::Type{T}, codomain::TensorSpace) where {T}
    return similar(t, T, codomain ← one(codomain))
end
# 2 arguments
function Base.similar(t::AbstractTensorMap, codomain::TensorSpace)
    return similar(t, similarstoragetype(t), codomain ← one(codomain))
end
Base.similar(t::AbstractTensorMap, P::TensorMapSpace) = similar(t, storagetype(t), P)
Base.similar(t::AbstractTensorMap, ::Type{T}) where {T} = similar(t, T, space(t))
# 1 argument
Base.similar(t::AbstractTensorMap) = similar(t, similarstoragetype(t), space(t))

# generic implementation for AbstractTensorMap -> returns `TensorMap`
function Base.similar(t::AbstractTensorMap, ::Type{TorA},
                      P::TensorMapSpace{S}) where {TorA,S}
    if TorA <: Number
        T = TorA
        A = similarstoragetype(t, T)
    elseif TorA <: DenseVector
        A = TorA
        T = scalartype(A)
    else
        throw(ArgumentError("Type $TorA not supported for similar"))
    end

    N₁ = length(codomain(P))
    N₂ = length(domain(P))
    return TensorMap{T,S,N₁,N₂,A}(undef, P)
end

# implementation in type-domain
function Base.similar(::Type{TT}, P::TensorMapSpace) where {TT<:AbstractTensorMap}
    return TensorMap{scalartype(TT)}(undef, P)
end
function Base.similar(::Type{TT}, cod::TensorSpace{S},
                      dom::TensorSpace{S}) where {TT<:AbstractTensorMap,S}
    return TensorMap{scalartype(TT)}(undef, cod, dom)
end

# Equality and approximality
#----------------------------
function Base.:(==)(t1::AbstractTensorMap, t2::AbstractTensorMap)
    (codomain(t1) == codomain(t2) && domain(t1) == domain(t2)) || return false
    for c in blocksectors(t1)
        block(t1, c) == block(t2, c) || return false
    end
    return true
end
function Base.hash(t::AbstractTensorMap, h::UInt)
    h = hash(codomain(t), h)
    h = hash(domain(t), h)
    for (c, b) in blocks(t)
        h = hash(c, hash(b, h))
    end
    return h
end

function Base.isapprox(t1::AbstractTensorMap, t2::AbstractTensorMap;
                       atol::Real=0,
                       rtol::Real=Base.rtoldefault(scalartype(t1), scalartype(t2), atol))
    d = norm(t1 - t2)
    if isfinite(d)
        return d <= max(atol, rtol * max(norm(t1), norm(t2)))
    else
        return false
    end
end

# Complex, real and imaginary
#----------------------------
function Base.complex(t::AbstractTensorMap)
    if scalartype(t) <: Complex
        return t
    else
        return copy!(similar(t, complex(scalartype(t))), t)
    end
end
function Base.complex(r::AbstractTensorMap{<:Real}, i::AbstractTensorMap{<:Real})
    return add(r, i, im * one(scalartype(i)))
end

function Base.real(t::AbstractTensorMap)
    if scalartype(t) <: Real
        return t
    else
        tr = similar(t, real(scalartype(t)))
        for (c, b) in blocks(t)
            block(tr, c) .= real(b)
        end
        return tr
    end
end
function Base.imag(t::AbstractTensorMap)
    if scalartype(t) <: Real
        return zerovector(t)
    else
        ti = similar(t, real(scalartype(t)))
        for (c, b) in blocks(t)
            block(ti, c) .= imag(b)
        end
        return ti
    end
end
