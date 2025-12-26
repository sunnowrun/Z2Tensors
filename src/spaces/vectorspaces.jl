

# VECTOR SPACES:
#==============================================================================#
abstract type VectorSpace end

Base.adjoint(V::VectorSpace) = dual(V)
sectortype(V::VectorSpace) = sectortype(typeof(V))
dual(V::VectorSpace) = conj(V)


# Hierarchy of elementary vector spaces
#---------------------------------------

abstract type ElementarySpace <: VectorSpace end
const IndexSpace = ElementarySpace

reduceddim(V::ElementarySpace) = sum(Base.Fix1(dim, V), sectors(V); init=0)


# Composite vector spaces
#-------------------------
abstract type CompositeSpace{S<:ElementarySpace} <: VectorSpace end


spacetype(S::Type{<:ElementarySpace}) = S
spacetype(V::ElementarySpace) = typeof(V) # = spacetype(typeof(V))
spacetype(::Type{<:CompositeSpace{S}}) where {S} = S
spacetype(V::CompositeSpace) = spacetype(typeof(V)) # = spacetype(typeof(V))

# field(P::Type{<:CompositeSpace}) = field(spacetype(P))
sectortype(P::Type{<:CompositeSpace}) = sectortype(spacetype(P))

# make ElementarySpace instances behave similar to ProductSpace instances
blocksectors(V::ElementarySpace) = collect(sectors(V))
blockdim(V::ElementarySpace, c::Sector) = dim(V, c)



include("gradedspace.jl")
include("productspace.jl")
include("homspace.jl")



# Partial order for vector spaces
#---------------------------------
function isisomorphic(V₁::VectorSpace, V₂::VectorSpace)
    spacetype(V₁) == spacetype(V₂) || return false
    for c in union(blocksectors(V₁), blocksectors(V₂))
        if blockdim(V₁, c) != blockdim(V₂, c)
            return false
        end
    end
    return true
end

function ismonomorphic(V₁::VectorSpace, V₂::VectorSpace)
    spacetype(V₁) == spacetype(V₂) || return false
    for c in blocksectors(V₁)
        if blockdim(V₁, c) > blockdim(V₂, c)
            return false
        end
    end
    return true
end

function isepimorphic(V₁::VectorSpace, V₂::VectorSpace)
    spacetype(V₁) == spacetype(V₂) || return false
    for c in blocksectors(V₂)
        if blockdim(V₁, c) < blockdim(V₂, c)
            return false
        end
    end
    return true
end

# unicode alternatives
const ≅ = isisomorphic
const ≾ = ismonomorphic
const ≿ = isepimorphic

≺(V₁::VectorSpace, V₂::VectorSpace) = V₁ ≾ V₂ && !(V₁ ≿ V₂)
≻(V₁::VectorSpace, V₂::VectorSpace) = V₁ ≿ V₂ && !(V₁ ≾ V₂)




infimum(V₁::S, V₂::S, V₃::S...) where {S<:ElementarySpace} = infimum(infimum(V₁, V₂), V₃...)

function supremum(V₁::S, V₂::S, V₃::S...) where {S<:ElementarySpace}
    return supremum(supremum(V₁, V₂), V₃...)
end
