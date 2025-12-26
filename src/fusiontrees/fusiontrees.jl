
struct FusionTree{I<:Sector,N}
    uncoupled::NTuple{N,I}
    coupled::I
    function FusionTree{I,N}(uncoupled::NTuple{N,I}, coupled::I) where {I<:Sector,N}
        return new{I,N}(uncoupled, coupled)
    end
end

function FusionTree{I}(uncoupled::NTuple{N,I}, coupled=one(I)) where {I<:Sector,N}
    return FusionTree{I,N}(map(s -> convert(I, s), uncoupled), convert(I, coupled))
end
FusionTree(uncoupled::NTuple{N,I}, coupled::I = unit(I)) where {I<:Sector,N} = FusionTree{I,N}(uncoupled, coupled)
# FusionTree(uncoupled::NTuple{N,I}) where {I<:Sector,N} = FusionTree{I,N}(uncoupled, one(I))



# Properties
sectortype(::Type{<:FusionTree{I}}) where {I<:Sector} = I
# FusionStyle(::Type{<:FusionTree{I}}) where {I<:Sector} = FusionStyle(I)
# BraidingStyle(::Type{<:FusionTree{I}}) where {I<:Sector} = BraidingStyle(I)
Base.length(::Type{<:FusionTree{<:Sector,N}}) where {N} = N

sectortype(f::FusionTree) = sectortype(typeof(f))
# FusionStyle(f::FusionTree) = FusionStyle(typeof(f))
# BraidingStyle(f::FusionTree) = BraidingStyle(typeof(f))
Base.length(f::FusionTree) = length(typeof(f))

# Hashing, important for using fusion trees as key in a dictionary
function Base.hash(f::FusionTree{I}, h::UInt) where {I}
    h = hash(f.coupled, hash(f.uncoupled, h))
    return h
end
function Base.:(==)(f₁::FusionTree{I,N}, f₂::FusionTree{I,N}) where {I<:Sector,N}
    f₁.coupled == f₂.coupled || return false
    @inbounds for i in 1:N
        f₁.uncoupled[i] == f₂.uncoupled[i] || return false
    end
    return true
end
Base.:(==)(f₁::FusionTree, f₂::FusionTree) = false

# Facilitate getting correct fusion tree types
function fusiontreetype(::Type{I}, N::Int) where {I<:Sector}
    return FusionTree{I,N}
end



# Manipulate fusion trees
include("manipulations.jl")

# Fusion tree iterators
include("iterator.jl")

# auxiliary routines
# _abelianinner: generate the inner indices for given outer indices in the abelian case
# _abelianinner(outer::Tuple{}) = ()
# function _abelianinner(outer::Tuple{I}) where {I<:Sector}
#     return isone(outer[1]) ? () : throw(SectorMismatch())
# end
# function _abelianinner(outer::Tuple{I,I}) where {I<:Sector}
#     return outer[1] == dual(outer[2]) ? () : throw(SectorMismatch())
# end
# function _abelianinner(outer::Tuple{I,I,I}) where {I<:Sector}
#     return isone(first(⊗(outer...))) ? () : throw(SectorMismatch())
# end
# function _abelianinner(outer::Tuple{I,I,I,I,Vararg{I}}) where {I<:Sector}
#     c = first(outer[1] ⊗ outer[2])
#     return (c, _abelianinner((c, TupleTools.tail2(outer)...))...)
# end
