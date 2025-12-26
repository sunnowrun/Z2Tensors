
struct AdjointTensorMap{T,S,N₁,N₂,TT<:AbstractTensorMap{T,S,N₂,N₁}} <:
       AbstractTensorMap{T,S,N₁,N₂}
    parent::TT
end
Base.parent(t::AdjointTensorMap) = t.parent

# Constructor: construct from taking adjoint of a tensor
Base.adjoint(t::AdjointTensorMap) = parent(t)
Base.adjoint(t::AbstractTensorMap) = AdjointTensorMap(t)

# Properties
space(t::AdjointTensorMap) = adjoint(space(parent(t)))
dim(t::AdjointTensorMap) = dim(parent(t))
storagetype(::Type{AdjointTensorMap{T,S,N₁,N₂,TT}}) where {T,S,N₁,N₂,TT} = storagetype(TT)

# Blocks and subblocks
#----------------------
block(t::AdjointTensorMap, s::Sector) = block(parent(t), s)'

blocks(t::AdjointTensorMap) = BlockIterator(t, blocks(parent(t)))

function blocktype(::Type{AdjointTensorMap{T,S,N₁,N₂,TT}}) where {T,S,N₁,N₂,TT}
    return Base.promote_op(adjoint, blocktype(TT))
end

function Base.iterate(iter::BlockIterator{<:AdjointTensorMap}, state...)
    next = iterate(iter.structure, state...)
    isnothing(next) && return next
    (c, b), newstate = next
    return c => adjoint(b), newstate
end

function Base.getindex(iter::BlockIterator{<:AdjointTensorMap}, c::Sector)
    return adjoint(Base.getindex(iter.structure, c))
end

function Base.getindex(t::AdjointTensorMap{T,S,N₁,N₂},
                       f₁::FusionTree{I,N₁}, f₂::FusionTree{I,N₂}) where {T,S,N₁,N₂,I}
    tp = parent(t)
    subblock = getindex(tp, f₂, f₁)
    return permutedims(conj(subblock), (domainind(tp)..., codomainind(tp)...))
end
function Base.setindex!(t::AdjointTensorMap{T,S,N₁,N₂}, v,
                        f₁::FusionTree{I,N₁},
                        f₂::FusionTree{I,N₂}) where {T,S,N₁,N₂,I}
    return copy!(getindex(t, f₁, f₂), v)
end

