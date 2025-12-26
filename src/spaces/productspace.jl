
struct ProductSpace{S<:ElementarySpace,N} <: CompositeSpace{S}
    spaces::NTuple{N,S}
    ProductSpace{S,N}(spaces::NTuple{N,S}) where {S<:ElementarySpace,N} = new{S,N}(spaces)
end
function ProductSpace{S,N}(spaces::Vararg{S,N}) where {S<:ElementarySpace,N}
    return ProductSpace{S,N}(spaces)
end

function ProductSpace{S}(spaces::Tuple{Vararg{S}}) where {S<:ElementarySpace}
    return ProductSpace{S,length(spaces)}(spaces)
end
ProductSpace{S}(spaces::Vararg{S}) where {S<:ElementarySpace} = ProductSpace{S}(spaces)

function ProductSpace(spaces::Tuple{S,Vararg{S}}) where {S<:ElementarySpace}
    return ProductSpace{S,length(spaces)}(spaces)
end
function ProductSpace(space1::ElementarySpace, rspaces::Vararg{ElementarySpace})
    return ProductSpace((space1, rspaces...))
end

ProductSpace(P::ProductSpace) = P

# constructors with conversion behaviour
function ProductSpace{S,N}(V::Vararg{ElementarySpace,N}) where {S<:ElementarySpace,N}
    return ProductSpace{S,N}(V)
end
function ProductSpace{S}(V::Vararg{ElementarySpace}) where {S<:ElementarySpace}
    return ProductSpace{S}(V)
end

function ProductSpace{S,N}(V::Tuple{Vararg{ElementarySpace,N}}) where {S<:ElementarySpace,N}
    return ProductSpace{S}(convert.(S, V))
end
function ProductSpace{S}(V::Tuple{Vararg{ElementarySpace}}) where {S<:ElementarySpace}
    return ProductSpace{S}(convert.(S, V))
end
function ProductSpace(V::Tuple{ElementarySpace,Vararg{ElementarySpace}})
    return ProductSpace(promote(V...))
end

# Corresponding methods
#-----------------------
dims(P::ProductSpace) = map(dim, P.spaces)
dim(P::ProductSpace, n::Int) = dim(P.spaces[n])
dim(P::ProductSpace) = prod(dims(P))

dual(P::ProductSpace{<:ElementarySpace,0}) = P
dual(P::ProductSpace) = ProductSpace(map(dual, reverse(P.spaces)))

# more specific methods

sectors(P::ProductSpace) = _sectors(P, sectortype(P))
function _sectors(P::ProductSpace{<:ElementarySpace,N}, ::Type{<:Sector}) where {N}
    return product(map(sectors, P.spaces)...)
end


function hassector(V::ProductSpace{<:ElementarySpace,N}, s::NTuple{N}) where {N}
    return reduce(&, map(hassector, V.spaces, s); init=true)
end


function dims(P::ProductSpace{<:ElementarySpace,N}, sector::NTuple{N,<:Sector}) where {N}
    return map(dim, P.spaces, sector)
end

function dim(P::ProductSpace{<:ElementarySpace,N}, sector::NTuple{N,<:Sector}) where {N}
    return reduce(*, dims(P, sector); init=1)
end

function blocksectors(P::ProductSpace{S,N}) where {S,N}
    I = sectortype(S)
    # if I == Trivial
    #     return OneOrNoneIterator(dim(P) != 0, Trivial())
    # end
    bs = Vector{I}()
    if N == 0
        push!(bs, one(I))
    elseif N == 1
        for s in sectors(P)
            push!(bs, first(s))
        end
    else
        for s in sectors(P)
            c = ⊗(s...)
            if !(c in bs)
                push!(bs, c)
            end
            # for c in ⊗(s...)
            #     if !(c in bs)
            #         push!(bs, c)
            #     end
            # end
        end
    end
    return sort!(bs)
end


function fusiontrees(P::ProductSpace{S,N}, blocksector::I) where {S,N,I}
    I == sectortype(S) || throw(SectorMismatch())
    uncoupled = map(sectors, P.spaces)
    isdualflags = map(isdual, P.spaces)
    return fusiontrees(uncoupled, blocksector)
end

hasblock(P::ProductSpace, c::Sector) = !isempty(fusiontrees(P, c))

function blockdim(P::ProductSpace, c::Sector)
    sectortype(P) == typeof(c) || throw(SectorMismatch())
    d = 0
    for f in fusiontrees(P, c)
        d += dim(P, f.uncoupled)
    end
    return d
end

function Base.:(==)(P1::ProductSpace{S,N}, P2::ProductSpace{S,N}) where {S<:ElementarySpace,N}
    return (P1.spaces == P2.spaces)
end
Base.:(==)(P1::ProductSpace, P2::ProductSpace) = false

# hashing S is necessary to have different hashes for empty productspace with different S
Base.hash(P::ProductSpace{S}, h::UInt) where {S} = hash(P.spaces, hash(S, h))

# Default construction from product of spaces
#---------------------------------------------
⊗(V::ElementarySpace, Vrest::ElementarySpace...) = ProductSpace(V, Vrest...)
⊗(P::ProductSpace) = P
function ⊗(P1::ProductSpace{S}, P2::ProductSpace{S}) where {S<:ElementarySpace}
    return ProductSpace{S}(tuple(P1.spaces..., P2.spaces...))
end
function ⊗(P::ProductSpace{S}, V::S) where {S<:ElementarySpace}
    return ProductSpace{S}(tuple(P.spaces..., V))
end


# unit element with respect to the monoidal structure of taking tensor products
Base.one(V::VectorSpace) = one(typeof(V))
Base.one(::Type{<:ProductSpace{S}}) where {S<:ElementarySpace} = ProductSpace{S,0}(())
Base.one(::Type{S}) where {S<:ElementarySpace} = ProductSpace{S,0}(())

Base.:^(V::ElementarySpace, N::Int) = ProductSpace{typeof(V),N}(ntuple(n -> V, N))
Base.:^(V::ProductSpace, N::Int) = ⊗(ntuple(n -> V, N)...)
function Base.literal_pow(::typeof(^), V::ElementarySpace, p::Val{N}) where {N}
    return ProductSpace{typeof(V),N}(ntuple(n -> V, p))
end

fuse(P::ProductSpace{S,0}) where {S<:ElementarySpace} = oneunit(S)
fuse(P::ProductSpace{S}) where {S<:ElementarySpace} = fuse(P.spaces...)

# Functionality for extracting and iterating over spaces
#--------------------------------------------------------
Base.length(P::ProductSpace) = length(P.spaces)
Base.getindex(P::ProductSpace, n::Integer) = P.spaces[n]

Base.iterate(P::ProductSpace, args...) = Base.iterate(P.spaces, args...)
Base.indexed_iterate(P::ProductSpace, args...) = Base.indexed_iterate(P.spaces, args...)

Base.eltype(::Type{<:ProductSpace{S}}) where {S<:ElementarySpace} = S
Base.eltype(P::ProductSpace) = eltype(typeof(P))

Base.IteratorEltype(::Type{<:ProductSpace}) = Base.HasEltype()
Base.IteratorSize(::Type{<:ProductSpace}) = Base.HasLength()

Base.reverse(P::ProductSpace) = ProductSpace(reverse(P.spaces))




# Promotion and conversion
# ------------------------
function Base.promote_rule(::Type{S}, ::Type{<:ProductSpace{S}}) where {S<:ElementarySpace}
    return ProductSpace{S}
end


# ElementarySpace to ProductSpace
Base.convert(::Type{<:ProductSpace}, V::S) where {S<:ElementarySpace} = ⊗(V)
