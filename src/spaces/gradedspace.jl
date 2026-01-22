
struct GradedSpace{I<:Sector,D} <: ElementarySpace
    dims::D # TODO: Tuple
    dual::Bool
end
sectortype(::Type{<:GradedSpace{I}}) where {I<:Sector} = I

function GradedSpace{I,NTuple{2,Int}}(dims; dual::Bool=false) where {I<:Z2Irrep}
    N = 2
    d = ntuple(n -> 0, N)
    isset = ntuple(n -> false, N)
    for (c, dc) in dims
        k = convert(I, c)
        i = k.n + 1 # findindex(values(I), k)
        k = dc < 0 && throw(ArgumentError("Sector $k has negative dimension $dc"))
        isset[i] && throw(ArgumentError("Sector $c appears multiple times"))
        isset = TupleTools.setindex(isset, true, i)
        d = TupleTools.setindex(d, dc, i)
    end
    return GradedSpace{I,NTuple{N,Int}}(d, dual)
end
function GradedSpace{I,NTuple{2,Int}}(dims::Pair; dual::Bool=false) where {I<:Z2Irrep}
    N = 2
    return GradedSpace{I,NTuple{N,Int}}((dims,); dual=dual)
end

GradedSpace{I,D}(; kwargs...) where {I<:Sector,D} = GradedSpace{I,D}((); kwargs...)
function GradedSpace{I,D}(d1::Pair, d2::Pair, dims::Vararg{Pair}; kwargs...) where {I<:Sector,D}
    return GradedSpace{I,D}((d1, d2, dims...); kwargs...)
end

function GradedSpace(dims::Tuple{Pair{I,<:Integer},Vararg{Pair{I,<:Integer}}}; dual::Bool=false) where {I<:Sector}
    N = length(values(I))
    return GradedSpace{I,NTuple{N,Int}}(dims; dual=dual)
end
function GradedSpace(dim1::Pair{I,<:Integer}, rdims::Vararg{Pair{I,<:Integer}}; dual::Bool=false) where {I<:Sector}
    N = length(values(I))
    return GradedSpace{I,NTuple{N,Int}}((dim1, rdims...); dual=dual)
end
function GradedSpace(dims::AbstractDict{I,<:Integer}; dual::Bool=false) where {I<:Sector}
    N = length(values(I))
    return GradedSpace{I,NTuple{N,Int}}(dims; dual=dual)
end
# not inferrable
GradedSpace(g::Base.Generator; dual::Bool=false) = GradedSpace(g...; dual=dual)
GradedSpace(g::AbstractDict; dual::Bool=false) = GradedSpace(g...; dual=dual)

Base.hash(V::GradedSpace, h::UInt) = hash(V.dual, hash(V.dims, h))

function dim(V::GradedSpace)
    return reduce(+, dim(V, c) * dim(c) for c in sectors(V);
                  init=zero(dim(one(sectortype(V)))))
end
function dim(V::GradedSpace{I,<:Tuple}, c::I) where {I<:Sector}
    return V.dims[c.n+1]
end

function sectors(V::GradedSpace{I,NTuple{N,Int}}) where {I<:Sector,N}
    return [Z2Irrep(i-1) for i in 1:2 if V.dims[i] != 0]
end

hassector(V::GradedSpace{I}, s::I) where {I<:Sector} = dim(V, s) != 0

Base.conj(V::GradedSpace) = typeof(V)(V.dims, !V.dual)
isdual(V::GradedSpace) = V.dual

# equality / comparison
function Base.:(==)(V₁::GradedSpace, V₂::GradedSpace)
    return sectortype(V₁) == sectortype(V₂) && (V₁.dims == V₂.dims) && V₁.dual == V₂.dual
end

Base.oneunit(S::Type{<:GradedSpace{I}}) where {I<:Sector} = S(one(I) => 1)
Base.zero(S::Type{<:GradedSpace{I}}) where {I<:Sector} = S(one(I) => 0)
Base.oneunit(V::GradedSpace) = oneunit(typeof(V))
Base.zero(V::GradedSpace) = zero(typeof(V))

# TODO: the following methods can probably be implemented more efficiently for
# `FiniteGradedSpace`, but we don't expect them to be used often in hot loops, so
# these generic definitions (which are still quite efficient) are good for now.
function ⊕(V₁::GradedSpace{I}, V₂::GradedSpace{I}) where {I<:Sector}
    dual1 = isdual(V₁)
    dual1 == isdual(V₂) ||
        throw(SpaceMismatch("Direct sum of a vector space and a dual space does not exist"))
    dims = SectorDict{I,Int}()
    for c in union(sectors(V₁), sectors(V₂))
        cout = ifelse(dual1, dual(c), c)
        dims[cout] = dim(V₁, c) + dim(V₂, c)
    end
    return typeof(V₁)(dims; dual=dual1)
end
⊕(V₁::GradedSpace{I}, V₂::GradedSpace{I}, Vs::GradedSpace{I}...) where {I<:Sector} = ⊕(⊕(V₁, V₂), Vs...)
⊕(V::GradedSpace) = V

function fuse(V₁::GradedSpace{I}, V₂::GradedSpace{I}) where {I<:Sector}
    dims = SectorDict{I,Int}()
    for a in sectors(V₁), b in sectors(V₂)
        c = a ⊗ b
        dims[c] = get(dims, c, 0) + Nsymbol(a, b, c) * dim(V₁, a) * dim(V₂, b)
        # for c in a ⊗ b
        #     dims[c] = get(dims, c, 0) + Nsymbol(a, b, c) * dim(V₁, a) * dim(V₂, b)
        # end
    end
    return typeof(V₁)(dims)
end
function fuse(V₁::GradedSpace{I}, V₂::GradedSpace{I}, V₃::GradedSpace{I}...) where {I<:Sector}
    return fuse(fuse(V₁, V₂), V₃...)
end
fuse(V::GradedSpace{I}) where {I<:Sector} = V

function infimum(V₁::GradedSpace{I}, V₂::GradedSpace{I}) where {I<:Sector}
    if V₁.dual == V₂.dual
        typeof(V₁)(c => min(dim(V₁, c), dim(V₂, c))
                   for c in
                       union(sectors(V₁), sectors(V₂)), dual in V₁.dual)
    else
        throw(SpaceMismatch("Infimum of space and dual space does not exist"))
    end
end

function supremum(V₁::GradedSpace{I}, V₂::GradedSpace{I}) where {I<:Sector}
    if V₁.dual == V₂.dual
        typeof(V₁)(c => max(dim(V₁, c), dim(V₂, c))
                   for c in
                       union(sectors(V₁), sectors(V₂)), dual in V₁.dual)
    else
        throw(SpaceMismatch("Supremum of space and dual space does not exist"))
    end
end




const Z2Space = GradedSpace{Z2Irrep,NTuple{2,Int}}
Z2Space(dims::NTuple{2,Int}; dual::Bool=false) = Z2Space(dims, dual)
Z2Space(dims::Vararg{Int,2}; dual::Bool=false) = Z2Space(dims, dual)

