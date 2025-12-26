function linearizepermutation(p1::NTuple{N₁,Int}, p2::NTuple{N₂},
                              n₁::Int, n₂::Int) where {N₁,N₂}
    p1′ = ntuple(Val(N₁)) do n
        return p1[n] > n₁ ? n₂ + 2n₁ + 1 - p1[n] : p1[n]
    end
    p2′ = ntuple(Val(N₂)) do n
        return p2[N₂ + 1 - n] > n₁ ? n₂ + 2n₁ + 1 - p2[N₂ + 1 - n] : p2[N₂ + 1 - n]
    end
    return (p1′..., p2′...)
end

function permutation2swaps(perm)
    p = collect(perm)
    @assert isperm(p)
    swaps = Vector{Int}()
    N = length(p)
    for k in 1:(N - 1)
        append!(swaps, (p[k] - 1):-1:k)
        for l in (k + 1):N
            if p[l] < p[k]
                p[l] += 1
            end
        end
        p[k] = k
    end
    return swaps
end


@noinline _boundserror(P, i) = throw(BoundsError(P, i))
@noinline _nontrivialspaceerror(P, i) = throw(ArgumentError(lazy"Attempting to remove a non-trivial space $(P[i])"))

const VecOrNumber{T<:Number} = Union{DenseVector{T},T}

"""
    _interleave(a::NTuple{N}, b::NTuple{N}) -> NTuple{2N}

Interleave two tuples of the same length.
"""
_interleave(::Tuple{}, ::Tuple{}) = ()
function _interleave(a::NTuple{N}, b::NTuple{N}) where {N}
    return (a[1], b[1], _interleave(tail(a), tail(b))...)
end
