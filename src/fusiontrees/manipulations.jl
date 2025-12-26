fusiontreedict(I) = SingletonDict


@inline function split(f::FusionTree{I,N}, M::Int) where {I,N}
    if M > N || M < 0
        throw(ArgumentError("M should be between 0 and N = $N"))
    elseif M === N
        (f, FusionTree{I}((f.coupled,), f.coupled))
    elseif M === 1
        f₁ = FusionTree{I}((f.uncoupled[1],), f.uncoupled[1])
        f₂ = FusionTree{I}(f.uncoupled, f.coupled)
        return f₁, f₂
    elseif M === 0
        f₁ = FusionTree{I}((), one(I))
        uncoupled2 = (one(I), f.uncoupled...)
        coupled2 = f.coupled
        return f₁, FusionTree{I}(uncoupled2, coupled2)
    else
        uncoupled1 = ntuple(n -> f.uncoupled[n], M)
        coupled1 = couple(uncoupled1)

        uncoupled2 = ntuple(N - M + 1) do n
            return n == 1 ? coupled1 : f.uncoupled[M + n - 1]
        end
        coupled2 = f.coupled
        f₁ = FusionTree{I}(uncoupled1, coupled1)
        f₂ = FusionTree{I}(uncoupled2, coupled2)
        return f₁, f₂
    end
end

function permute(f1::FusionTree{ZNIrrep{2}}, f2::FusionTree{ZNIrrep{2}},
                            p1::IndexTuple{N₁}, p2::IndexTuple{N₂}) where {N₁, N₂}
    uncoupled = (f1.uncoupled..., dual.(f2.uncoupled)...)
    uncoupled1′, uncoupled2′ = TupleTools.getindices(uncoupled, p1), TupleTools.getindices(uncoupled, p2)
    uncoupled2′ = ntuple(i->dual(uncoupled2′[i]), Val(N₂))

    coupled1′ = ⊗(uncoupled1′...)
    coupled2′ = ⊗(uncoupled2′...)

    f1′ = FusionTree(uncoupled1′, coupled1′)
    f2′ = FusionTree(uncoupled2′, coupled2′)
    # compute the sign
    coeff = (coupled1′ == coupled2′) ? 1 : 0
    return fusiontreedict(ZNIrrep{2})((f1′, f2′) => coeff)
end
