
function couple(uncoupled::NTuple{N,I}) where {N,I<:Sector}
    ⊗(uncoupled...)
    # first(⊗(uncoupled...))
end
function couple(uncoupled::NTuple{0})
    one(Z2Irrep)
end



function fusiontrees(uncoupled::NTuple{N,I}, coupled::I) where {N,I<:Sector}
    if couple(uncoupled) == coupled
        return (FusionTree{I,N}(uncoupled, coupled), )
    else
        return ()
    end
end
function fusiontrees(uncoupled::NTuple{0,I}, coupled::I) where {I<:Sector}
    if one(I) == coupled
        return (FusionTree{I,0}(uncoupled, coupled), )
    else
        return ()
    end
end
function fusiontrees(uncoupleds::Tuple, coupled::I) where {I<:Sector}
    trees = ((fusiontrees(uncoupled, coupled) for uncoupled in Iterators.product(uncoupleds...))...,)
    return TupleTools.flatten(trees)
end


