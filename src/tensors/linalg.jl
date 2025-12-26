# Basic algebra
#---------------
Base.copy(t::AbstractTensorMap) = Base.copy!(similar(t), t)

Base.:-(t::AbstractTensorMap) = VectorInterface.scale(t, -one(scalartype(t)))

Base.:+(t1::AbstractTensorMap, t2::AbstractTensorMap) = VectorInterface.add(t1, t2)
function Base.:-(t1::AbstractTensorMap, t2::AbstractTensorMap)
    return VectorInterface.add(t1, t2, -one(scalartype(t1)))
end

Base.:*(t::AbstractTensorMap, α::Number) = VectorInterface.scale(t, α)
Base.:*(α::Number, t::AbstractTensorMap) = VectorInterface.scale(t, α)

Base.:/(t::AbstractTensorMap, α::Number) = *(t, one(scalartype(t)) / α)
Base.:\(α::Number, t::AbstractTensorMap) = *(t, one(scalartype(t)) / α)

LinearAlgebra.normalize!(t::AbstractTensorMap, p::Real=2) = scale!(t, inv(norm(t, p)))
LinearAlgebra.normalize(t::AbstractTensorMap, p::Real=2) = scale(t, inv(norm(t, p)))

# destination allocation for matrix multiplication
function compose_dest(A::AbstractTensorMap, B::AbstractTensorMap)
    TC = TO.promote_contract(scalartype(A), scalartype(B), One)
    pA = (codomainind(A), domainind(A))
    pB = (codomainind(B), domainind(B))
    pAB = (codomainind(A), ntuple(i -> i + numout(A), numin(B)))
    return TO.tensoralloc_contract(TC,
                                   A, pA, false,
                                   B, pB, false,
                                   pAB, Val(false))
end

function compose(A::AbstractTensorMap, B::AbstractTensorMap)
    C = compose_dest(A, B)
    return mul!(C, A, B)
end
Base.:*(t1::AbstractTensorMap, t2::AbstractTensorMap) = compose(t1, t2)

Base.exp(t::AbstractTensorMap) = exp!(copy(t))
function Base.:^(t::AbstractTensorMap, p::Integer)
    return p < 0 ? Base.power_by_squaring(inv(t), -p) : Base.power_by_squaring(t, p)
end

# Special purpose constructors
#------------------------------
Base.zero(t::AbstractTensorMap) = VectorInterface.zerovector(t)
function Base.one(t::AbstractTensorMap)
    domain(t) == codomain(t) ||
        throw(SectorMismatch("no identity if domain and codomain are different"))
    return one!(similar(t))
end
function one!(t::AbstractTensorMap)
    domain(t) == codomain(t) ||
        throw(SectorMismatch("no identity if domain and codomain are different"))
    for (c, b) in blocks(t)
        one!(b)
    end
    return t
end


id(V::TensorSpace) = id(Float64, V)
function id(A::Type, V::TensorSpace{S}) where {S}
    W = V ← V
    N = length(codomain(W))
    dst = tensormaptype(S, N, N, A)(undef, W)
    return id!(dst)
end
const id! = one!

function isomorphism!(t::AbstractTensorMap)
    domain(t) ≅ codomain(t) ||
        throw(SpaceMismatch(lazy"domain and codomain are not isomorphic: $(space(t))"))
    for (_, b) in blocks(t)
        one!(b)
    end
    return t
end

# function unitary!(t::AbstractTensorMap)
#     # InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:unitary)
#     return isomorphism!(t)
# end

function isometry!(t::AbstractTensorMap)
    # InnerProductStyle(t) === EuclideanInnerProduct() ||
    #     throw_invalid_innerproduct(:isometry)
    domain(t) ≾ codomain(t) ||
        throw(SpaceMismatch(lazy"domain and codomain are not isometrically embeddable: $(space(t))"))
    for (_, b) in blocks(t)
        one!(b)
    end
    return t
end

# expand methods with default arguments
for morphism in (:isomorphism, :unitary, :isometry)
    morphism! = Symbol(morphism, :!)
    @eval begin
        $morphism(V::TensorMapSpace) = $morphism(Float64, V)
        $morphism(codomain::TensorSpace, domain::TensorSpace) = $morphism(codomain ← domain)
        function $morphism(A::Type, codomain::TensorSpace, domain::TensorSpace)
            return $morphism(A, codomain ← domain)
        end
        function $morphism(A::Type, V::TensorMapSpace{S,N₁,N₂}) where {S,N₁,N₂}
            t = tensormaptype(S, N₁, N₂, A)(undef, V)
            return $morphism!(t)
        end
        $morphism(t::AbstractTensorMap) = $morphism!(similar(t))
    end
end

# Diagonal tensors
# ----------------
# TODO: consider adding a specialised DiagonalTensorMap type
function LinearAlgebra.diag(t::AbstractTensorMap)
    return SectorDict(c => LinearAlgebra.diag(b) for (c, b) in blocks(t))
end
function LinearAlgebra.diagm(codom::VectorSpace, dom::VectorSpace, v::SectorDict)
    return TensorMap(SectorDict(c => LinearAlgebra.diagm(blockdim(codom, c),
                                                         blockdim(dom, c), b)
                                for (c, b) in v), codom ← dom)
end
LinearAlgebra.isdiag(t::AbstractTensorMap) = all(LinearAlgebra.isdiag ∘ last, blocks(t))

# In-place methods
#------------------
# Wrapping the blocks in a StridedView enables multithreading if JULIA_NUM_THREADS > 1
# TODO: reconsider this strategy, consider spawning different threads for different blocks

# Copy, adjoint and fill:
function Base.copy!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap)
    space(tdst) == space(tsrc) || throw(SpaceMismatch("$(space(tdst)) ≠ $(space(tsrc))"))
    for ((c, bdst), (_, bsrc)) in zip(blocks(tdst), blocks(tsrc))
        copy!(StridedView(bdst), StridedView(bsrc))
    end
    return tdst
end
function Base.copy!(tdst::TensorMap, tsrc::TensorMap)
    space(tdst) == space(tsrc) || throw(SpaceMismatch("$(space(tdst)) ≠ $(space(tsrc))"))
    copy!(tdst.data, tsrc.data)
    return tdst
end
function Base.fill!(t::AbstractTensorMap, value::Number)
    for (c, b) in blocks(t)
        fill!(b, value)
    end
    return t
end
function Base.fill!(t::TensorMap, value::Number)
    fill!(t.data, value)
    return t
end
function LinearAlgebra.adjoint!(tdst::AbstractTensorMap,
                                tsrc::AbstractTensorMap)
    # InnerProductStyle(tdst) === EuclideanInnerProduct() ||
    #     throw_invalid_innerproduct(:adjoint!)
    space(tdst) == adjoint(space(tsrc)) ||
        throw(SpaceMismatch("$(space(tdst)) ≠ adjoint($(space(tsrc)))"))
    for c in blocksectors(tdst)
        adjoint!(StridedView(block(tdst, c)), StridedView(block(tsrc, c)))
    end
    return tdst
end

# Basic vector space methods: recycle VectorInterface implementation
function LinearAlgebra.rmul!(t::AbstractTensorMap, α::Number)
    return iszero(α) ? zerovector!(t) : scale!(t, α)
end
function LinearAlgebra.lmul!(α::Number, t::AbstractTensorMap)
    return iszero(α) ? zerovector!(t) : scale!(t, α)
end

function LinearAlgebra.mul!(t1::AbstractTensorMap, t2::AbstractTensorMap, α::Number)
    return scale!(t1, t2, α)
end
function LinearAlgebra.mul!(t1::AbstractTensorMap, α::Number, t2::AbstractTensorMap)
    return scale!(t1, t2, α)
end

# TODO: remove VectorInterface namespace when we renamed TensorKit.add!
function LinearAlgebra.axpy!(α::Number, t1::AbstractTensorMap, t2::AbstractTensorMap)
    return VectorInterface.add!(t2, t1, α)
end
function LinearAlgebra.axpby!(α::Number, t1::AbstractTensorMap, β::Number,
                              t2::AbstractTensorMap)
    return VectorInterface.add!(t2, t1, α, β)
end

# inner product and norm only valid for spaces with Euclidean inner product
LinearAlgebra.dot(t1::AbstractTensorMap, t2::AbstractTensorMap) = inner(t1, t2)

function LinearAlgebra.norm(t::AbstractTensorMap, p::Real=2)
    return norm(t.data)
    # InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:norm)
    return _norm(blocks(t), p, float(zero(real(scalartype(t)))))
end
function LinearAlgebra.norm(t::AdjointTensorMap, p::Real=2)
    return norm(t.parent)
end
function _norm(blockiter, p::Real, init::Real)
    if p == Inf
        return mapreduce(max, blockiter; init=init) do (c, b)
            return isempty(b) ? init : oftype(init, LinearAlgebra.normInf(b))
        end
    elseif p == 2
        n² = mapreduce(+, blockiter; init=init) do (c, b)
            return isempty(b) ? init : oftype(init, dim(c) * LinearAlgebra.norm2(b)^2)
        end
        return sqrt(n²)
    elseif p == 1
        return mapreduce(+, blockiter; init=init) do (c, b)
            return isempty(b) ? init : oftype(init, dim(c) * sum(abs, b))
        end
    elseif p > 0
        nᵖ = mapreduce(+, blockiter; init=init) do (c, b)
            return isempty(b) ? init : oftype(init, dim(c) * LinearAlgebra.normp(b, p)^p)
        end
        return (nᵖ)^inv(oftype(nᵖ, p))
    else
        msg = "Norm with non-positive p is not defined for `AbstractTensorMap`"
        throw(ArgumentError(msg))
    end
end

_default_rtol(t) = eps(real(float(scalartype(t)))) * min(dim(domain(t)), dim(codomain(t)))

# TensorMap trace
function LinearAlgebra.tr(t::AbstractTensorMap)
    domain(t) == codomain(t) ||
        throw(SpaceMismatch("Trace of a tensor only exist when domain == codomain"))
    s = zero(scalartype(t))
    for (c, b) in blocks(t)
        s += dim(c) * tr(b)
    end
    return s
end

# TensorMap multiplication
function LinearAlgebra.mul!(tC::AbstractTensorMap,
                            tA::AbstractTensorMap,
                            tB::AbstractTensorMap, α=true, β=false)
    compose(space(tA), space(tB)) == space(tC) ||
        throw(SpaceMismatch(lazy"$(space(tC)) ≠ $(space(tA)) * $(space(tB))"))

    iterC = blocks(tC)
    iterA = blocks(tA)
    iterB = blocks(tB)
    nextA = iterate(iterA)
    nextB = iterate(iterB)
    nextC = iterate(iterC)
    while !isnothing(nextC)
        (cC, C), stateC = nextC
        if !isnothing(nextA) && !isnothing(nextB)
            (cA, A), stateA = nextA
            (cB, B), stateB = nextB
            if cA == cC && cB == cC
                mul!(C, A, B, α, β)
                nextA = iterate(iterA, stateA)
                nextB = iterate(iterB, stateB)
                nextC = iterate(iterC, stateC)
            elseif cA < cC
                nextA = iterate(iterA, stateA)
            elseif cB < cC
                nextB = iterate(iterB, stateB)
            else
                if β != one(β)
                    rmul!(C, β)
                end
                nextC = iterate(iterC, stateC)
            end
        else
            if β != one(β)
                rmul!(C, β)
            end
            nextC = iterate(iterC, stateC)
        end
    end
    return tC
end


# concatenate tensors
function catdomain(t1::TT, t2::TT) where {S,N₁,TT<:AbstractTensorMap{<:Any,S,N₁,1}}
    codomain(t1) == codomain(t2) ||
        throw(SpaceMismatch("codomains of tensors to concatenate must match:\n" *
                            "$(codomain(t1)) ≠ $(codomain(t2))"))
    V1, = domain(t1)
    V2, = domain(t2)
    isdual(V1) == isdual(V2) ||
        throw(SpaceMismatch("cannot horizontally concatenate tensors whose domain has non-matching duality"))

    V = V1 ⊕ V2
    T = promote_type(scalartype(t1), scalartype(t2))
    t = similar(t1, T, codomain(t1) ← V)
    for (c, b) in blocks(t)
        b[:, 1:dim(V1, c)] .= block(t1, c)
        b[:, dim(V1, c) .+ (1:dim(V2, c))] .= block(t2, c)
    end
    return t
end
function catcodomain(t1::TT, t2::TT) where {S,N₂,TT<:AbstractTensorMap{<:Any,S,1,N₂}}
    domain(t1) == domain(t2) ||
        throw(SpaceMismatch("domains of tensors to concatenate must match:\n" *
                            "$(domain(t1)) ≠ $(domain(t2))"))
    V1, = codomain(t1)
    V2, = codomain(t2)
    isdual(V1) == isdual(V2) ||
        throw(SpaceMismatch("cannot vertically concatenate tensors whose codomain has non-matching duality"))

    V = V1 ⊕ V2
    T = promote_type(scalartype(t1), scalartype(t2))
    t = similar(t1, T, V ← domain(t1))
    for (c, b) in blocks(t)
        b[1:dim(V1, c), :] .= block(t1, c)
        b[dim(V1, c) .+ (1:dim(V2, c)), :] .= block(t2, c)
    end
    return t
end

# tensor product of tensors
function ⊗(t1::AbstractTensorMap, t2::AbstractTensorMap)
    (S = spacetype(t1)) === spacetype(t2) ||
        throw(SpaceMismatch("spacetype(t1) ≠ spacetype(t2)"))
    cod1, cod2 = codomain(t1), codomain(t2)
    dom1, dom2 = domain(t1), domain(t2)
    cod = cod1 ⊗ cod2
    dom = dom1 ⊗ dom2
    T = promote_type(scalartype(t1), scalartype(t2))
    t = zerovector!(similar(t1, T, cod ← dom))
    for (f1l, f1r) in fusiontrees(t1)
        for (f2l, f2r) in fusiontrees(t2)
            c1 = f1l.coupled # = f1r.coupled
            c2 = f2l.coupled # = f2r.coupled
            for c in c1 ⊗ c2
                for μ in 1:Nsymbol(c1, c2, c)
                    for (fl, coeff1) in merge(f1l, f2l, c, μ)
                        for (fr, coeff2) in merge(f1r, f2r, c, μ)
                            d1 = dim(cod1, f1l.uncoupled)
                            d2 = dim(cod2, f2l.uncoupled)
                            d3 = dim(dom1, f1r.uncoupled)
                            d4 = dim(dom2, f2r.uncoupled)
                            m1 = sreshape(t1[f1l, f1r], (d1, 1, d3, 1))
                            m2 = sreshape(t2[f2l, f2r], (1, d2, 1, d4))
                            m = sreshape(t[fl, fr], (d1, d2, d3, d4))
                            m .+= coeff1 .* conj(coeff2) .* m1 .* m2
                        end
                    end
                end
            end
        end
    end
    return t
end

