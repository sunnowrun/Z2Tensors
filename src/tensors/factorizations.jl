# Tensor factorization
#----------------------
function factorisation_scalartype(t::AbstractTensorMap)
    T = scalartype(t)
    return promote_type(Float32, typeof(zero(T) / sqrt(abs2(one(T)))))
end
factorisation_scalartype(f, t) = factorisation_scalartype(t)

function permutedcopy_oftype(t::AbstractTensorMap, T::Type{<:Number}, p::Index2Tuple)
    return permute!(similar(t, T, permute(space(t), p)), t, p)
end
function copy_oftype(t::AbstractTensorMap, T::Type{<:Number})
    return copy!(similar(t, T), t)
end

function tsvd(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    tcopy = permutedcopy_oftype(t, factorisation_scalartype(tsvd, t), p)
    return tsvd!(tcopy; kwargs...)
end

function LinearAlgebra.svdvals(t::AbstractTensorMap)
    tcopy = copy_oftype(t, factorisation_scalartype(tsvd, t))
    return LinearAlgebra.svdvals!(tcopy)
end

function leftorth(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    tcopy = permutedcopy_oftype(t, factorisation_scalartype(leftorth, t), p)
    return leftorth!(tcopy; kwargs...)
end

function rightorth(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
    tcopy = permutedcopy_oftype(t, factorisation_scalartype(rightorth, t), p)
    return rightorth!(tcopy; kwargs...)
end

function tsvd(t::AbstractTensorMap; kwargs...)
    tcopy = copy_oftype(t, float(scalartype(t)))
    return tsvd!(tcopy; kwargs...)
end
function leftorth(t::AbstractTensorMap; alg::OFA=QRpos(), kwargs...)
    tcopy = copy_oftype(t, float(scalartype(t)))
    return leftorth!(tcopy; alg=alg, kwargs...)
end
function rightorth(t::AbstractTensorMap; alg::OFA=LQpos(), kwargs...)
    tcopy = copy_oftype(t, float(scalartype(t)))
    return rightorth!(tcopy; alg=alg, kwargs...)
end

# Orthogonal factorizations (mutation for recycling memory):
# only possible if scalar type is floating point
# only correct if Euclidean inner product
#------------------------------------------------------------------------------------------
const RealOrComplexFloat = Union{AbstractFloat,Complex{<:AbstractFloat}}

function leftorth!(t::TensorMap{<:RealOrComplexFloat};
                   alg::Union{QR,QRpos,SVD,SDD,Polar}=QRpos(),
                   atol::Real=zero(float(real(scalartype(t)))),
                   rtol::Real=(alg ∉ (SVD(), SDD())) ? zero(float(real(scalartype(t)))) :
                              eps(real(float(one(scalartype(t))))) * iszero(atol))
    # InnerProductStyle(t) === EuclideanInnerProduct() ||
    #     throw_invalid_innerproduct(:leftorth!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    dims = SectorDict{I,Int}()

    # compute QR factorization for each block
    if !isempty(blocks(t))
        generator = Base.Iterators.map(blocks(t)) do (c, b)
            # Qc, Rc = MatrixAlgebra.leftorth!(b, alg, atol)
            Qc, Rc = leftorth!(b, alg, atol)
            dims[c] = size(Qc, 2)
            return c => (Qc, Rc)
        end
        QRdata = SectorDict(generator)
    end

    # construct new space
    S = spacetype(t)
    V = S(dims)
    if alg isa Polar
        @assert V ≅ domain(t)
        W = domain(t)
    elseif length(domain(t)) == 1 && domain(t) ≅ V
        W = domain(t)
    elseif length(codomain(t)) == 1 && codomain(t) ≅ V
        W = codomain(t)
    else
        W = ProductSpace(V)
    end

    # construct output tensors
    T = float(scalartype(t))
    Q = similar(t, T, codomain(t) ← W)
    R = similar(t, T, W ← domain(t))
    if !isempty(blocks(t))
        for (c, (Qc, Rc)) in QRdata
            copy!(block(Q, c), Qc)
            copy!(block(R, c), Rc)
        end
    end
    return Q, R
end

function rightorth!(t::TensorMap{<:RealOrComplexFloat};
                    alg::Union{LQ,LQpos,SVD,SDD,Polar}=LQpos(),
                    atol::Real=zero(float(real(scalartype(t)))),
                    rtol::Real=(alg ∉ (SVD(), SDD())) ? zero(float(real(scalartype(t)))) :
                               eps(real(float(one(scalartype(t))))) * iszero(atol))
    # InnerProductStyle(t) === EuclideanInnerProduct() ||
    #     throw_invalid_innerproduct(:rightorth!)
    if !iszero(rtol)
        atol = max(atol, rtol * norm(t))
    end
    I = sectortype(t)
    dims = SectorDict{I,Int}()

    # compute LQ factorization for each block
    if !isempty(blocks(t))
        generator = Base.Iterators.map(blocks(t)) do (c, b)
            # Lc, Qc = MatrixAlgebra.rightorth!(b, alg, atol)
            Lc, Qc = rightorth!(b, alg, atol)
            dims[c] = size(Qc, 1)
            return c => (Lc, Qc)
        end
        LQdata = SectorDict(generator)
    end

    # construct new space
    S = spacetype(t)
    V = S(dims)
    if alg isa Polar
        @assert V ≅ codomain(t)
        W = codomain(t)
    elseif length(codomain(t)) == 1 && codomain(t) ≅ V
        W = codomain(t)
    elseif length(domain(t)) == 1 && domain(t) ≅ V
        W = domain(t)
    else
        W = ProductSpace(V)
    end

    # construct output tensors
    T = float(scalartype(t))
    L = similar(t, T, codomain(t) ← W)
    Q = similar(t, T, W ← domain(t))
    if !isempty(blocks(t))
        for (c, (Lc, Qc)) in LQdata
            copy!(block(L, c), Lc)
            copy!(block(Q, c), Qc)
        end
    end
    return L, Q
end

function leftorth!(t::AdjointTensorMap; alg::OFA=QRpos())
    # InnerProductStyle(t) === EuclideanInnerProduct() ||
    #     throw_invalid_innerproduct(:leftorth!)
    return map(adjoint, reverse(rightorth!(adjoint(t); alg=alg')))
end

function rightorth!(t::AdjointTensorMap; alg::OFA=LQpos())
    # InnerProductStyle(t) === EuclideanInnerProduct() ||
    #     throw_invalid_innerproduct(:rightorth!)
    return map(adjoint, reverse(leftorth!(adjoint(t); alg=alg')))
end

#------------------------------#
# Singular value decomposition #
#------------------------------#
function LinearAlgebra.svdvals!(t::TensorMap{<:RealOrComplexFloat})
    return SectorDict(c => LinearAlgebra.svdvals!(b) for (c, b) in blocks(t))
end
LinearAlgebra.svdvals!(t::AdjointTensorMap) = svdvals!(adjoint(t))

function tsvd!(t::TensorMap{<:RealOrComplexFloat};
               trunc=NoTruncation(), p::Real=2, alg=SDD())
    return _tsvd!(t, alg, trunc, p)
end
function tsvd!(t::AdjointTensorMap; trunc=NoTruncation(), p::Real=2, alg=SDD())
    u, s, vt, err = tsvd!(adjoint(t); trunc=trunc, p=p, alg=alg)
    return adjoint(vt), adjoint(s), adjoint(u), err
end

# implementation dispatches on algorithm
function _tsvd!(t::TensorMap{<:RealOrComplexFloat}, alg::Union{SVD,SDD},
                trunc::TruncationScheme, p::Real=2)
    # early return
    if isempty(blocksectors(t))
        truncerr = zero(real(scalartype(t)))
        return _empty_svdtensors(t)..., truncerr
    end

    # compute SVD factorization for each block
    S = spacetype(t)
    SVDdata, dims = _compute_svddata!(t, alg)
    Σdata = SectorDict(c => Σ for (c, (U, Σ, V)) in SVDdata)
    truncdim = _compute_truncdim(Σdata, trunc, p)
    truncerr = _compute_truncerr(Σdata, truncdim, p)

    # construct output tensors
    U, Σ, V⁺ = _create_svdtensors(t, SVDdata, truncdim)
    return U, Σ, V⁺, truncerr
end

# helper functions
function _compute_svddata!(t::TensorMap, alg::Union{SVD,SDD})
    # InnerProductStyle(t) === EuclideanInnerProduct() || throw_invalid_innerproduct(:tsvd!)
    I = sectortype(t)
    dims = SectorDict{I,Int}()
    generator = Base.Iterators.map(blocks(t)) do (c, b)
        # U, Σ, V = MatrixAlgebra.svd!(b, alg)
        U, Σ, V = _svd!(b, alg)
        dims[c] = length(Σ)
        return c => (U, Σ, V)
    end
    SVDdata = SectorDict(generator)
    return SVDdata, dims
end

function _create_svdtensors(t::TensorMap{<:RealOrComplexFloat}, SVDdata, dims)
    T = scalartype(t)
    S = spacetype(t)
    W = S(dims)

    Tr = real(T)
    A = similarstoragetype(t, Tr)
    Σ = DiagonalTensorMap{Tr,S,A}(undef, W)

    U = similar(t, codomain(t) ← W)
    V⁺ = similar(t, W ← domain(t))
    for (c, (Uc, Σc, V⁺c)) in SVDdata
        r = Base.OneTo(dims[c])
        copy!(block(U, c), view(Uc, :, r))
        copy!(block(Σ, c), Diagonal(view(Σc, r)))
        copy!(block(V⁺, c), view(V⁺c, r, :))
    end
    return U, Σ, V⁺
end

function _empty_svdtensors(t::TensorMap{<:RealOrComplexFloat})
    T = scalartype(t)
    S = spacetype(t)
    I = sectortype(t)
    dims = SectorDict{I,Int}()
    W = S(dims)

    Tr = real(T)
    A = similarstoragetype(t, Tr)
    Σ = DiagonalTensorMap{Tr,S,A}(undef, W)

    U = similar(t, codomain(t) ← W)
    V⁺ = similar(t, W ← domain(t))
    return U, Σ, V⁺
end



tsvd(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...) = tsvd(t, (p₁, p₂); kwargs...)
leftorth(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...) = leftorth(t, (p₁, p₂); kwargs...)
rightorth(t::AbstractTensorMap, p₁::IndexTuple, p₂::IndexTuple; kwargs...) = rightorth(t, (p₁, p₂); kwargs...)
