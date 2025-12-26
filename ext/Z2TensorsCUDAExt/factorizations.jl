

function TK.leftorth!(t::CuTensorMap{<:RealOrComplexFloat};
                   alg::Union{QR,QRpos,SVD,SDD}=QR(),
                   atol::Real=zero(float(real(scalartype(t)))),
                   rtol::Real=zero(float(real(scalartype(t)))))
    if isa(alg, QR)
        return qr_compact!(t, CUSOLVER_HouseholderQR())
    elseif isa(alg, QRpos)
        return qr_compact!(t, CUSOLVER_HouseholderQR(positive=true))
    else
        U, S, Vd = svd_compact!(t, CUSOLVER_QRIteration())
        return U, S*Vd
    end
end

function TK.rightorth!(t::CuTensorMap{<:RealOrComplexFloat};
                    alg::Union{LQ,LQpos,SVD,SDD}=LQ(),
                    atol::Real=zero(float(real(scalartype(t)))),
                    rtol::Real=zero(float(real(scalartype(t)))))
    if isa(alg, LQ)
        return lq_compact!(t, LQViaTransposedQR(CUSOLVER_HouseholderQR()))
    elseif isa(alg, LQpos)
        return lq_compact!(t, LQViaTransposedQR(CUSOLVER_HouseholderQR(positive=true)))
    else
        U, S, Vd = svd_compact!(t, CUSOLVER_QRIteration())
        return U*S, Vd
    end
end

diagview(t::AbstractTensorMap) = SectorDict(c => diagview(b) for (c, b) in blocks(t))
diagview(d::LinearAlgebra.Diagonal) = d.diag
function TK.tsvd!(t::CuTensorMap{<:RealOrComplexFloat};
               trunc=NoTruncation(), p::Real=2, alg=SDD())
    # early return
    if isempty(blocksectors(t))
        truncerr = zero(real(scalartype(t)))
        return _empty_svdtensors(t)..., truncerr
    end

    U, S, Vᴴ = svd_compact!(t, CUSOLVER_QRIteration())

    Σdata = diagview(fromcu(S))
    truncdim = _compute_truncdim(Σdata, trunc, p)
    truncerr = _compute_truncerr(Σdata, truncdim, p)
    
    V_truncated = spacetype(t)(truncdim)
    Ũ = similar(U, codomain(U) ← V_truncated)
    truncate_domain!(Ũ, U, truncdim)

    S̃ = typeof(S)(undef, V_truncated)
    # S̃ = DiagonalTensorMap{scalartype(S)}(undef, V_truncated)
    truncate_diagonal!(S̃, S, truncdim)
    Ṽᴴ = similar(Vᴴ, V_truncated ← domain(Vᴴ))
    truncate_codomain!(Ṽᴴ, Vᴴ, truncdim)

    return Ũ, S̃, Ṽᴴ, truncerr    
end

function truncate_domain!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, inds)
    for (c, b) in blocks(tdst)
        I = get(inds, c, nothing)
        @assert !isnothing(I)
        copy!(b, @view(block(tsrc, c)[:, Base.OneTo(I)]))
    end
    return tdst
end
function truncate_codomain!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, inds)
    for (c, b) in blocks(tdst)
        I = get(inds, c, nothing)
        @assert !isnothing(I)
        copy!(b, @view(block(tsrc, c)[Base.OneTo(I), :]))
    end
    return tdst
end
function truncate_diagonal!(Ddst::DiagonalTensorMap, Dsrc::DiagonalTensorMap, inds)
    for (c, b) in blocks(Ddst)
        I = get(inds, c, nothing)
        @assert !isnothing(I)
        copy!(diagview(b), view(diagview(block(Dsrc, c)), Base.OneTo(I)))
    end
    return Ddst
end

