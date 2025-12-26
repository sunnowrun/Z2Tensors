# Simple reference to getting and setting BLAS threads
#------------------------------------------------------
set_num_blas_threads(n::Integer) = LinearAlgebra.BLAS.set_num_threads(n)
get_num_blas_threads() = LinearAlgebra.BLAS.get_num_threads()

# Factorization algorithms
#--------------------------
abstract type FactorizationAlgorithm end
abstract type OrthogonalFactorizationAlgorithm <: FactorizationAlgorithm end

struct QRpos <: OrthogonalFactorizationAlgorithm
end
struct QR <: OrthogonalFactorizationAlgorithm
end
struct LQ <: OrthogonalFactorizationAlgorithm
end
struct LQpos <: OrthogonalFactorizationAlgorithm
end
struct SDD <: OrthogonalFactorizationAlgorithm # lapack's default divide and conquer algorithm
end
struct SVD <: OrthogonalFactorizationAlgorithm
end
struct Polar <: OrthogonalFactorizationAlgorithm
end

Base.adjoint(::QRpos) = LQpos()
Base.adjoint(::QR) = LQ()
Base.adjoint(::LQpos) = QRpos()
Base.adjoint(::LQ) = QR()

Base.adjoint(alg::Union{SVD,SDD,Polar}) = alg

const OFA = OrthogonalFactorizationAlgorithm
const SVDAlg = Union{SVD,SDD}



# TODO: define for CuMatrix if we support this
function one!(A::StridedMatrix)
    length(A) > 0 || return A
    copyto!(A, LinearAlgebra.I)
    return A
end

safesign(s::Real) = ifelse(s < zero(s), -one(s), +one(s))
safesign(s::Complex) = ifelse(iszero(s), one(s), s / abs(s))

function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{QR,QRpos}, atol::Real)
    iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
    m, n = size(A)
    k = min(m, n)
    A, T = LAPACK.geqrt!(A, min(minimum(size(A)), 36))
    Q = similar(A, m, k)
    for j in 1:k
        for i in 1:m
            Q[i, j] = i == j
        end
    end
    Q = LAPACK.gemqrt!('L', 'N', A, T, Q)
    R = triu!(A[1:k, :])

    if isa(alg, QRpos)
        @inbounds for j in 1:k
            s = safesign(R[j, j])
            @simd for i in 1:m
                Q[i, j] *= s
            end
        end
        @inbounds for j in size(R, 2):-1:1
            for i in 1:min(k, j)
                R[i, j] = R[i, j] * conj(safesign(R[i, i]))
            end
        end
    end
    return Q, R
end

function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD,Polar}, atol::Real)
    U, S, V = alg isa SVD ? LAPACK.gesvd!('S', 'S', A) : LAPACK.gesdd!('S', A)
    if isa(alg, Union{SVD,SDD})
        n = count(s -> s .> atol, S)
        if n != length(S)
            return U[:, 1:n], lmul!(Diagonal(S[1:n]), V[1:n, :])
        else
            return U, lmul!(Diagonal(S), V)
        end
    else
        iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
        # TODO: check Lapack to see if we can recycle memory of A
        Q = mul!(A, U, V)
        Sq = map!(sqrt, S, S)
        SqV = lmul!(Diagonal(Sq), V)
        R = SqV' * SqV
        return Q, R
    end
end

function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{LQ,LQpos},
                    atol::Real)
    iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
    # TODO: geqrfp seems a bit slower than geqrt in the intermediate region around
    # matrix size 100, which is the interesting region. => Investigate and fix
    m, n = size(A)
    k = min(m, n)
    At = transpose!(similar(A, n, m), A)

    # if isa(alg, RQ) || isa(alg, RQpos)
    #     @assert m <= n

    #     mhalf = div(m, 2)
    #     # swap columns in At
    #     @inbounds for j in 1:mhalf, i in 1:n
    #         At[i, j], At[i, m + 1 - j] = At[i, m + 1 - j], At[i, j]
    #     end
    #     Qt, Rt = leftorth!(At, isa(alg, RQ) ? QR() : QRpos(), atol)

    #     @inbounds for j in 1:mhalf, i in 1:n
    #         Qt[i, j], Qt[i, m + 1 - j] = Qt[i, m + 1 - j], Qt[i, j]
    #     end
    #     @inbounds for j in 1:mhalf, i in 1:m
    #         Rt[i, j], Rt[m + 1 - i, m + 1 - j] = Rt[m + 1 - i, m + 1 - j], Rt[i, j]
    #     end
    #     if isodd(m)
    #         j = mhalf + 1
    #         @inbounds for i in 1:mhalf
    #             Rt[i, j], Rt[m + 1 - i, j] = Rt[m + 1 - i, j], Rt[i, j]
    #         end
    #     end
    #     Q = transpose!(A, Qt)
    #     R = transpose!(similar(A, (m, m)), Rt) # TODO: efficient in place
    #     return R, Q
    # else
        Qt, Lt = leftorth!(At, alg', atol)
        if m > n
            L = transpose!(A, Lt)
            Q = transpose!(similar(A, (n, n)), Qt) # TODO: efficient in place
        else
            Q = transpose!(A, Qt)
            L = transpose!(similar(A, (m, m)), Lt) # TODO: efficient in place
        end
        return L, Q
    # end
end

function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD,Polar}, atol::Real)
    U, S, V = alg isa SVD ? LAPACK.gesvd!('S', 'S', A) : LAPACK.gesdd!('S', A)
    if isa(alg, Union{SVD,SDD})
        n = count(s -> s .> atol, S)
        if n != length(S)
            return rmul!(U[:, 1:n], Diagonal(S[1:n])), V[1:n, :]
        else
            return rmul!(U, Diagonal(S)), V
        end
    else
        iszero(atol) || throw(ArgumentError("nonzero atol not supported by $alg"))
        Q = mul!(A, U, V)
        Sq = map!(sqrt, S, S)
        USq = rmul!(U, Diagonal(Sq))
        L = USq * USq'
        return L, Q
    end
end

function _svd!(A::StridedMatrix{T}, alg::Union{SVD,SDD}) where {T<:BlasFloat}
    # fix another type instability in LAPACK wrappers
    TT = Tuple{Matrix{T},Vector{real(T)},Matrix{T}}
    U, S, V = alg isa SVD ? LAPACK.gesvd!('S', 'S', A)::TT : LAPACK.gesdd!('S', A)::TT
    return U, S, V
end




# using MatrixAlgebraKit: MatrixAlgebraKit, left_orth!, right_orth!, LAPACK_QRIteration, LAPACK_DivideAndConquer, trunctol, svd_compact!

# function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{QR,QRpos}, atol::Real)
#     left_orth!(A, alg=:qr, positive = alg isa QRpos)
# end
# function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD}, atol::Real)
#     alg′ = MatrixAlgebraKit.TruncatedAlgorithm(alg isa SVD ? LAPACK_QRIteration() : LAPACK_DivideAndConquer(), trunctol(atol=atol))
#     left_orth!(A, alg=MatrixAlgebraKit.LeftOrthAlgorithm{:svd}(alg′))
# end
# function leftorth!(A::StridedMatrix{<:BlasFloat}, alg::Polar, atol::Real)
#     left_orth!(A, alg=:polar)
# end

# function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{LQ,LQpos}, atol::Real)
#     right_orth!(A, alg=:lq, positive = alg isa QRpos)
# end
# function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Union{SVD,SDD}, atol::Real)
#     alg′ = MatrixAlgebraKit.TruncatedAlgorithm(alg isa SVD ? LAPACK_QRIteration() : LAPACK_DivideAndConquer(), trunctol(atol=atol))
#     right_orth!(A, alg=MatrixAlgebraKit.RightOrthAlgorithm{:svd}(alg′))
# end
# function rightorth!(A::StridedMatrix{<:BlasFloat}, alg::Polar, atol::Real)
#     right_orth!(A, alg=:polar)
# end

# function _svd!(A::StridedMatrix{T}, alg::Union{SVD,SDD}) where {T<:BlasFloat}
#     alg′ = (alg isa SVD) ? LAPACK_QRIteration() : LAPACK_DivideAndConquer()
#     U, S, V = svd_compact!(A, alg=alg′)
#     return U, S.diag, V
# end
