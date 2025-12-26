
using CUDA, MatrixAlgebraKit
const MAK = MatrixAlgebraKit
using MatrixAlgebraKit: TruncatedAlgorithm, TruncationByOrder, Diagonal, @check_size, @check_scalar, 
        check_input, _gpu_Xgesvdr!, truncate, norm, diagview, default_fixgauge, gaugefix!


function MAK.svd_trunc!(A::AbstractMatrix, USVᴴ, alg::TruncatedAlgorithm{<:CUSOLVER_Randomized, <:TruncationByOrder})
    check_input(svd_trunc!, A, USVᴴ, alg.alg)
    U, S, Vᴴ = USVᴴ
    _gpu_Xgesvdr!(A, S.diag, U, Vᴴ; k=alg.trunc.howmany, alg.alg.kwargs...)

    # TODO: make sure that truncation is based on maxrank, otherwise this might be wrong
    (Utr, Str, Vᴴtr), _ = truncate(svd_trunc!, (U, S, Vᴴ), alg.trunc)

    # normal `truncation_error!` does not work here since `S` is not the full singular value spectrum
    ϵ = sqrt(norm(A)^2 - norm(diagview(Str))^2) # is there a more accurate way to do this?

    do_gauge_fix = get(alg.alg.kwargs, :fixgauge, default_fixgauge())::Bool
    do_gauge_fix && gaugefix!(svd_trunc!, Utr, Vᴴtr)

    return Utr, Str, Vᴴtr, ϵ
end


# A = CuArray(randn(100,90));
# trunc = truncrank(20)
# U1, S1, Vd1, err1 = svd_trunc(A; alg = TruncatedAlgorithm(CUSOLVER_Randomized(), trunc))
# A1 = U1 * S1 * Vd1
# U2, S2, Vd2, err2 = svd_trunc(A, trunc=trunc)
# A2 = U2 * S2 * Vd2


