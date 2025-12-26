



function foreachblock(f, t::AbstractTensorMap, ts::AbstractTensorMap...; scheduler = nothing)
    tensors = (t, ts...)
    allsectors = union(blocksectors.(tensors)...)
    foreach(allsectors) do c
        return f(c, block.(tensors, Ref(c)))
    end
    return nothing
end
function foreachblock(f, t::AbstractTensorMap; scheduler = nothing)
    foreach(blocks(t)) do (c, b)
        return f(c, (b,))
    end
    return nothing
end


# Algorithm selection
# -------------------
for f in [:svd_compact, :qr_compact, :lq_compact]
    f! = Symbol(f, :!)
    @eval function MAK.default_algorithm(::typeof($f!), ::Type{T}; kwargs...) where {T <: AbstractTensorMap}
        return MAK.default_algorithm($f!, blocktype(T); kwargs...)
    end
    @eval function MAK.copy_input(::typeof($f), t::AbstractTensorMap)
        return copy_oftype(t, factorisation_scalartype($f, t))
    end
end

# _select_truncation(f, ::AbstractTensorMap, trunc::TruncationStrategy) = trunc
# function _select_truncation(::typeof(left_null!), ::AbstractTensorMap, trunc::NamedTuple)
#     return MAK.null_truncation_strategy(; trunc...)
# end

AbstractAlgorithm = MAK.AbstractAlgorithm
macro check_space(x, V)
    return esc(:($MatrixAlgebraKit.@check_size($x, $V, $space)))
end
macro check_scalar(x, y, op = :identity, eltype = :scalartype)
    return esc(:($MatrixAlgebraKit.@check_scalar($x, $y, $op, $eltype)))
end

# Generic Implementations
# -----------------------
for f! in (:qr_compact!, :lq_compact!, :svd_compact!)
    @eval function MAK.$f!(t::AbstractTensorMap, F, alg::AbstractAlgorithm)
        MAK.check_input($f!, t, F, alg)

        foreachblock(t, F...) do _, bs
            factors = Base.tail(bs)
            factors′ = $f!(first(bs), factors, alg)
            # deal with the case where the output is not in-place
            for (f′, f) in zip(factors′, factors)
                f′ === f || copy!(f, f′)
            end
            return nothing
        end

        return F
    end
end


function MAK.check_input(::typeof(svd_compact!), t::AbstractTensorMap, USVᴴ, ::AbstractAlgorithm)
    U, S, Vᴴ = USVᴴ

    # type checks
    @assert U isa AbstractTensorMap
    @assert S isa DiagonalTensorMap
    @assert Vᴴ isa AbstractTensorMap

    # scalartype checks
    @check_scalar U t
    @check_scalar S t real
    @check_scalar Vᴴ t

    # space checks
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(U, codomain(t) ← V_cod)
    @check_space(S, V_cod ← V_dom)
    @check_space(Vᴴ, V_dom ← domain(t))

    return nothing
end

function MAK.initialize_output(::typeof(svd_compact!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    U = similar(t, codomain(t) ← V_cod)
    rt = real(scalartype(t))
    mem = storagetype(t).parameters[3]
    S = DiagonalTensorMap{rt, typeof(V_cod), CuArray{rt,1,mem}}(undef, V_cod)
    Vᴴ = similar(t, V_dom ← domain(t))
    return U, S, Vᴴ
end



# QR decomposition
# ----------------
function MAK.check_input(::typeof(qr_compact!), t::AbstractTensorMap, QR, ::AbstractAlgorithm)
    Q, R = QR

    # type checks
    @assert Q isa AbstractTensorMap
    @assert R isa AbstractTensorMap

    # scalartype checks
    @check_scalar Q t
    @check_scalar R t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(Q, codomain(t) ← V_Q)
    @check_space(R, V_Q ← domain(t))

    return nothing
end

function MAK.initialize_output(::typeof(qr_compact!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    Q = similar(t, codomain(t) ← V_Q)
    R = similar(t, V_Q ← domain(t))
    return Q, R
end

# LQ decomposition
# ----------------
function MAK.check_input(::typeof(lq_compact!), t::AbstractTensorMap, LQ, ::AbstractAlgorithm)
    L, Q = LQ

    # type checks
    @assert L isa AbstractTensorMap
    @assert Q isa AbstractTensorMap

    # scalartype checks
    @check_scalar L t
    @check_scalar Q t

    # space checks
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    @check_space(L, codomain(t) ← V_Q)
    @check_space(Q, V_Q ← domain(t))

    return nothing
end

function MAK.initialize_output(::typeof(lq_compact!), t::AbstractTensorMap, ::AbstractAlgorithm)
    V_Q = infimum(fuse(codomain(t)), fuse(domain(t)))
    L = similar(t, codomain(t) ← V_Q)
    Q = similar(t, V_Q ← domain(t))
    return L, Q
end


