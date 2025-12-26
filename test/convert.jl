
using TupleTools, Strided
using Z2TensorKit: couple
const TO = TensorOperations


_kron(A, B, C, D...) = _kron(_kron(A, B), C, D...)
function _kron(A, B)
    sA = size(A)
    sB = size(B)
    s = map(*, sA, sB)
    C = similar(A, promote_type(eltype(A), eltype(B)), s)
    for IA in eachindex(IndexCartesian(), A)
        for IB in eachindex(IndexCartesian(), B)
            I = CartesianIndex(IB.I .+ (IA.I .- 1) .* sB)
            C[I] = A[IA] * B[IB]
        end
    end
    return C
end


# converting to actual array
function Base.convert(A::Type{<:AbstractArray}, f::FusionTree{I,0}) where {I}
    X = convert(A, fusiontensor(one(I), one(I), one(I)))[1, 1, :]
    return X
end
function Base.convert(A::Type{<:AbstractArray}, f::FusionTree{I,1}) where {I}
    c = f.coupled
    # if f.isdual[1]
    #     sqrtdc = sqrtdim(c)
    #     Zcbartranspose = sqrtdc * convert(A, fusiontensor(conj(c), c, one(c)))[:, :, 1, 1]
    #     X = conj!(Zcbartranspose) # we want Zcbar^†
    # else
        X = convert(A, fusiontensor(c, one(c), c))[:, 1, :, 1, 1]
    # end
    return X
end

function Base.convert(A::Type{<:AbstractArray}, f::FusionTree{I,2}) where {I}
    a, b = f.uncoupled
    c = f.coupled
    μ = 1
    C = convert(A, fusiontensor(a, b, c))[:, :, :, μ]
    X = C
    # if isduala
    #     Za = convert(A, FusionTree((a,), a, (isduala,)))
    #     @tensor X[a′, b, c] := Za[a′, a] * X[a, b, c]
    # end
    # if isdualb
    #     Zb = convert(A, FusionTree((b,), b, (isdualb,)))
    #     @tensor X[a, b′, c] := Zb[b′, b] * X[a, b, c]
    # end
    return X
end

function Base.convert(A::Type{<:AbstractArray}, f::FusionTree{I,N}) where {I,N}
    c12 = couple((f.uncoupled[1], f.uncoupled[2]))
    tailout = (c12, TupleTools.tail2(f.uncoupled)...)
    # isdualout = (false, TupleTools.tail2(f.isdual)...)
    ftail = FusionTree(tailout, f.coupled)
    Ctail = convert(A, ftail)

    f₁ = FusionTree((f.uncoupled[1], f.uncoupled[2]), c12)
    C1 = convert(A, f₁)
    dtail = size(Ctail)
    d1 = size(C1)
    X = similar(C1, (d1[1], d1[2], Base.tail(dtail)...))
    trivialtuple = ntuple(identity, Val(N))
    return TO.tensorcontract!(X,
                              C1, ((1, 2), (3,)), false,
                              Ctail, ((1,), Base.tail(trivialtuple)), false,
                              ((trivialtuple..., N + 1), ()))
end

# TODO: is this piracy?
function Base.convert(A::Type{<:AbstractArray},
                      (f₁, f₂)::Tuple{FusionTree{I},FusionTree{I}}) where {I}
    F₁ = convert(A, f₁)
    F₂ = convert(A, f₂)
    sz1 = size(F₁)
    sz2 = size(F₂)
    d1 = TupleTools.front(sz1)
    d2 = TupleTools.front(sz2)

    return reshape(reshape(F₁, TupleTools.prod(d1), sz1[end]) *
                   reshape(F₂, TupleTools.prod(d2), sz2[end])', (d1..., d2...))
end



# axes
Base.axes(V::GradedSpace) = Base.OneTo(dim(V))
function Base.axes(V::GradedSpace{I}, c::I) where {I<:Sector}
    offset = 0
    for c′ in sectors(V)
        c′ == c && break
        offset += dim(c′) * dim(V, c′)
    end
    return (offset + 1):(offset + dim(c) * dim(V, c))
end
Base.axes(P::ProductSpace) = map(axes, P.spaces)
Base.axes(P::ProductSpace, n::Int) = axes(P.spaces[n])
function Base.axes(P::ProductSpace{<:ElementarySpace,N},
                   sectors::NTuple{N,<:Sector}) where {N}
    return map(axes, P.spaces, sectors)
end


# Conversion to Array:
#----------------------
# probably not optimized for speed, only for checking purposes
function Base.convert(::Type{Array}, t::AbstractTensorMap)
    I = sectortype(t)
    # if I === Trivial
    #     convert(Array, t[])
    # else
        cod = codomain(t)
        dom = domain(t)
        # T = sectorscalartype(I) <: Complex ? complex(scalartype(t)) :
        #     sectorscalartype(I) <: Integer ? scalartype(t) : float(scalartype(t))
        T = scalartype(t)
        A = zeros(T, dims(cod)..., dims(dom)...)
        for (f₁, f₂) in fusiontrees(t)
            F = convert(Array, (f₁, f₂))
            Aslice = StridedView(A)[axes(cod, f₁.uncoupled)..., axes(dom, f₂.uncoupled)...]
            add!(Aslice, StridedView(_kron(convert(Array, t[f₁, f₂]), F)))
        end
        return A
    # end
end






