

type_repr(T::Type) = repr(T)

function Base.show(io::IO, c::ZNIrrep)
    I = typeof(c)
    print_type = get(io, :typeinfo, nothing) !== I
    print_type && print(io, type_repr(I), '(')
    print(io, charge(c))
    print_type && print(io, ')')
    return nothing
end

type_repr(::Type{<:GradedSpace{Z2Irrep}}) = "Z2Space"


function Base.show(io::IO, V::GradedSpace{I}) where {I<:Sector}
    print(io, type_repr(typeof(V)), "(")
    seperator = ""
    comma = ", "
    io2 = IOContext(io, :typeinfo => I)
    for c in sectors(V)
        if isdual(V)
            print(io2, seperator, dual(c), "=>", dim(V, c))
        else
            print(io2, seperator, c, "=>", dim(V, c))
        end
        seperator = comma
    end
    print(io, ")")
    V.dual && print(io, "'")
    return nothing
end
function Base.show(io::IO, P::ProductSpace{S}) where {S<:ElementarySpace}
    spaces = P.spaces
    if length(spaces) == 0
        print(io, "ProductSpace{", S, ", 0}")
    end
    if length(spaces) == 1
        print(io, "ProductSpace")
    end
    print(io, "(")
    for i in 1:length(spaces)
        i == 1 || print(io, " ⊗ ")
        show(io, spaces[i])
    end
    return print(io, ")")
end

function Base.show(io::IO, W::HomSpace)
    if length(W.codomain) == 1
        print(io, W.codomain[1])
    else
        print(io, W.codomain)
    end
    print(io, " ← ")
    if length(W.domain) == 1
        print(io, W.domain[1])
    else
        print(io, W.domain)
    end
end




function Base.show(io::IO, t::FusionTree{I}) where {I<:Sector}
    return print(IOContext(io, :typeinfo => I), "FusionTree{", type_repr(I), "}(",
                    t.uncoupled, ", ", t.coupled, ")")
end



function Base.show(io::IO, t::AdjointTensorMap)
    if get(io, :compact, false)
        print(io, "AdjointTensorMap(", codomain(t), " ← ", domain(t), ")")
        return
    end
    println(io, "AdjointTensorMap(", codomain(t), " ← ", domain(t), "):")
    for (f₁, f₂) in fusiontrees(t)
        println(io, "* Data for sector ", f₁.uncoupled, " ← ", f₂.uncoupled, ":")
        Base.print_array(io, t[f₁, f₂])
        println(io)
    end
end

function Base.show(io::IO, t::DiagonalTensorMap)
    summary(io, t)
    get(io, :compact, false) && return nothing
    println(io, ":")

    for (c, b) in blocks(t)
        println(io, "* Data for sector ", c, ":")
        Base.print_array(io, b)
        println(io)
    end
    return nothing
end

function Base.show(io::IO, t::TensorMap)
    if get(io, :compact, false)
        print(io, "TensorMap(", space(t), ")")
        return
    end
    println(io, "TensorMap(", space(t), "):")
    for (f₁, f₂) in fusiontrees(t)
        println(io, "* Data for sector ", f₁.uncoupled, " ← ", f₂.uncoupled, ":")
        Base.print_array(io, t[f₁, f₂])
        println(io)
    end
end



