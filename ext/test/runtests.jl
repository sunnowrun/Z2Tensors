push!(LOAD_PATH, dirname(dirname(dirname(Base.@__DIR__))) * "/Z2TensorKit/src")


using Test
using TestExtras
using Random
using Combinatorics
using TensorOperations
using Base.Iterators: take, product
using LinearAlgebra: LinearAlgebra

using Z2TensorKit
using Z2TensorKit: fusiontensor#, pentagon_equation, hexagon_equation # ProductSector, 
const TK = Z2TensorKit

Random.seed!(1234)

smallset(::Type{I}) where {I<:Sector} = take(values(I), 5)
function randsector(::Type{I}) where {I<:Sector}
    s = collect(smallset(I))
    a = rand(s)
    while a == one(a) # don't use trivial label
        a = rand(s)
    end
    return a
end
function hasfusiontensor(I::Type{<:Sector})
    try
        fusiontensor(one(I), one(I), one(I))
        return true
    catch e
        if e isa MethodError
            return false
        else
            rethrow(e)
        end
    end
end

sectorlist = (Z2Irrep,)

# spaces
VZ2 = (Z2Space(0 => 1, 1 => 1),
        Z2Space(0 => 1, 1 => 2)',
        Z2Space(0 => 3, 1 => 2)',
        Z2Space(0 => 2, 1 => 3),
        Z2Space(0 => 2, 1 => 5))

Ti = time()



include("cutensors.jl")

