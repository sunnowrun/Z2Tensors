

abstract type Sector end

struct ZNIrrep{N} <: Sector#<: AbstractIrrep{ℤ{N}}
    n::UInt8
    function ZNIrrep{N}(n::Integer) where {N}
        # N ≤ SMALL_ZN_CUTOFF || throw(DomainError(N, "N exceeds the maximal value, use `LargeZNIrrep` instead"))
        return new{N}(UInt8(mod(n, N)))
    end
end

const Z2Irrep = ZNIrrep{2}



# Nsymbol(a::Z2Irrep, b::Z2Irrep, c::Z2Irrep) = c == first(a ⊗ b)
Nsymbol(a::Z2Irrep, b::Z2Irrep, c::Z2Irrep) = c == a ⊗ b
function Fsymbol(a::Z2Irrep, b::Z2Irrep, c::Z2Irrep, d::Z2Irrep, e::Z2Irrep, f::Z2Irrep)
    return Int(Nsymbol(a, b, e) * Nsymbol(e, c, d) * Nsymbol(b, c, f) * Nsymbol(a, f, d))
end
frobenius_schur_phase(a::Z2Irrep) = 1
Asymbol(a::Z2Irrep, b::Z2Irrep, c::Z2Irrep) = Int(Nsymbol(a, b, c))
Bsymbol(a::Z2Irrep, b::Z2Irrep, c::Z2Irrep) = Int(Nsymbol(a, b, c))
Rsymbol(a::Z2Irrep, b::Z2Irrep, c::Z2Irrep) = Int(Nsymbol(a, b, c))

function fusiontensor(a::Z2Irrep, b::Z2Irrep, c::Z2Irrep)
    return fill(Int(Nsymbol(a, b, c)), (1, 1, 1, 1))
end



# modulus(c::Z2Irrep) = 2
# modulus(::Type{Z2Irrep}) = 2


charge(c::Z2Irrep) = Int(c.n)

Base.convert(::Type{Z2Irrep}, n::Real) = Z2Irrep(n)

unit(::Type{Z2Irrep}) = Z2Irrep(zero(UInt8))
dual(c::Z2Irrep) = c # typeof(c)(N - c.n)

# ⊗() = (one(Z2Irrep),)
# ⊗(c::Z2Irrep) = (c,)
# ⊗(c::Z2Irrep, cs::Vararg{Z2Irrep}) = (Z2Irrep(sum(c.n for c in cs) + c.n),)
⊗() = one(Z2Irrep)
⊗(c::Z2Irrep) = c
⊗(c::Z2Irrep, cs::Vararg{Z2Irrep}) = Z2Irrep(sum(c.n for c in cs) + c.n)
const otimes = ⊗


Base.hash(c::ZNIrrep, h::UInt) = hash(c.n, h)
Base.isless(c1::I, c2::I) where {I <: ZNIrrep} = isless(c1.n, c2.n)

unit(a::Z2Irrep) = unit(typeof(a))
Base.one(a::Z2Irrep) = unit(a)
Base.one(::Type{Z2Irrep}) = unit(Z2Irrep)

function isunit(a::Z2Irrep)
    a == unit(a)
end
Base.isone(a::Z2Irrep) = isunit(a)

Base.conj(a::Z2Irrep) = dual(a)

dim(a::Z2Irrep) = 1

