# TensorMap & Tensor:
# general tensor implementation with arbitrary symmetries
#==========================================================#


struct TensorMap{T,S<:IndexSpace,N₁,N₂,A<:DenseVector{T}} <: AbstractTensorMap{T,S,N₁,N₂}
    data::A
    space::TensorMapSpace{S,N₁,N₂}

    # uninitialized constructors
    function TensorMap{T,S,N₁,N₂,A}(::UndefInitializer,
                                    space::TensorMapSpace{S,N₁,N₂}) where {T,S<:IndexSpace,
                                                                           N₁,N₂,
                                                                           A<:DenseVector{T}}
        d = fusionblockstructure(space).totaldim
        data = A(undef, d)
        if !isbitstype(T)
            zerovector!(data)
        end
        return TensorMap{T,S,N₁,N₂,A}(data, space)
    end

    # constructors from data
    function TensorMap{T,S,N₁,N₂,A}(data::A,
                                    space::TensorMapSpace{S,N₁,N₂}) where {T,S<:IndexSpace,
                                                                           N₁,N₂,
                                                                           A<:DenseVector{T}}
        # T ⊆ field(S) || @warn("scalartype(data) = $T ⊈ $(field(S)))", maxlog = 1)
        I = sectortype(S)
        # T <: Real && !(sectorscalartype(I) <: Real) &&
        #     @warn("Tensors with real data might be incompatible with sector type $I",
        #           maxlog = 1)
        return new{T,S,N₁,N₂,A}(data, space)
    end
end

const Tensor{T,S,N,A} = TensorMap{T,S,N,0,A}

function tensormaptype(S::Type{<:IndexSpace}, N₁, N₂, TorA::Type)
    if TorA <: Number
        return TensorMap{TorA,S,N₁,N₂,Vector{TorA}}
    elseif TorA <: DenseVector
        return TensorMap{scalartype(TorA),S,N₁,N₂,TorA}
    else
        throw(ArgumentError("argument $TorA should specify a scalar type (`<:Number`) or a storage type `<:DenseVector{<:Number}`"))
    end
end

# Basic methods for characterising a tensor:
#--------------------------------------------
space(t::TensorMap) = t.space

storagetype(::Type{<:TensorMap{T,S,N₁,N₂,A}}) where {T,S,N₁,N₂,A<:DenseVector{T}} = A

dim(t::TensorMap) = length(t.data)

# General TensorMap constructors
#--------------------------------
# undef constructors
function TensorMap{T}(::UndefInitializer, V::TensorMapSpace{S,N₁,N₂}) where {T,S,N₁,N₂}
    return TensorMap{T,S,N₁,N₂,Vector{T}}(undef, V)
end
function TensorMap{T}(::UndefInitializer, codomain::TensorSpace{S},
                      domain::TensorSpace{S}) where {T,S}
    return TensorMap{T}(undef, codomain ← domain)
end
function Tensor{T}(::UndefInitializer, V::TensorSpace{S}) where {T,S}
    return TensorMap{T}(undef, V ← one(V))
end

# constructor starting from vector = independent data (N₁ + N₂ = 1 is special cased below)
# documentation is captured by the case where `data` is a general array
# here, we force the `T` argument to distinguish it from the more general constructor below
function TensorMap{T}(data::A,
                      V::TensorMapSpace{S,N₁,N₂}) where {T,S,N₁,N₂,A<:DenseVector{T}}
    return TensorMap{T,S,N₁,N₂,A}(data, V)
end
function TensorMap{T}(data::DenseVector{T}, codomain::TensorSpace{S},
                      domain::TensorSpace{S}) where {T,S}
    return TensorMap(data, codomain ← domain)
end

# constructor starting from block data
function TensorMap(data::AbstractDict{<:Sector,<:AbstractMatrix},
                   V::TensorMapSpace{S,N₁,N₂}) where {S,N₁,N₂}
    T = eltype(valtype(data))
    t = TensorMap{T}(undef, V)
    for (c, b) in blocks(t)
        haskey(data, c) || throw(SectorMismatch("no data for block sector $c"))
        datac = data[c]
        size(datac) == size(b) ||
            throw(DimensionMismatch("wrong size of block for sector $c"))
        copy!(b, datac)
    end
    for (c, b) in data
        c ∈ blocksectors(t) || isempty(b) ||
            throw(SectorMismatch("data for block sector $c not expected"))
    end
    return t
end
function TensorMap(data::AbstractDict{<:Sector,<:AbstractMatrix}, codom::TensorSpace{S},
                   dom::TensorSpace{S}) where {S}
    return TensorMap(data, codom ← dom)
end


for (fname, felt) in ((:zeros, :zero), (:ones, :one))
    @eval begin
        function Base.$fname(codomain::TensorSpace{S},
                             domain::TensorSpace{S}=one(codomain)) where {S<:IndexSpace}
            return Base.$fname(codomain ← domain)
        end
        function Base.$fname(::Type{T}, codomain::TensorSpace{S},
                             domain::TensorSpace{S}=one(codomain)) where {T,S<:IndexSpace}
            return Base.$fname(T, codomain ← domain)
        end
        Base.$fname(V::TensorMapSpace) = Base.$fname(Float64, V)
        function Base.$fname(::Type{T}, V::TensorMapSpace) where {T}
            t = TensorMap{T}(undef, V)
            fill!(t, $felt(T))
            return t
        end
    end
end

for randf in (:rand, :randn, :randexp)
    randfun = GlobalRef(Random, randf)
    randfun! = GlobalRef(Random, Symbol(randf, :!))

    @eval begin
        # converting `codomain` and `domain` into `HomSpace`
        function $randfun(codomain::TensorSpace{S},
                          domain::TensorSpace{S}) where {S<:IndexSpace}
            return $randfun(codomain ← domain)
        end
        function $randfun(::Type{T}, codomain::TensorSpace{S},
                          domain::TensorSpace{S}) where {T,S<:IndexSpace}
            return $randfun(T, codomain ← domain)
        end
        function $randfun(rng::Random.AbstractRNG, ::Type{T},
                          codomain::TensorSpace{S},
                          domain::TensorSpace{S}) where {T,S<:IndexSpace}
            return $randfun(rng, T, codomain ← domain)
        end

        # accepting single `TensorSpace`
        $randfun(codomain::TensorSpace) = $randfun(codomain ← one(codomain))
        function $randfun(::Type{T}, codomain::TensorSpace) where {T}
            return $randfun(T, codomain ← one(codomain))
        end
        function $randfun(rng::Random.AbstractRNG, ::Type{T},
                          codomain::TensorSpace) where {T}
            return $randfun(rng, T, codomain ← one(domain))
        end

        # filling in default eltype
        $randfun(V::TensorMapSpace) = $randfun(Float64, V)
        function $randfun(rng::Random.AbstractRNG, V::TensorMapSpace)
            return $randfun(rng, Float64, V)
        end

        # filling in default rng
        function $randfun(::Type{T}, V::TensorMapSpace) where {T}
            return $randfun(Random.default_rng(), T, V)
        end
        $randfun!(t::AbstractTensorMap) = $randfun!(Random.default_rng(), t)

        # implementation
        function $randfun(rng::Random.AbstractRNG, ::Type{T},
                          V::TensorMapSpace) where {T}
            t = TensorMap{T}(undef, V)
            $randfun!(rng, t)
            return t
        end

        function $randfun!(rng::Random.AbstractRNG, t::AbstractTensorMap)
            for (_, b) in blocks(t)
                $randfun!(rng, b)
            end
            return t
        end
    end
end

function TensorMap(data::AbstractVector, V::TensorMapSpace{S,N₁,N₂}) where {S<:IndexSpace,N₁,N₂}
    T = eltype(data)
    @assert length(data) == dim(V)
    if data isa DenseVector # refer to specific data-capturing constructor
        return TensorMap{T}(data, V)
    else
        return TensorMap{T}(collect(data), V)
    end
end
function TensorMap(data::AbstractArray, codom::TensorSpace{S}, dom::TensorSpace{S}) where {S}
    return TensorMap(data, codom ← dom)
end
function Tensor(data::AbstractArray, codom::TensorSpace)
    return TensorMap(data, codom ← one(codom))
end




# Efficient copy constructors
#-----------------------------
Base.copy(t::TensorMap) = typeof(t)(copy(t.data), t.space)

# Conversion between TensorMap and Dict, for read and write purpose
#------------------------------------------------------------------
function Base.convert(::Type{Dict}, t::AbstractTensorMap)
    d = Dict{Symbol,Any}()
    d[:codomain] = repr(codomain(t))
    d[:domain] = repr(domain(t))
    data = Dict{String,Any}()
    for (c, b) in blocks(t)
        data[repr(c)] = Array(b)
    end
    d[:data] = data
    return d
end
function Base.convert(::Type{TensorMap}, d::Dict{Symbol,Any})
    try
        codomain = eval(Meta.parse(d[:codomain]))
        domain = eval(Meta.parse(d[:domain]))
        data = SectorDict(eval(Meta.parse(c)) => b for (c, b) in d[:data])
        return TensorMap(data, codomain, domain)
    catch e # sector unknown in TensorKit.jl; user-defined, hopefully accessible in Main
        codomain = Base.eval(Main, Meta.parse(d[:codomain]))
        domain = Base.eval(Main, Meta.parse(d[:domain]))
        data = SectorDict(Base.eval(Main, Meta.parse(c)) => b for (c, b) in d[:data])
        return TensorMap(data, codomain, domain)
    end
end

# Getting and setting the data at the block level
#-------------------------------------------------
block(t::TensorMap, c::Sector) = blocks(t)[c]

blocks(t::TensorMap) = BlockIterator(t, fusionblockstructure(t).blockstructure)

function blocktype(::Type{TT}) where {TT<:TensorMap}
    A = storagetype(TT)
    T = eltype(A)
    return Base.ReshapedArray{T,2,SubArray{T,1,A,Tuple{UnitRange{Int}},true},Tuple{}}
end

function Base.iterate(iter::BlockIterator{<:TensorMap}, state...)
    next = iterate(iter.structure, state...)
    isnothing(next) && return next
    (c, (sz, r)), newstate = next
    return c => reshape(view(iter.t.data, r), sz), newstate
end

function Base.getindex(iter::BlockIterator{<:TensorMap}, c::Sector)
    sectortype(iter.t) === typeof(c) || throw(SectorMismatch())
    (d₁, d₂), r = get(iter.structure, c) do
        # is s is not a key, at least one of the two dimensions will be zero:
        # it then does not matter where exactly we construct a view in `t.data`,
        # as it will have length zero anyway
        d₁′ = blockdim(codomain(iter.t), c)
        d₂′ = blockdim(domain(iter.t), c)
        l = d₁′ * d₂′
        return (d₁′, d₂′), 1:l
    end
    return reshape(view(iter.t.data, r), (d₁, d₂))
end

# Indexing and getting and setting the data at the subblock level
#-----------------------------------------------------------------
@inline function Base.getindex(t::TensorMap{T,S,N₁,N₂},
                               f₁::FusionTree{I,N₁},
                               f₂::FusionTree{I,N₂}) where {T,S,N₁,N₂,I<:Sector}
    structure = fusionblockstructure(t)
    @boundscheck begin
        haskey(structure.fusiontreeindices, (f₁, f₂)) || throw(SectorMismatch())
    end
    @inbounds begin
        i = structure.fusiontreeindices[(f₁, f₂)]
        sz, str, offset = structure.fusiontreestructure[i]
        return StridedView(t.data, sz, str, offset)
    end
end


@propagate_inbounds function Base.setindex!(t::TensorMap{T,S,N₁,N₂},
                                            v,
                                            f₁::FusionTree{I,N₁},
                                            f₂::FusionTree{I,N₂}) where {T,S,N₁,N₂,
                                                                         I<:Sector}
    return copy!(getindex(t, f₁, f₂), v)
end

@inline function Base.getindex(t::TensorMap, sectors::Tuple{I,Vararg{I}}) where {I<:Sector}
    I === sectortype(t) || throw(SectorMismatch("Not a valid sectortype for this tensor."))
    # FusionStyle(I) isa UniqueFusion ||
    #     throw(SectorMismatch("Indexing with sectors only possible if unique fusion"))
    length(sectors) == numind(t) ||
        throw(ArgumentError("Number of sectors does not match."))
    s₁ = TupleTools.getindices(sectors, codomainind(t))
    s₂ = map(dual, TupleTools.getindices(sectors, domainind(t)))
    c1 = length(s₁) == 0 ? one(I) : (length(s₁) == 1 ? s₁[1] : ⊗(s₁...)) # first(⊗(s₁...))
    @boundscheck begin
        c2 = length(s₂) == 0 ? one(I) : (length(s₂) == 1 ? s₂[1] : ⊗(s₂...)) # first(⊗(s₂...))
        c2 == c1 || throw(SectorMismatch("Not a valid sector for this tensor"))
        hassector(codomain(t), s₁) && hassector(domain(t), s₂)
    end
    f₁ = FusionTree(s₁, c1)
    f₂ = FusionTree(s₂, c1)
    @inbounds begin
        return t[f₁, f₂]
    end
end
@propagate_inbounds function Base.getindex(t::TensorMap, sectors::Tuple)
    return t[map(sectortype(t), sectors)]
end

# Complex, real and imaginary parts
#-----------------------------------
for f in (:real, :imag, :complex)
    @eval begin
        function Base.$f(t::TensorMap)
            return TensorMap($f(t.data), space(t))
        end
    end
end

# Conversion and promotion:
#---------------------------
Base.convert(::Type{TensorMap}, t::TensorMap) = t
function Base.convert(::Type{TensorMap}, t::AbstractTensorMap)
    return copy!(TensorMap{scalartype(t)}(undef, space(t)), t)
end

function Base.convert(TT::Type{TensorMap{T,S,N₁,N₂,A}},
                      t::AbstractTensorMap{<:Any,S,N₁,N₂}) where {T,S,N₁,N₂,A}
    if typeof(t) === TT
        return t
    else
        tnew = TT(undef, space(t))
        return copy!(tnew, t)
    end
end

function Base.promote_rule(::Type{<:TT₁},
                           ::Type{<:TT₂}) where {S,N₁,N₂,
                                                 TT₁<:TensorMap{<:Any,S,N₁,N₂},
                                                 TT₂<:TensorMap{<:Any,S,N₁,N₂}}
    A = VectorInterface.promote_add(storagetype(TT₁), storagetype(TT₂))
    T = scalartype(A)
    return TensorMap{T,S,N₁,N₂,A}
end



function Base.empty(::Type{TensorMap{T,S,N₁,N₂,A}}) where {T,S<:IndexSpace,N₁,N₂,A<:DenseVector{T}}
    space = ⊗((zero(S) for i in 1:N₁)...) ← ⊗((zero(S) for i in 1:N₂)...)
    TensorMap{T,S,N₁,N₂,A}(undef, space)
end
Base.empty(t::TensorMap) = empty(typeof(t))

