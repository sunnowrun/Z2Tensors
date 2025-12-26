
spacelist = (VZ2,) # (Vtr, VZ2, Vℤ₃, VU₁)#, VSU₃)


for V in spacelist
    I = sectortype(first(V))
    Istr = "$I" # TK.type_repr(I)
    println("---------------------------------------")
    println("Tensors with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensors with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        @timedtestset "Basic tensor properties" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            for T in (Int, Float32, Float64, ComplexF32, ComplexF64, BigFloat)
                t = @constinferred zeros(T, W)
                @test @constinferred(hash(t)) == hash(deepcopy(t))
                @test scalartype(t) == T
                @test norm(t) == 0
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) == TensorMap{T,spacetype(t),5,0,Vector{T}}
                # blocks
                bs = @constinferred blocks(t)
                (c, b1), state = @constinferred Nothing iterate(bs)
                @test c == first(blocksectors(W))
                next = @constinferred Nothing iterate(bs, state)
                b2 = @constinferred block(t, first(blocksectors(t)))
                @test b1 == b2
                @test eltype(bs) === typeof(b1) === TK.blocktype(t)
            end
        end
        @timedtestset "Tensor Dict conversion" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            for T in (Int, Float32, ComplexF64)
                t = @constinferred rand(T, W)
                d = convert(Dict, t)
                @test t == convert(TensorMap, d)
            end
        end
        if hasfusiontensor(I)
            @timedtestset "Tensor Array conversion" begin
                W1 = V1 ← one(V1)
                W2 = one(V2) ← V2
                W3 = V1 ⊗ V2 ← one(V1)
                W4 = V1 ← V2
                W5 = one(V1) ← V1 ⊗ V2
                W6 = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
                for W in (W1, W2, W3, W4, W5, W6)
                    for T in (Int, Float32, ComplexF64)
                        if T == Int
                            t = TensorMap{T}(undef, W)
                            for (_, b) in blocks(t)
                                rand!(b, -20:20)
                            end
                        else
                            t = @constinferred randn(T, W)
                        end
                        a = @constinferred convert(Array, t)
                        # b = reshape(a, dim(codomain(W)), dim(domain(W)))
                        # @test t ≈ @constinferred TensorMap(a, W)
                        # @test t ≈ @constinferred TensorMap(b, W)
                        # @test t === @constinferred TensorMap(t.data, W)
                    end
                end
                for T in (Int, Float32, ComplexF64)
                    t = randn(T, V1 ⊗ V2 ← zero(V1))
                    a = convert(Array, t)
                    @test norm(a) == 0
                end
            end
        end
        @timedtestset "Basic linear algebra" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            for T in (Float32, ComplexF64)
                t = @constinferred rand(T, W)
                @test scalartype(t) == T
                @test space(t) == W
                @test space(t') == W'
                @test dim(t) == dim(space(t))
                @test codomain(t) == codomain(W)
                @test domain(t) == domain(W)
                # blocks for adjoint
                bs = @constinferred blocks(t')
                (c, b1), state = @constinferred Nothing iterate(bs)
                @test c == first(blocksectors(W'))
                next = @constinferred Nothing iterate(bs, state)
                b2 = @constinferred block(t', first(blocksectors(t')))
                @test b1 == b2
                @test eltype(bs) === typeof(b1) === TK.blocktype(t')
                # linear algebra
                @test isa(@constinferred(norm(t)), real(T))
                @test norm(t)^2 ≈ dot(t, t)
                α = rand(T)
                @test norm(α * t) ≈ abs(α) * norm(t)
                @test norm(t + t, 2) ≈ 2 * norm(t, 2)
                @test norm(t + t, 1) ≈ 2 * norm(t, 1)
                @test norm(t + t, Inf) ≈ 2 * norm(t, Inf)
                p = 3 * rand(Float64)
                @test norm(t + t, p) ≈ 2 * norm(t, p)
                @test norm(t) ≈ norm(t')

                t2 = @constinferred rand!(similar(t))
                β = rand(T)
                @test @constinferred(dot(β * t2, α * t)) ≈ conj(β) * α * conj(dot(t, t2))
                @test dot(t2, t) ≈ conj(dot(t, t2))
                @test dot(t2, t) ≈ conj(dot(t2', t'))
                @test dot(t2, t) ≈ dot(t', t2')

                i1 = @constinferred(isomorphism(T, V1 ⊗ V2, V2 ⊗ V1))
                i2 = @constinferred(isomorphism(Vector{T}, V2 ⊗ V1, V1 ⊗ V2))
                @test i1 * i2 == @constinferred(id(T, V1 ⊗ V2))
                @test i2 * i1 == @constinferred(id(Vector{T}, V2 ⊗ V1))

                w = @constinferred(isometry(T, V1 ⊗ (oneunit(V1) ⊕ oneunit(V1)),
                                            V1))
                @test dim(w) == 2 * dim(V1 ← V1)
                @test w' * w == id(Vector{T}, V1)
                @test w * w' == (w * w')^2
            end
        end

        if hasfusiontensor(I)
            @timedtestset "Basic linear algebra: test via conversion" begin
                W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
                for T in (Float32, ComplexF64)
                    t = rand(T, W)
                    t2 = @constinferred rand!(similar(t))
                    @test norm(t, 2) ≈ norm(convert(Array, t), 2)
                    @test dot(t2, t) ≈ dot(convert(Array, t2), convert(Array, t))
                    α = rand(T)
                    @test convert(Array, α * t) ≈ α * convert(Array, t)
                    @test convert(Array, t + t) ≈ 2 * convert(Array, t)
                end
            end
            @timedtestset "Real and imaginary parts" begin
                W = V1 ⊗ V2
                for T in (Float64, ComplexF64, ComplexF32)
                    t = @constinferred randn(T, W, W)

                    tr = @constinferred real(t)
                    @test scalartype(tr) <: Real
                    @test real(convert(Array, t)) == convert(Array, tr)

                    ti = @constinferred imag(t)
                    @test scalartype(ti) <: Real
                    @test imag(convert(Array, t)) == convert(Array, ti)

                    tc = @inferred complex(t)
                    @test scalartype(tc) <: Complex
                    @test complex(convert(Array, t)) == convert(Array, tc)

                    tc2 = @inferred complex(tr, ti)
                    @test tc2 ≈ tc
                end
            end
        end
        @timedtestset "Tensor conversion" begin
            W = V1 ⊗ V2
            t = @constinferred randn(W ← W)
            @test typeof(convert(TensorMap, t')) == typeof(t)
            tc = complex(t)
            @test convert(typeof(tc), t) == tc
            @test typeof(convert(typeof(tc), t)) == typeof(tc)
            @test typeof(convert(typeof(tc), t')) == typeof(tc)
            @test Base.promote_typeof(t, tc) == typeof(tc)
            @test Base.promote_typeof(tc, t) == typeof(tc + t)
        end
        @timedtestset "Permutations: test via inner product invariance" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            t = rand(ComplexF64, W)
            t′ = randn!(similar(t))
            for k in 0:5
                for p in permutations(1:5)
                    p1 = ntuple(n -> p[n], k)
                    p2 = ntuple(n -> p[k + n], 5 - k)
                    t2 = @constinferred permute(t, (p1, p2))
                    @test norm(t2) ≈ norm(t)
                    t2′ = permute(t′, (p1, p2))
                    @test dot(t2′, t2) ≈ dot(t′, t) #≈ dot(transpose(t2′), transpose(t2))
                end

                # t3 = VERSION < v"1.7" ? repartition(t, k) :
                #      @constinferred repartition(t, $k)
                # @test norm(t3) ≈ norm(t)
                # t3′ = @constinferred repartition!(similar(t3), t′)
                # @test norm(t3′) ≈ norm(t′)
                # @test dot(t′, t) ≈ dot(t3′, t3)
            end
        end
        begin
            @timedtestset "Permutations: test via conversion" begin
                W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
                t = rand(ComplexF64, W)
                a = convert(Array, t)
                for k in 0:5
                    for p in permutations(1:5)
                        p1 = ntuple(n -> p[n], k)
                        p2 = ntuple(n -> p[k + n], 5 - k)
                        t2 = permute(t, (p1, p2))
                        a2 = convert(Array, t2)
                        @test a2 ≈ permutedims(a, (p1..., p2...))
                        # @test convert(Array, transpose(t2)) ≈
                        #       permutedims(a2, (5, 4, 3, 2, 1))
                    end

                    # t3 = repartition(t, k)
                    # a3 = convert(Array, t3)
                    # @test a3 ≈ permutedims(a,
                    #                        (ntuple(identity, k)...,
                    #                         reverse(ntuple(i -> i + k, 5 - k))...))
                end
            end
        end
        @timedtestset "Full trace: test self-consistency" begin
            t = rand(ComplexF64, V1 ⊗ V2' ⊗ V2 ⊗ V1')
            t2 = permute(t, ((1, 2), (4, 3)))
            s = @constinferred tr(t2)
            @test conj(s) ≈ tr(t2')
            # if !isdual(V1)
            #     t2 = twist!(t2, 1)
            # end
            # if isdual(V2)
            #     t2 = twist!(t2, 2)
            # end
            ss = tr(t2)
            @tensor s2 = t[a, b, b, a]
            @tensor t3[a, b] := t[a, c, c, b]
            @tensor s3 = t3[a, a]
            @test ss ≈ s2
            @test ss ≈ s3
        end
        @timedtestset "Partial trace: test self-consistency" begin
            t = rand(ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3')
            @tensor t2[a, b] := t[c, d, b, d, c, a]
            @tensor t4[a, b, c, d] := t[d, e, b, e, c, a]
            @tensor t5[a, b] := t4[a, b, c, c]
            @test t2 ≈ t5
        end
        begin
            @timedtestset "Trace: test via conversion" begin
                t = rand(ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3')
                @tensor t2[a, b] := t[c, d, b, d, c, a]
                @tensor t3[a, b] := convert(Array, t)[c, d, b, d, c, a]
                @test t3 ≈ convert(Array, t2)
            end
        end
        @timedtestset "Trace and contraction" begin
            t1 = rand(ComplexF64, V1 ⊗ V2 ⊗ V3)
            t2 = rand(ComplexF64, V2' ⊗ V4 ⊗ V1')
            @tensor t3[1,2,3,4,5,6] := t1[1,2,3] * t2[4,5,6]
            # t3 = t1 ⊗ t2
            @tensor ta[a, b] := t1[x, y, a] * t2[y, b, x]
            @tensor tb[a, b] := t3[x, y, a, y, b, x]
            @test ta ≈ tb
        end
        begin
            @timedtestset "Tensor contraction: test via conversion" begin
                A1 = randn(ComplexF64, V1' ⊗ V2', V3')
                A2 = randn(ComplexF64, V3 ⊗ V4, V5)
                rhoL = randn(ComplexF64, V1, V1)
                rhoR = randn(ComplexF64, V5, V5)' # test adjoint tensor
                H = randn(ComplexF64, V2 ⊗ V4, V2 ⊗ V4)
                @tensor HrA12[a, s1, s2, c] := rhoL[a, a'] * conj(A1[a', t1, b]) *
                                               A2[b, t2, c'] * rhoR[c', c] *
                                               H[s1, s2, t1, t2]

                @tensor HrA12array[a, s1, s2, c] := convert(Array, rhoL)[a, a'] *
                                                    conj(convert(Array, A1)[a', t1, b]) *
                                                    convert(Array, A2)[b, t2, c'] *
                                                    convert(Array, rhoR)[c', c] *
                                                    convert(Array, H)[s1, s2, t1, t2]

                @test HrA12array ≈ convert(Array, HrA12)
            end
        end
        @timedtestset "diag/diagm" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            t = randn(ComplexF64, W)
            d = LinearAlgebra.diag(t)
            D = LinearAlgebra.diagm(codomain(t), domain(t), d)
            @test LinearAlgebra.isdiag(D)
            @test LinearAlgebra.diag(D) == d
        end
        @timedtestset "Factorization" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            for T in (Float32, ComplexF64)
                # Test both a normal tensor and an adjoint one.
                ts = (rand(T, W), rand(T, W)')
                for t in ts
                    @testset "leftorth with $alg" for alg in
                                                      (TK.QR(), TK.QRpos(),
                                                       TK.Polar(), TK.SVD(),
                                                       TK.SDD())
                        Q, R = @constinferred leftorth(t, ((3, 4, 2), (1, 5)); alg=alg)
                        QdQ = Q' * Q
                        @test QdQ ≈ one(QdQ)
                        @test Q * R ≈ permute(t, ((3, 4, 2), (1, 5)))
                        if alg isa Polar
                            # @test isposdef(R)
                            @test domain(R) == codomain(R) == space(t, 1)' ⊗ space(t, 5)'
                        end
                    end
                    @testset "rightorth with $alg" for alg in
                                                       (TK.LQ(), TK.LQpos(),
                                                        TK.Polar(), TK.SVD(),
                                                        TK.SDD())
                        L, Q = @constinferred rightorth(t, ((3, 4), (2, 1, 5)); alg=alg)
                        QQd = Q * Q'
                        @test QQd ≈ one(QQd)
                        @test L * Q ≈ permute(t, ((3, 4), (2, 1, 5)))
                        if alg isa Polar
                            # @test isposdef(L)
                            @test domain(L) == codomain(L) == space(t, 3) ⊗ space(t, 4)
                        end
                    end
                    @testset "tsvd with $alg" for alg in (TK.SVD(), TK.SDD())
                        U, S, V = @constinferred tsvd(t, ((3, 4, 2), (1, 5)); alg=alg)
                        UdU = U' * U
                        @test UdU ≈ one(UdU)
                        VVd = V * V'
                        @test VVd ≈ one(VVd)
                        t2 = permute(t, ((3, 4, 2), (1, 5)))
                        @test U * S * V ≈ t2

                        s = LinearAlgebra.svdvals(t2)
                        s′ = LinearAlgebra.diag(S)
                        for (c, b) in s
                            @test b ≈ s′[c]
                        end
                    end
                end
                @testset "empty tensor" begin
                    t = randn(T, V1 ⊗ V2, zero(V1))
                    @testset "leftorth with $alg" for alg in
                                                      (TK.QR(), TK.QRpos(),
                                                       TK.Polar(), TK.SVD(),
                                                       TK.SDD())
                        Q, R = @constinferred leftorth(t; alg=alg)
                        @test Q == t
                        @test dim(Q) == dim(R) == 0
                    end
                    @testset "rightorth with $alg" for alg in
                                                       (TK.LQ(), TK.LQpos(),
                                                        TK.Polar(), TK.SVD(),
                                                        TK.SDD())
                        L, Q = @constinferred rightorth(copy(t'); alg=alg)
                        @test Q == t'
                        @test dim(Q) == dim(L) == 0
                    end
                    @testset "tsvd with $alg" for alg in (TK.SVD(), TK.SDD())
                        U, S, V = @constinferred tsvd(t; alg=alg)
                        @test U == t
                        @test dim(U) == dim(S) == dim(V)
                    end
                end
                # t = rand(T, V1 ⊗ V1' ⊗ V2 ⊗ V2')
            end
        end
        @timedtestset "Tensor truncation" begin
            for T in (Float32, ComplexF64)
                for p in (1, 2, 3, Inf)
                    # Test both a normal tensor and an adjoint one.
                    ts = (randn(T, V1 ⊗ V2 ⊗ V3, V4 ⊗ V5),
                          randn(T, V4 ⊗ V5, V1 ⊗ V2 ⊗ V3)')
                    for t in ts
                        U₀, S₀, V₀, = tsvd(t)
                        t = rmul!(t, 1 / norm(S₀, p))
                        U, S, V, ϵ = @constinferred tsvd(t; trunc=truncerr(5e-1), p=p)
                        U′, S′, V′, ϵ′ = tsvd(t; trunc=truncerr(nextfloat(ϵ)), p=p)
                        @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                        U′, S′, V′, ϵ′ = tsvd(t; trunc=truncdim(ceil(Int, dim(domain(S)))), p=p)
                        @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                        U′, S′, V′, ϵ′ = tsvd(t; trunc=truncspace(space(S, 1)), p=p)
                        @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                        # results with truncationcutoff cannot be compared because they don't take degeneracy into account, and thus truncate differently
                        U, S, V, ϵ = tsvd(t; trunc=truncbelow(1 / dim(domain(S₀))), p=p)
                        U′, S′, V′, ϵ′ = tsvd(t; trunc=truncspace(space(S, 1)), p=p)
                        @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                    end
                end
            end
        end
    end
end

@timedtestset "Deligne tensor product: test via conversion" begin
    @testset for Vlist1 in (VZ2,), Vlist2 in (VZ2,)
        V1, V2, V3, V4, V5 = Vlist1
        W1, W2, W3, W4, W5 = Vlist2
        for T in (Float32, ComplexF64)
            t1 = rand(T, V1 ⊗ V2, V3' ⊗ V4)
            t2 = rand(T, W2, W1 ⊗ W1')
            @tensor t[1,2,5;3,4,6,7] := t1[1,2,3,4] * t2[5,6,7]
            d1 = dim(codomain(t1))
            d2 = dim(codomain(t2))
            d3 = dim(domain(t1))
            d4 = dim(domain(t2))
            At = convert(Array, t)
            @test reshape(At, (d1, d2, d3, d4)) ≈
                  reshape(convert(Array, t1), (d1, 1, d3, 1)) .*
                  reshape(convert(Array, t2), (1, d2, 1, d4))
        end
    end
end
