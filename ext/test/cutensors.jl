ENV["CUDA_VISIBLE_DEVICES"] = 5

push!(LOAD_PATH, dirname(dirname(dirname(Base.@__DIR__))) * "/Z2TensorKit/ext/Z2TensorKitCUDAExt/")
using CUDA, Z2TensorKitCUDAExt

spacelist = (VZ2,) # (Vtr, VZ2, Vℤ₃, VU₁)#, VSU₃)


for V in spacelist
    I = sectortype(first(V))
    Istr = "$I" # TK.type_repr(I)
    println("---------------------------------------")
    println("CuTensors with symmetry: $Istr")
    println("---------------------------------------")
    @timedtestset "Tensors with symmetry: $Istr" verbose = true begin
        V1, V2, V3, V4, V5 = V
        @timedtestset "Basic tensor properties" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            for T in (Int, Float32, Float64, ComplexF32, ComplexF64)
                t = @constinferred zeros(T, W)
                t = tocu(t)
                # @test @constinferred(hash(t)) == hash(deepcopy(t))
                @test scalartype(t) == T
                @test norm(t) == 0
                @test codomain(t) == W
                @test space(t) == (W ← one(W))
                @test domain(t) == one(W)
                @test typeof(t) == TensorMap{T, spacetype(t), 5, 0, CuArray{T, 1, CUDA.DeviceMemory}}
                # blocks
                bs = @constinferred blocks(t)
                (c, b1), state = @constinferred Nothing iterate(bs)
                @test c == first(blocksectors(W))
                next = @constinferred Nothing iterate(bs, state)
                b2 = @constinferred block(t, first(blocksectors(t)))
                @test b1 == b2
                # @test eltype(bs) === typeof(b1) === TK.blocktype(t)
            end
        end

        @timedtestset "Basic linear algebra" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            for T in (Float32, ComplexF64)
                t = @constinferred rand(T, W)
                t = tocu(t)
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
                # @test eltype(bs) === typeof(b1) === TK.blocktype(t')
                # linear algebra
                @test isa(@constinferred(norm(t)), real(T))
                @test norm(t)^2 ≈ dot(t, t)
                α = rand(T)
                @test norm(α * t) ≈ abs(α) * norm(t)
                @test norm(t + t, 2) ≈ 2 * norm(t, 2)
                @test norm(t + t, 1) ≈ 2 * norm(t, 1)
                @test norm(t + t, Inf) ≈ 2 * norm(t, Inf)

                t2 = @constinferred rand!(similar(t))
                β = rand(T)
                @test @constinferred(dot(β * t2, α * t)) ≈ conj(β) * α * conj(dot(t, t2))
                @test dot(t2, t) ≈ conj(dot(t, t2))
                @test dot(t2, t) ≈ conj(dot(t2', t'))
                @test dot(t2, t) ≈ dot(t', t2')
            end
        end

        @timedtestset "Permutations: test via inner product invariance" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            t = tocu(rand(ComplexF64, W))
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
            end
        end

        @timedtestset "Full trace: test self-consistency" begin
            t = tocu(rand(ComplexF64, V1 ⊗ V2' ⊗ V2 ⊗ V1'))
            t2 = permute(t, ((1, 2), (4, 3)))
            s = @constinferred tr(t2)
            @test conj(s) ≈ tr(t2')

            ss = tr(t2)
            @tensor s2 = t[a, b, b, a]
            @tensor t3[a, b] := t[a, c, c, b]
            @tensor s3 = t3[a, a]
            @test ss ≈ s2
            @test ss ≈ s3
        end
        @timedtestset "Partial trace: test self-consistency" begin
            t = tocu(rand(ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3'))
            @tensor t2[a, b] := t[c, d, b, d, c, a]
            @tensor t4[a, b, c, d] := t[d, e, b, e, c, a]
            @tensor t5[a, b] := t4[a, b, c, c]
            @test t2 ≈ t5
        end

        @timedtestset "Trace and contraction" begin
            t1 = tocu(rand(ComplexF64, V1 ⊗ V2 ⊗ V3))
            t2 = tocu(rand(ComplexF64, V2' ⊗ V4 ⊗ V1'))
            @tensor t3[1,2,3,4,5,6] := t1[1,2,3] * t2[4,5,6]
            # t3 = t1 ⊗ t2
            @tensor ta[a, b] := t1[x, y, a] * t2[y, b, x]
            @tensor tb[a, b] := t3[x, y, a, y, b, x]
            @test ta ≈ tb

            t2 = tocu(rand(ComplexF64, V2 ⊗ V4 ⊗ V1))
            @tensor t3[1,2,3,4,5,6] := t1[1,2,3] * conj(t2[4,5,6])
            @tensor ta[a, b] := t1[x, y, a] * conj(t2[y, b, x])
            @tensor tb[a, b] := t3[x, y, a, y, b, x]
            @test ta ≈ tb
        end

        @timedtestset "Factorization" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            for T in (Float32, ComplexF64)
                # Test both a normal tensor and an adjoint one.
                ts = (tocu(rand(T, W)), tocu(rand(T, W))')
                for t in ts
                    @testset "leftorth with $alg" for alg in (TK.QR(), TK.QRpos(), TK.SVD(), TK.SDD())
                        Q, R = @constinferred leftorth(t, ((3, 4, 2), (1, 5)); alg=alg)
                        QdQ = Q' * Q
                        @test QdQ ≈ one(QdQ)
                        @test Q * R ≈ permute(t, ((3, 4, 2), (1, 5)))
                    end
                    @testset "rightorth with $alg" for alg in (TK.LQ(), TK.LQpos(), TK.SVD(), TK.SDD())
                        L, Q = @constinferred rightorth(t, ((3, 4), (2, 1, 5)); alg=alg)
                        QQd = Q * Q'
                        @test QQd ≈ one(QQd)
                        @test L * Q ≈ permute(t, ((3, 4), (2, 1, 5)))
                    end
                    @testset "tsvd with $alg" for alg in (TK.SVD(), TK.SDD())
                        U, S, V = @constinferred tsvd(t, ((3, 4, 2), (1, 5)); alg=alg)
                        UdU = U' * U
                        @test UdU ≈ one(UdU)
                        VVd = V * V'
                        @test VVd ≈ one(VVd)
                        t2 = permute(t, ((3, 4, 2), (1, 5)))
                        @test U * S * V ≈ t2
                    end
                end
                @testset "empty tensor" begin
                    t = tocu(randn(T, V1 ⊗ V2, zero(V1)))
                    @testset "leftorth with $alg" for alg in (TK.QR(), TK.QRpos(), TK.SVD(), TK.SDD())
                        Q, R = @constinferred leftorth(t; alg=alg)
                        @test Q == t
                        @test dim(Q) == dim(R) == 0
                    end
                    @testset "rightorth with $alg" for alg in (TK.LQ(), TK.LQpos(), TK.SVD(), TK.SDD())
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
            end
        end
        
        @timedtestset "Tensor truncation" begin
            for T in (Float32, ComplexF64)
                for p in (1, 2, 3, Inf)
                    # Test both a normal tensor and an adjoint one.
                    ts = (tocu(randn(T, V1 ⊗ V2 ⊗ V3, V4 ⊗ V5)),
                          tocu(randn(T, V4 ⊗ V5, V1 ⊗ V2 ⊗ V3))')
                    for t in ts
                        U₀, S₀, V₀, = tsvd(t)
                        t = rmul!(t, 1 / norm(S₀, p))
                        U, S, V, ϵ = @constinferred tsvd(t; trunc=truncerr(5e-1), p=p)
                        U′, S′, V′, ϵ′ = tsvd(t; trunc=truncerr(nextfloat(ϵ)), p=p)
                        @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                        U′, S′, V′, ϵ′ = tsvd(t; trunc=truncdim(ceil(Int, dim(domain(S)))),
                                              p=p)
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






# how is @tensor implemented
# why don't using cuTENSOR will raise ERROR
function test(d=10)
    s = Z2Space(0=>d, 1=>d)
    t1 = randn(ComplexF64, s⊗s, s⊗s⊗s)
    t2 = randn(ComplexF64, s⊗s, s⊗s⊗s)
    @time @tensor t3[4,1,6;5,7,8] := t1[1,2,3,4,5] * t2[3,6,2,7,8]
    
    cut1, cut2 = tocu(t1), tocu(t2)
    @time @tensor cut3[4,1,6;5,7,8] := cut1[1,2,3,4,5] * cut2[3,6,2,7,8]
    t3′ = fromcu(cut3)
    println(norm(t3′ - t3))
end
# test()
function test2(d=10)
    s = Z2Space(0=>d, 1=>d)
    t1 = randn(ComplexF64, s⊗s, s⊗s⊗s)
    @time res1 = tsvd(t1, ((1,3),(4,2,5)))
    # println(typeof(res1))

    cut1 = tocu(t1)
    # println(typeof(cut1))
    # qr_compact(cut1, CUSOLVER_HouseholderQR())
    @time res2 = tsvd(cut1, ((1,3),(4,2,5)))
    println(typeof(res2))
end
# test2()

function test3()
    s = Z2Space(0=>10, 1=>10)
    for t1 in (randn(ComplexF64, s⊗s, s⊗s⊗s), randn(ComplexF64, s⊗s⊗s, s⊗s))

        cut1 = tocu(t1)
        qr_compact(cut1, CUSOLVER_HouseholderQR())
        lq_compact(cut1, LQViaTransposedQR(CUSOLVER_HouseholderQR()))
        svd_compact(cut1, CUSOLVER_QRIteration())
        # svd_compact(cut1, CUSOLVER_Jacobi())
        # svd_compact(cut1, CUSOLVER_SVDPolar())
        # svd_compact(cut1, CUSOLVER_DivideAndConquer())

    end
    
    return
end
# test3()


function test_performance(d=10, N=5)
    s = Z2Space(0=>d, 1=>d)
    p = Z2Space(0=>1, 1=>1)
    t1 = randn(ComplexF64, s⊗p, s)
    t2 = randn(ComplexF64, s⊗p, s)
    println(Base.summarysize(t1) / (2^30), " GiB")

    println("permute")
    @time for i in 1:N
        permute(t1, ((3,), (1,2)))
    end
    @time for i in 1:N
        fromcu(permute(tocu(t1), ((3,), (1,2))))
    end

    println("contraction")
    @time for i in 1:N
        @tensor t3[1,2;4,5] := t1[1,2,3] * t2[3,4,5]
    end
    @time for i in 1:N
        cut1, cut2 = tocu(t1), tocu(t2)
        @tensor cut3[1,2;4,5] := cut1[1,2,3] * cut2[3,4,5]
        fromcu(cut3)
    end

    println("svd")
    @time for i in 1:N
        tsvd(t1, ((1,2),(3,)))
    end
    @time for i in 1:N
        u, s, vd = tsvd(tocu(t1), ((1,2),(3,)))
        fromcu(u)
        fromcu(vd)
    end
end
# test_performance()
# julia> test_performance(200, 100)
# 0.002384297549724579 GiB
# permute
#   0.113347 seconds (600 allocations: 244.173 MiB, 17.73% gc time)
#   0.145052 seconds (27.40 k allocations: 245.064 MiB, 3.72% gc time)
# contraction
#   2.081206 seconds (1.60 k allocations: 732.501 MiB, 0.10% gc time)
#   0.120427 seconds (50.70 k allocations: 489.865 MiB, 1.09% gc time)
# svd
#   4.720180 seconds (7.20 k allocations: 1.474 GiB, 0.14% gc time)
#   9.637170 seconds (7.14 M allocations: 667.847 MiB, 0.31% gc time)


