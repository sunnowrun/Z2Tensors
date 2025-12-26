println("------------------------------------")
println("|     Fields and vector spaces     |")
println("------------------------------------")
@timedtestset "Fields and vector spaces" verbose = true begin
    @timedtestset "HomSpace" begin
        for (V1, V2, V3, V4, V5) in (VZ2,)
            W = TK.HomSpace(V1 ⊗ V2, V3 ⊗ V4 ⊗ V5)
            @test W == (V3 ⊗ V4 ⊗ V5 → V1 ⊗ V2)
            @test W == (V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5)
            @test W' == (V1 ⊗ V2 → V3 ⊗ V4 ⊗ V5)
            # @test eval(Meta.parse(sprint(show, W))) == W
            # @test eval(Meta.parse(sprint(show, typeof(W)))) == typeof(W)
            @test spacetype(W) == typeof(V1)
            @test sectortype(W) == sectortype(V1)
            @test W[1] == V1
            @test W[2] == V2
            @test W[3] == V3'
            @test W[4] == V4'
            @test W[5] == V5'
            @test @constinferred(hash(W)) == hash(deepcopy(W)) != hash(W')
            @test W == deepcopy(W)
            @test W == @constinferred permute(W, ((1, 2), (3, 4, 5)))
            @test permute(W, ((2, 4, 5), (3, 1))) == (V2 ⊗ V4' ⊗ V5' ← V3 ⊗ V1')
            @test (V1 ⊗ V2 ← V1 ⊗ V2) == @constinferred TK.compose(W, W')
        end
    end
end
