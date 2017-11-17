

@testset "Auxiliary Functions Test" begin

    @testset "check constant columns" begin

		try PLS.check_constant_cols(Matrix([1.0 1;1 2;1 3])) catch @test true end
		try PLS.check_constant_cols(Matrix([1.0;1;1])) catch @test true end
		try PLS.check_constant_cols(Matrix([1.0 2 3])) catch @test true end
		try PLS.check_constant_cols([1.0; 1; 1]) catch @test true end

		@test PLS.check_constant_cols([1.0 1;2 2;3 3])
		@test PLS.check_constant_cols([1.0;2;3])

	end

	@testset "centralize" begin

		X        = reshape([1; 2; 3.0],(3,1))
		X        = PLS.centralize_data(X,mean(X,1),std(X,1))
		@test all(X .== [-1,0,1.0])

	end

	@testset "decentralize" begin

		Xo        = [1; 2; 3.0][:,:]
		Xn        = [-1,0,1.0][:,:]
		Xn        = PLS.decentralize_data(Xn,mean(Xo,1),std(Xo,1))
		@test all(Xn .== [1; 2; 3.0])

	end

	@testset "checkdata" begin

         try PLS.check_params(2,1,"linear") catch @test true end
		 try PLS.check_params(-1,2,"linear") catch @test true end
		 try PLS.check_params(1,2,"x") catch @test true end

		 @test PLS.check_params(1,2,"linear")

	end

	@testset "checkparams" begin

		 try PLS.check_data(Matrix{Float64}(0,0), 0) catch @test true end
		 try PLS.check_data(Matrix{Float64}(1,1), 10) catch @test true end
		 @test PLS.check_data(Matrix{Float64}(1,1), 1)

	end

end;
