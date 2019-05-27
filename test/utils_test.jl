

@testset "Auxiliary Functions Test" begin

    @testset "check constant columns" begin

		try
			PLSRegressor.check_constant_cols(Matrix([1.0 1;1 2;1 3]))
		catch
			@test true
		end
		try
			PLSRegressor.check_constant_cols(Matrix([1.0;1;1]))
		catch
			@test true
		end
		try
			PLSRegressor.check_constant_cols(Matrix([1.0 2 3]))
		catch
			@test true
		end
		try
			PLSRegressor.check_constant_cols([1.0; 1; 1])
		catch
			@test true
		end

		@test PLSRegressor.check_constant_cols([1.0 1;2 2;3 3])
		@test PLSRegressor.check_constant_cols([1.0;2;3])

	end

	@testset "centralize" begin

		X        = reshape([1; 2; 3.0],(3,1))
		X        = PLSRegressor.centralize_data(X,mean(X,1),std(X,1))
		@test all(X .== [-1,0,1.0])

	end

	@testset "decentralize" begin

		Xo        = [1; 2; 3.0][:,:]
		Xn        = [-1,0,1.0][:,:]
		Xn        = PLSRegressor.decentralize_data(Xn,mean(Xo,1),std(Xo,1))
		@test all(Xn .== [1; 2; 3.0])

	end

	@testset "checkdata" begin

         try
			 PLSRegressor.check_params(2,1,"linear")
		 catch
			 @test true
		 end
		 try
			 PLSRegressor.check_params(-1,2,"linear")
		 catch
			 @test true
		 end
		 try
			 PLSRegressor.check_params(1,2,"x")
		 catch
			 @test true
		 end

		 @test PLSRegressor.check_params(1,2,"linear")

	end

	@testset "checkparams" begin

		 try
			 PLSRegressor.check_data(Matrix{Float64}(0,0), 0)
		 catch
			 @test true
		 end
		 try
			 PLSRegressor.check_data(Matrix{Float64}(1,1), 10)
		 catch
			 @test true
		 end
		 @test PLSRegressor.check_data(Matrix{Float64}(1,1), 1)

	end

end;
