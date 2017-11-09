using PLS
using Base.Test

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
		Xo        = reshape([1; 2; 3.0],(3,1))
		Xn        = reshape([-1,0,1.0],(3,1))
		Xn        = PLS.decentralize_data(Xn,mean(Xo,1),std(Xo,1))
		@test all(Xn .== [1; 2; 3.0])
	end

	@testset "checkdata" begin
         try PLS.check_params(2,1) catch @test true end
		 try PLS.check_params(-1,2) catch @test true end
		 @test PLS.check_params(1,2)
	end

	@testset "checkparams" begin
		 try PLS.check_data(Matrix{Float64}(0,0), 0) catch @test true end
		 try PLS.check_data(Matrix{Float64}(1,1), 10) catch @test true end
		 @test PLS.check_data(Matrix{Float64}(1,1), 1)
	end
end

@testset "Pediction Tests (in sample)" begin
	@testset "Single Column Prediction Test" begin
		X        = reshape([1; 2; 3.0],(3,1))
		Y        = [1; 2; 3.0]
		model    = PLS.fit(X,Y,nfactors=1)
		pred     = PLS.transform(model,X)
		@test isequal(round.(pred),[1; 2; 3.0])
	end

	@testset "Constant Values Prediction Tests (Ax + b) | A=0, b=1 " begin
		X        = [1 3;2 1;3 2.0]
		Y        = [1; 1; 1.0]
		try PLS.fit(X,Y,nfactors=2) catch @test true end
	end

	@testset "Linear Prediction Tests " begin
		X        = [1 2; 2 4; 4.0 6]
		Y        = [2; 4; 6.0]
		model    = PLS.fit(X,Y,nfactors=2)
		pred     = PLS.transform(model,X)
		@test isequal(round.(pred),[2; 4; 6.0])

		X           = [1 -2; 2 -4; 4.0 -6]
		Y           = [-2; -4; -6.0]
		model       = PLS.fit(X,Y,nfactors=2)
		pred        = PLS.transform(model,X)
		@test isequal(round.(pred),[-2; -4; -6.0])
	end

	@testset "Linear Prediction Tests (Ax + b)" begin
		Xtr        = [1 2; 2 4; 4.0 6]
		Ytr        = [2; 4; 6.0]
		Xt         = [6 8; 8 10; 10.0 12] # same sample
		model    = PLS.fit(Xtr,Ytr,nfactors=2)
		pred     = PLS.transform(model,Xt)
		@test isequal(round.(pred),[8; 10; 12.0])

		Xtr        = [1 2; 2 4.0; 4.0 6; 6 8]
		Ytr        = [2; 4; 6.0; 8]
		Xt         = [1 2; 2 4.0] # a subsample

		model    = PLS.fit(Xtr,Ytr,nfactors=2,centralize=true)
		pred     = PLS.transform(model,Xt)
		@test isequal(round.(pred),[2; 4])
	end
end

@testset "Pediction Tests (out of sample)" begin
	@testset "Linear Prediction Tests (Ax + b) | A>0" begin
		Xtr        = [1 2; 2 4; 4.0 6]
		Ytr        = [2; 4; 6.0]
		Xt         = [6 8; 8 10; 10.0 12]
		model    = PLS.fit(Xtr,Ytr,nfactors=2)
		pred     = PLS.transform(model,Xt)
		@test isequal(round.(pred),[8; 10; 12.0])

		Xtr        = [1 2; 2 4; 4.0 6]
		Ytr        = [4; 6; 8.0]
		Xt         = [6 8; 8 10; 10.0 12]
		model    = PLS.fit(Xtr,Ytr,nfactors=2)
		pred     = PLS.transform(model,Xt)
		@test isequal(round.(pred),[10; 12; 14.0])
	end

	@testset "Linear Prediction Tests (Ax + b) | A<0" begin
		Xtr        = [1 -2; 2 -4; 4.0 -6]
		Ytr        = [-2; -4; -6.0]
		Xt         = [6 -8; 8 -10; 10.0 -12]
		model    = PLS.fit(Xtr,Ytr,nfactors=2)
		pred     = PLS.transform(model,Xt)
		@test isequal(round.(pred),[-8; -10; -12.0])

		Xtr        = [1 -2; 2 -4; 4.0 -6]
		Ytr        = [-4; -6; -8.0]
		Xt         = [6 -8; 8 -10; 10.0 -12]
		model    = PLS.fit(Xtr,Ytr,nfactors=2)
		pred     = PLS.transform(model,Xt)
		@test isequal(round.(pred),[-10; -12; -14.0])
	end
end;
