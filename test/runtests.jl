using PLS
using JLD
using Base.Test
reload("PLS")

@testset "PLS2 Prediction Tests (in sample)" begin

	@testset "Single Column Prediction Test" begin

		X        = reshape([1; 2; 3.0],(3,1))
		Y        = [1 1; 2 2; 3 3.0]
		model    = PLS.fit(X,Y,nfactors=1)
		pred     = PLS.transform(model,X)

		@test isequal(round.(pred),[1 1; 2 2; 3 3.0])

	end

	@testset "Constant Values Prediction Tests (Ax + b) | A=0, b=1 " begin

		X        = [1 3;2 1;3 2.0]
		Y        = [1 1; 1 1; 1 1.0]
		try PLS.fit(X,Y,nfactors=2) catch @test true end

	end

	@testset "Linear Prediction Tests " begin


		X        = [1 2; 2 4; 4 6.0]
		Y        = [4 2;6 4;8 6.0]
		model    = PLS.fit(X,Y,nfactors=2)
		pred     = PLS.transform(model,X)
		@test isequal(round.(pred),[4 2;6 4;8 6.0])

		X           = [1 -2; 2 -4; 4 -6.0]
		Y           = [-4 -2;-6 -4;-8 -6.0]
		model       = PLS.fit(X,Y,nfactors=2)
		pred        = PLS.transform(model,X)
		@test isequal(round.(pred),[-4 -2;-6 -4;-8 -6.0])


	end


end

@testset "PLS2 Pediction Tests (out of sample)" begin


	@testset "Linear Prediction Tests (Ax + b) | A>0" begin


		Xtr        = [1 2;2 4;3 6;6 12;7 14.0]
		Ytr        = [2 2;4 4;6 6;12 12;14 14.0]
		Xt         = [4 8;5 10.0]
		model    = PLS.fit(Xtr,Ytr,nfactors=2)
		pred     = PLS.transform(model,Xt)
		@test isequal(round.(pred),[8 8;10 10.0])

        #=
		Xtr        = [1 2; 2 4; 4.0 6]
		Ytr        = [6 4;8 6;10 8.0]
		Xt         = [6 8; 8 10; 10.0 12]
		model    = PLS.fit(Xtr,Ytr,nfactors=2)
		pred     = PLS.transform(model,Xt)
		@test isequal(round.(pred),[12 10;14 12;16 14.0])
        =#

	end
    #=
	@testset "Linear Prediction Tests (Ax + b) | A<0" begin



		Xtr        = [1 -2; 2 -4; 4.0 -6]
		Ytr        = [-4 -2;-6 -4;-8 -6.0]
		Xt         = [6 -8; 8 -10; 10.0 -12]
		model    = PLS.fit(Xtr,Ytr,nfactors=2)
		pred     = PLS.transform(model,Xt)
		@test isequal(round.(pred),[-10 -8;-12 -10;-14 -12.0])


		Xtr        = [1 -2; 2 -4; 4.0 -6]
		Ytr        = [-6 -4;-8 -6;-10 -8.0]
		Xt         = [6 -8; 8 -10; 10.0 -12]
		model    = PLS.fit(Xtr,Ytr,nfactors=2)
		pred     = PLS.transform(model,Xt)
		@test isequal(round.(pred),[-12 -10;-14 -12;-16 -14.0])

	end
    =#
end;


const MODEL_FILENAME = "pls_model.jld" # jld filename for storing the model

@testset "Test Saving and Loading Models" begin



	Xtr        = [1 -2; 2 -4; 4.0 -6]
	Ytr        = [-2; -4; -6.0]
	Xt         = [6 -8; 8 -10; 10.0 -12]
	model1    = PLS.fit(Xtr,Ytr,nfactors=2)
	pred1     = PLS.transform(model1,Xt)

	PLS.save(model1)
	model2    = PLS.load()

	pred2     = PLS.transform(model2,Xt)
    rm(MODEL_FILENAME)
	@test all(pred1 .== pred2)


end

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

end;


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

end;


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
