@testset "Test Saving and Loading PLS1 Models" begin



	Xtr        = [1 -2; 2 -4; 4.0 -6]
	Ytr        = [-2; -4; -6.0]
	Xt         = [6 -8; 8 -10; 10.0 -12]
	model1    = PLS.fit(Xtr,Ytr,nfactors=2)
	pred1     = PLS.predict(model1,Xt)

	PLS.save(model1)
	model2    = PLS.load()

	pred2     = PLS.predict(model2,Xt)
    rm(PLS.MODEL_FILENAME)
	@test all(pred1 .== pred2)


end


@testset "PLS1 Pediction Tests (in sample)" begin

	@testset "Single Column Prediction Test" begin

		X        = [1; 2; 3.0][:,:]
		Y        = [1; 2; 3.0]
		model    = PLS.fit(X,Y,nfactors=1)
		pred     = PLS.predict(model,X)
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
		pred     = PLS.predict(model,X)
		@test isequal(round.(pred),[2; 4; 6.0])

		X           = [1 -2; 2 -4; 4.0 -6]
		Y           = [-2; -4; -6.0]
		model       = PLS.fit(X,Y,nfactors=2)
		pred        = PLS.predict(model,X)
		@test isequal(round.(pred),[-2; -4; -6.0])

	end

	@testset "Linear Prediction Tests (Ax + b)" begin


		Xtr        = [1 2; 2 4; 4.0 6]
		Ytr        = [2; 4; 6.0]
		Xt         = [6 8; 8 10; 10.0 12] # same sample
		model    = PLS.fit(Xtr,Ytr,nfactors=2)
		pred     = PLS.predict(model,Xt)
		@test isequal(round.(pred),[8; 10; 12.0])

		Xtr        = [1 2; 2 4.0; 4.0 6; 6 8]
		Ytr        = [2; 4; 6.0; 8]
		Xt         = [1 2; 2 4.0] # a subsample

		model    = PLS.fit(Xtr,Ytr,nfactors=2,centralize=true)
		pred     = PLS.predict(model,Xt)
		@test isequal(round.(pred),[2; 4])

	end

end;


@testset "PLS1 Pediction Tests (out of sample)" begin


	@testset "Linear Prediction Tests (Ax + b) | A>0" begin


		Xtr        = [1 2; 2 4; 4.0 6]
		Ytr        = [2; 4; 6.0]
		Xt         = [6 8; 8 10; 10.0 12]
		model    = PLS.fit(Xtr,Ytr,nfactors=2)
		pred     = PLS.predict(model,Xt)
		@test isequal(round.(pred),[8; 10; 12.0])


		Xtr        = [1 2; 2 4; 4.0 6]
		Ytr        = [4; 6; 8.0]
		Xt         = [6 8; 8 10; 10.0 12]
		model    = PLS.fit(Xtr,Ytr,nfactors=2)
		pred     = PLS.predict(model,Xt)
		@test isequal(round.(pred),[10; 12; 14.0])


	end

	@testset "Linear Prediction Tests (Ax + b) | A<0" begin



		Xtr        = [1 -2; 2 -4; 4.0 -6]
		Ytr        = [-2; -4; -6.0]
		Xt         = [6 -8; 8 -10; 10.0 -12]
		model    = PLS.fit(Xtr,Ytr,nfactors=2)
		pred     = PLS.predict(model,Xt)
		@test isequal(round.(pred),[-8; -10; -12.0])


		Xtr        = [1 -2; 2 -4; 4.0 -6]
		Ytr        = [-4; -6; -8.0]
		Xt         = [6 -8; 8 -10; 10.0 -12]
		model    = PLS.fit(Xtr,Ytr,nfactors=2)
		pred     = PLS.predict(model,Xt)
		@test isequal(round.(pred),[-10; -12; -14.0])

	end

end;
