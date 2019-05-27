@testset "Test Saving and Loading PLS1 Models" begin



	Xtr        = [1 -2; 2 -4; 4.0 -6]
	Ytr        = [-2; -4; -6.0]
	Xt         = [6 -8; 8 -10; 10.0 -12]
	model1    = PLSRegressor.fit(Xtr,Ytr,nfactors=2)
	pred1     = PLSRegressor.predict(model1,Xt)

	PLSRegressor.save(model1)
	model2    = PLSRegressor.load()

	pred2     = PLSRegressor.predict(model2,Xt)
    rm(PLSRegressor.MODEL_FILENAME)
	@test all(pred1 .== pred2)


end


@testset "PLS1 Pediction Tests (in sample)" begin

	@testset "Single Column Prediction Test" begin

		X        = [1; 2; 3.0][:,:]
		Y        = [1; 2; 3.0]
		model    = PLSRegressor.fit(X,Y,nfactors=1)
		pred     = PLSRegressor.predict(model,X)
		@test isequal(round.(pred),[1; 2; 3.0])

	end


	@testset "Constant Values Prediction Tests (Ax + b) | A=0, b=1 " begin

		X        = [1 3;2 1;3 2.0]
		Y        = [1; 1; 1.0]
		try
			PLSRegressor.fit(X,Y,nfactors=2)
		catch
			@test true
		end

	end

	@testset "Linear Prediction Tests " begin


		X        = [1 2; 2 4; 4.0 6]
		Y        = [2; 4; 6.0]
		model    = PLSRegressor.fit(X,Y,nfactors=2)
		pred     = PLSRegressor.predict(model,X)
		@test isequal(round.(pred),[2; 4; 6.0])

		X           = [1 -2; 2 -4; 4.0 -6]
		Y           = [-2; -4; -6.0]
		model       = PLSRegressor.fit(X,Y,nfactors=2)
		pred        = PLSRegressor.predict(model,X)
		@test isequal(round.(pred),[-2; -4; -6.0])

	end

	@testset "Linear Prediction Tests (Ax + b)" begin


		Xtr        = [1 2; 2 4; 4.0 6]
		Ytr        = [2; 4; 6.0]
		Xt         = [6 8; 8 10; 10.0 12] # same sample
		model    = PLSRegressor.fit(Xtr,Ytr,nfactors=2)
		pred     = PLSRegressor.predict(model,Xt)
		@test isequal(round.(pred),[8; 10; 12.0])

		Xtr        = [1 2; 2 4.0; 4.0 6; 6 8]
		Ytr        = [2; 4; 6.0; 8]
		Xt         = [1 2; 2 4.0] # a subsample

		model    = PLSRegressor.fit(Xtr,Ytr,nfactors=2,centralize=true)
		pred     = PLSRegressor.predict(model,Xt)
		@test isequal(round.(pred),[2; 4])

	end

end;


@testset "PLS1 Pediction Tests (out of sample)" begin


	@testset "Linear Prediction Tests (Ax + b) | A>0" begin


		Xtr        = [1 2; 2 4; 4.0 6]
		Ytr        = [2; 4; 6.0]
		Xt         = [6 8; 8 10; 10.0 12]
		model    = PLSRegressor.fit(Xtr,Ytr,nfactors=2)
		pred     = PLSRegressor.predict(model,Xt)
		@test isequal(round.(pred),[8; 10; 12.0])


		Xtr        = [1 2; 2 4; 4.0 6]
		Ytr        = [4; 6; 8.0]
		Xt         = [6 8; 8 10; 10.0 12]
		model    = PLSRegressor.fit(Xtr,Ytr,nfactors=2)
		pred     = PLSRegressor.predict(model,Xt)
		@test isequal(round.(pred),[10; 12; 14.0])


	end

	@testset "Linear Prediction Tests (Ax + b) | A<0" begin



		Xtr        = [1 -2; 2 -4; 4.0 -6]
		Ytr        = [-2; -4; -6.0]
		Xt         = [6 -8; 8 -10; 10.0 -12]
		model    = PLSRegressor.fit(Xtr,Ytr,nfactors=2)
		pred     = PLSRegressor.predict(model,Xt)
		@test isequal(round.(pred),[-8; -10; -12.0])


		Xtr        = [1 -2; 2 -4; 4.0 -6]
		Ytr        = [-4; -6; -8.0]
		Xt         = [6 -8; 8 -10; 10.0 -12]
		model    = PLSRegressor.fit(Xtr,Ytr,nfactors=2)
		pred     = PLSRegressor.predict(model,Xt)
		@test isequal(round.(pred),[-10; -12; -14.0])

	end

end;
