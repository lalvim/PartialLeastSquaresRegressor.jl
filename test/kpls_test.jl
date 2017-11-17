

@testset "KPLS Pediction Tests (in sample)" begin

    @testset "Test KPLS Single Non Linear Target" begin

        srand(1)

        z(x)     = 4.26 * (exp.(-x) - 4 * exp.(-2.0*x) + 3 * exp.(-3.0*x))
        x_values = linspace(0.0,3.5,100)
        z_pure   = z(x_values)
        noise    = randn(100)
        z_noisy  = z_pure + noise
        X        = collect(x_values)
        Y        = z_noisy #z_pure
        model    = PLS.fit(X,Y,nfactors=1,kernel="rbf",width=0.01)
    	Y_pred   = PLS.predict(model,X)
        @test mean(abs.(Y .- Y_pred)) < 1e-2

    end

    @testset "Test KPLS Single Target (Linear Target)" begin


        X        = [1 2; 2 4; 4.0 6]
        Y        = [-2; -4; -6.0]
        model    = PLS.fit(X,Y,nfactors=1,kernel="rbf",width=0.01)
    	Y_pred     = PLS.predict(model,X)
        @test mean(abs.(Y .- Y_pred)) < 1e-6

        X        = [1 2; 2 4; 4.0 6]
        Y        = [2; 4; 6.0]
        model    = PLS.fit(X,Y,nfactors=1,kernel="rbf",width=0.01)
    	Y_pred     = PLS.predict(model,X)
        @test mean(abs.(Y .- Y_pred)) < 1e-6

    end

    @testset "Test KPLS Multiple Target (Linear Target)" begin


        X        = [1; 2; 3.0]
        Y        = [1 1; 2 2; 3 3.0]
        model    = PLS.fit(X,Y,nfactors=1,kernel="rbf",width=0.01)
    	Y_pred     = PLS.predict(model,X)
        @test mean(abs.(Y .- Y_pred)) < 1e-6

        X        = [1; 2; 3.0]
        Y        = [1 -1; 2 -2; 3 -3.0]
        model    = PLS.fit(X,Y,nfactors=1,kernel="rbf",width=0.01)
    	Y_pred     = PLS.predict(model,X)
        @test mean(abs.(Y .- Y_pred)) < 1e-6

        @testset "Linear Prediction Tests " begin


        X        = [1 2; 2 4; 4 6.0]
        Y        = [4 2;6 4;8 6.0]
        model    = PLS.fit(X,Y,nfactors=1,kernel="rbf",width=0.01)
    	Y_pred     = PLS.predict(model,X)
        @test mean(abs.(Y .- Y_pred)) < 1e-6

        X           = [1 -2; 2 -4; 4 -6.0]
        Y           = [-4 -2;-6 -4;-8 -6.0]
        model    = PLS.fit(X,Y,nfactors=1,kernel="rbf",width=0.01)
    	Y_pred     = PLS.predict(model,X)
        @test mean(abs.(Y .- Y_pred)) < 1e-6


        end


    end

end;

### not saving yeat.

@testset "Test Saving and Loading KPLS Models" begin


	Xtr        = [1 -2; 2 -4; 4.0 -6]
	Ytr        = [-2; -4; -6.0]
	Xt         = [6 -8; 8 -10; 10.0 -12]
	model1    = PLS.fit(Xtr,Ytr,nfactors=1,kernel="rbf",width=0.01)
	pred1     = PLS.predict(model1,Xt)

	PLS.save(model1)
	model2    = PLS.load()

	pred2     = PLS.predict(model2,Xt)
    rm(PLS.MODEL_FILENAME)
	@test all(pred1 .== pred2)


end;
