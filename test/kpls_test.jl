@testset "KPLS Pediction Tests (in sample)" begin

    @testset "Test KPLS Single Non Linear Target" begin

        Random.seed!(1)

        z(x)     = 4.26 * (exp.(-x) - 4 * exp.(-2.0*x) + 3 * exp.(-3.0*x))
        x_values = Array(range(0.0,step=3.5,length=100))
        z_pure   = z(x_values)
        noise    = Random.randn(100)
        z_noisy  = z_pure + noise
        X        = MLJBase.table(collect(x_values)[:,:])
        Y        = z_noisy #[:,:] #z_pure


        pls_pipe = pipe(1, kernel=true)

		pls_machine    = MLJBase.machine(pls_pipe, X, Y)

        train = range(1,stop=nrows(X))
        MLJBase.fit!(pls_machine, rows=train,verbosity=0)
        yhat = MLJBase.predict(pls_machine, rows=train);
        @test MLJBase.mae(yhat, Y[train]) |> mean < 1e-2


    end

    @testset "Test KPLS Single Target (Linear Target)" begin


        X        = MLJBase.table([1 2; 2 4; 4.0 6])
        Y        = [-2; -4; -6.0] #[:,:]


        pls_pipe = pipe(1, kernel=true)

		pls_machine    = MLJBase.machine(pls_pipe, X, Y)


        train = range(1,stop=nrows(X))
        MLJBase.fit!(pls_machine, rows=train,verbosity=0)
        yhat = MLJBase.predict(pls_machine, rows=train);
        @test MLJBase.mae(yhat, Y[train]) |> mean < 1e-2

        X        = MLJBase.table([1 2; 2 4; 4.0 6])
        Y        = [2; 4; 6.0]#[:,:]


        pls_pipe = pipe(1, kernel=true)

		pls_machine    = MLJBase.machine(pls_pipe, X, Y)

        train = range(1,stop=nrows(X))
        MLJBase.fit!(pls_machine, rows=train,verbosity=0)
        yhat = MLJBase.predict(pls_machine, rows=train);
        @test MLJBase.mae(yhat, Y[train]) |> mean < 1e-2

    end

    @testset "Test KPLS Multiple Target (Linear Target)" begin


        X        = MLJBase.table([1; 2; 3.0][:,:])
        Y        = MLJBase.table([1 1; 2 2; 3 3.0]) #[:,:]

        pls_pipe = pipe(1, kernel=true)

		pls_machine    = MLJBase.machine(pls_pipe, X, Y)

        train = range(1,stop=nrows(X))
        MLJBase.fit!(pls_machine, rows=train,verbosity=0)
        yhat = MLJBase.predict(pls_machine, rows=train);
        @test abs.(MLJBase.matrix(yhat) .- MLJBase.matrix(Y)[train,:]) |> mean < 1e-6

        X        = MLJBase.table([1; 2; 3.0][:,:])
        Y        = MLJBase.table([1 -1; 2 -2; 3 -3.0])#[:,:]


        pls_pipe = pipe(1, kernel=true)

		pls_machine    = MLJBase.machine(pls_pipe, X, Y)

        train = range(1,stop=nrows(X))
        MLJBase.fit!(pls_machine, rows=train,verbosity=0)
        yhat = MLJBase.predict(pls_machine, rows=train);
        @test abs.( MLJBase.matrix(yhat) .- MLJBase.matrix(Y)[train,:]) |> mean < 1e-6

        @testset "Linear Prediction Tests " begin


        X        = MLJBase.table([1 2; 2 4; 4 6.0])
        Y        = MLJBase.table([4 2;6 4;8 6.0])#[:,:]

        pls_pipe = pipe(1, kernel=true)

		pls_machine    = MLJBase.machine(pls_pipe, X, Y)

        train = range(1,stop=nrows(X))
        MLJBase.fit!(pls_machine, rows=train,verbosity=0)
        yhat = MLJBase.predict(pls_machine, rows=train);
        @test abs.( MLJBase.matrix(yhat) .- MLJBase.matrix(Y)[train,:]) |> mean < 1e-6

        X           = MLJBase.table([1 -2; 2 -4; 4 -6.0])
        Y           = MLJBase.table([-4 -2;-6 -4;-8 -6.0])#[:,:]

        pls_pipe = pipe(1, kernel=true)

		pls_machine    = MLJBase.machine(pls_pipe, X, Y)

        train = range(1,stop=nrows(X))
        MLJBase.fit!(pls_machine, rows=train,verbosity=0)
        yhat = MLJBase.predict(pls_machine, rows=train);
        @test abs.(MLJBase.matrix(yhat) .- MLJBase.matrix(Y)[train,:]) |> mean < 1e-6


        end


    end

end;
