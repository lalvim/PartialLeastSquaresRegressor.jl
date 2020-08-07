@testset "PLS2 Prediction Tests (in sample)" begin

	@testset "Single Column Prediction Test" begin

		X        = MLJ.table([1; 2; 3.0][:,:])
		Y        = MLJ.table([1 1; 2 2; 3.0 3.0])

        pls_pipe       = MLJ.@pipeline prediction_type=:deterministic MLJ.Standardizer() PLSRegressor.PLS(n_factors=1)  target = MLJ.Standardizer()


		pls_machine    = MLJ.machine(pls_pipe, X, Y)

		train = range(1,stop=length(X))
		MLJ.fit!(pls_machine, rows=train,force=true)
		pred = MLJ.predict(pls_machine, rows=train);
		pred = MLJ.matrix(pred)

		@test isequal(round.(pred),[1 1; 2 2; 3 3.0])

	end

	@testset "Linear Prediction Tests " begin


		X        = MLJ.table([1 2; 2 4; 4 6.0])
		Y        = MLJ.table([4 2;6 4;8 6.0])

        pls_pipe       = MLJ.@pipeline prediction_type=:deterministic MLJ.Standardizer() PLSRegressor.PLS(n_factors=2)  target = MLJ.Standardizer()

		pls_machine    = MLJ.machine(pls_pipe, X, Y)

        train = range(1,stop=length(X))
        MLJ.fit!(pls_machine, rows=train,force=true)
        pred = MLJ.predict(pls_machine, rows=train);
		pred = MLJ.matrix(pred)

		@test isequal(round.(pred),[4 2;6 4;8 6.0])

		X           = MLJ.table([1 -2; 2 -4; 4 -6.0])
		Y           = MLJ.table([-4 -2;-6 -4;-8 -6.0])

        pls_pipe       = MLJ.@pipeline prediction_type=:deterministic MLJ.Standardizer() PLSRegressor.PLS(n_factors=2)  target = MLJ.Standardizer()

		pls_machine    = MLJ.machine(pls_pipe, X, Y)

        train = range(1,stop=length(X))
        MLJ.fit!(pls_machine, rows=train,force=true)
        pred = MLJ.predict(pls_machine, rows=train);
		pred = MLJ.matrix(pred)

		@test isequal(round.(pred),[-4 -2;-6 -4;-8 -6.0])


	end


end

@testset "PLS2 Pediction Tests (out of sample)" begin


	@testset "Linear Prediction Tests (Ax + b) | A>0" begin


		X        = MLJ.table([1 2;2 4;3 6;6 12;7 14.0;4 8;5 10.0])
		Y        = MLJ.table([2 2;4 4;6 6;12 12;14 14.0;8 8;10 10.0])

        pls_pipe       = MLJ.@pipeline prediction_type=:deterministic MLJ.Standardizer() PLSRegressor.PLS(n_factors=2)  target = MLJ.Standardizer()

		pls_machine    = MLJ.machine(pls_pipe, X, Y)

		train = range(1,stop=5)
		test  = range(6,stop=7)


        MLJ.fit!(pls_machine, rows=train,force=true)
        pred = MLJ.predict(pls_machine, rows=test);
		pred = MLJ.matrix(pred)

		@test isequal(round.(pred),[8 8;10 10.0])


		X        = MLJ.table([1 2;2 4;3 6;6 12;7 14.0; 4 8;5 10.0])
		Y        = MLJ.table([2 4;4 6;6 8;12 14;14 16.0; 8 10;10 12.0])


    	pls_pipe       = MLJ.@pipeline prediction_type=:deterministic MLJ.Standardizer() PLSRegressor.PLS(n_factors=2)  target = MLJ.Standardizer()

		pls_machine    = MLJ.machine(pls_pipe, X, Y)

		train = range(1,stop=5)
		test  = range(6,stop=7)

        MLJ.fit!(pls_machine, rows=train,force=true)
        pred = MLJ.predict(pls_machine, rows=test);
		pred = MLJ.matrix(pred)

		@test isequal(round.(pred),[8 10;10 12.0])


	end

	@testset "Linear Prediction Tests (Ax + b) | A<0" begin


		X        = MLJ.table([1 -2;2 -4;3 -6;6 -12;7 -14.0; 4 -8;5 -10.0])
		Y        = MLJ.table([2 -2;4 -4;6 -6;12 -12;14 -14.0; 8 -8;10 -10.0])

        pls_pipe       = MLJ.@pipeline prediction_type=:deterministic MLJ.Standardizer() PLSRegressor.PLS(n_factors=2)  target = MLJ.Standardizer()

		pls_machine    = MLJ.machine(pls_pipe, X, Y)

		train = range(1,stop=5)
		test  = range(6,stop=7)

        MLJ.fit!(pls_machine, rows=train,force=true)
        pred = MLJ.predict(pls_machine, rows=test);
		pred = MLJ.matrix(pred)

	    @test isequal(round.(pred),[8 -8;10 -10.0])


		X        = MLJ.table([1 -2;2 -4;3 -6;6 -12;7 -14.0; 4 -8;5 -10.0])
		Y        = MLJ.table([2 -4;4 -6;6 -8;12 -14;14 -16.0; 8 -10;10 -12.0])

        pls_pipe       = MLJ.@pipeline prediction_type=:deterministic MLJ.Standardizer() PLSRegressor.PLS(n_factors=2)  target = MLJ.Standardizer()

		pls_machine    = MLJ.machine(pls_pipe, X, Y)

		train = range(1,stop=5)
		test  = range(6,stop=7)

        MLJ.fit!(pls_machine, rows=train,force=true)
        pred = MLJ.predict(pls_machine, rows=test);
		pred = MLJ.matrix(pred)

		@test isequal(round.(pred),[8 -10;10 -12.0])


	end


end;
