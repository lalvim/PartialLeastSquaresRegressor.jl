@testset "PLS2 Prediction Tests (in sample)" begin

	@testset "Single Column Prediction Test" begin

		X        = MLJBase.table([1; 2; 3.0][:,:])
		Y        = MLJBase.table([1 1; 2 2; 3.0 3.0])

        pls_pipe  = pipe(1)

		pls_machine    = MLJBase.machine(pls_pipe, X, Y)

		train = range(1,stop=nrows(X))
		MLJBase.fit!(pls_machine, rows=train,verbosity=0)
		pred = MLJBase.predict(pls_machine, rows=train);
		pred = MLJBase.matrix(pred)

		@test isequal(round.(pred),[1 1; 2 2; 3 3.0])

	end

	@testset "Linear Prediction Tests " begin


		X        = MLJBase.table([1 2; 2 4; 4 6.0])
		Y        = MLJBase.table([4 2;6 4;8 6.0])

        pls_pipe = pipe(2)

		pls_machine    = MLJBase.machine(pls_pipe, X, Y)

        train = range(1,stop=nrows(X))
        MLJBase.fit!(pls_machine, rows=train,verbosity=0)
        pred = MLJBase.predict(pls_machine, rows=train);
		pred = MLJBase.matrix(pred)

		@test isequal(round.(pred),[4 2;6 4;8 6.0])

		X           = MLJBase.table([1 -2; 2 -4; 4 -6.0])
		Y           = MLJBase.table([-4 -2;-6 -4;-8 -6.0])

        pls_pipe = pipe(2)

		pls_machine    = MLJBase.machine(pls_pipe, X, Y)

        train = range(1,stop=nrows(X))
        MLJBase.fit!(pls_machine, rows=train,verbosity=0)
        pred = MLJBase.predict(pls_machine, rows=train);
		pred = MLJBase.matrix(pred)

		@test isequal(round.(pred),[-4 -2;-6 -4;-8 -6.0])


	end


end

@testset "PLS2 Pediction Tests (out of sample)" begin


	@testset "Linear Prediction Tests (Ax + b) | A>0" begin


		X        = MLJBase.table([1 2;2 4;3 6;6 12;7 14.0;4 8;5 10.0])
		Y        = MLJBase.table([2 2;4 4;6 6;12 12;14 14.0;8 8;10 10.0])

        pls_pipe = pipe(2)

		pls_machine    = MLJBase.machine(pls_pipe, X, Y)

		train = range(1,stop=5)
		test  = range(6,stop=7)


        MLJBase.fit!(pls_machine, rows=train,verbosity=0)
        pred = MLJBase.predict(pls_machine, rows=test);
		pred = MLJBase.matrix(pred)

		@test isequal(round.(pred),[8 8;10 10.0])


		X        = MLJBase.table([1 2;2 4;3 6;6 12;7 14.0; 4 8;5 10.0])
		Y        = MLJBase.table([2 4;4 6;6 8;12 14;14 16.0; 8 10;10 12.0])


    	pls_pipe = pipe(2)

		pls_machine    = MLJBase.machine(pls_pipe, X, Y)

		train = range(1,stop=5)
		test  = range(6,stop=7)

        MLJBase.fit!(pls_machine, rows=train,verbosity=0)
        pred = MLJBase.predict(pls_machine, rows=test);
		pred = MLJBase.matrix(pred)

		@test isequal(round.(pred),[8 10;10 12.0])


	end

	@testset "Linear Prediction Tests (Ax + b) | A<0" begin


		X        = MLJBase.table([1 -2;2 -4;3 -6;6 -12;7 -14.0; 4 -8;5 -10.0])
		Y        = MLJBase.table([2 -2;4 -4;6 -6;12 -12;14 -14.0; 8 -8;10 -10.0])

        pls_pipe = pipe(2)

		pls_machine    = MLJBase.machine(pls_pipe, X, Y)

		train = range(1,stop=5)
		test  = range(6,stop=7)

        MLJBase.fit!(pls_machine, rows=train,verbosity=0)
        pred = MLJBase.predict(pls_machine, rows=test);
		pred = MLJBase.matrix(pred)

	    @test isequal(round.(pred),[8 -8;10 -10.0])


		X        = MLJBase.table([1 -2;2 -4;3 -6;6 -12;7 -14.0; 4 -8;5 -10.0])
		Y        = MLJBase.table([2 -4;4 -6;6 -8;12 -14;14 -16.0; 8 -10;10 -12.0])

        pls_pipe = pipe(2)

		pls_machine    = MLJBase.machine(pls_pipe, X, Y)

		train = range(1,stop=5)
		test  = range(6,stop=7)

        MLJBase.fit!(pls_machine, rows=train,verbosity=0)
        pred = MLJBase.predict(pls_machine, rows=test);
		pred = MLJBase.matrix(pred)

		@test isequal(round.(pred),[8 -10;10 -12.0])


	end


end;
