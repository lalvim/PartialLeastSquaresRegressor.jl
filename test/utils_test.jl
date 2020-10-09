@testset "Auxiliary Functions Test" begin
    @testset "check constant columns" begin
		@test_throws Exception PartialLeastSquaresRegressor.check_constant_cols([1.0 1;1 2;1 3])
		@test_throws Exception PartialLeastSquaresRegressor.check_constant_cols([1.0;1;1][:,:])
		@test_throws Exception PartialLeastSquaresRegressor.check_constant_cols([1.0 2 3])
		@test_throws Exception PartialLeastSquaresRegressor.check_constant_cols([1.0; 1; 1][:,:])

		@test PartialLeastSquaresRegressor.check_constant_cols([1.0 1;2 2;3 3])
		@test PartialLeastSquaresRegressor.check_constant_cols([1.0;2;3][:,:])
	end

	@testset "checkparams" begin
	     #@test_logs PLSRegressor.check_params(2,1,"linear")
		 @test_throws Exception PartialLeastSquaresRegressor.check_params(-1,2,"linear")
		 @test_throws Exception PartialLeastSquaresRegressor.check_params(1,2,"x")

		 @test PartialLeastSquaresRegressor.check_params(1,2,"linear")
	end

	@testset "checkdata" begin
		 @test_throws Exception PartialLeastSquaresRegressor.check_data(zeros(0,0), 0)
		 @test_throws Exception PartialLeastSquaresRegressor.check_data(zeros(1,1), 10)
		 @test PartialLeastSquaresRegressor.check_data(zeros(1,1), 1)
	end
end
