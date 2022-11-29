@testset "generic MLJ interface tests" begin
    regressors = [
        PartialLeastSquaresRegressor.PLSRegressor,
        PartialLeastSquaresRegressor.KPLSRegressor,
    ]

    @testset "univariate target" begin
        failures, summary = MLJTestInterface.test(
            regressors,
            MLJTestInterface.make_regression()...;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=false, # set to true to debug
        )
        @test isempty(failures)
    end
    @testset "multivariate target" begin
        failures, summary = MLJTestInterface.test(
            regressors,
            MLJBase.make_regression(n_targets=3)...;
            mod=@__MODULE__,
            verbosity=0, # bump to debug
            throw=false, # set to true to debug
        )
        @test isempty(failures)
    end
end
