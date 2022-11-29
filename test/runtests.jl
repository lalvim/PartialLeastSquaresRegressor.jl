using PartialLeastSquaresRegressor
using Test
using MLJBase
using MLJModels
using Statistics
using Random

# helper to wrap a partial least square regressor in standardization of inputs and target
function pipe(n_factors; kernel=false)
    atom  = kernel ?
        PartialLeastSquaresRegressor.KPLSRegressor(; n_factors, kernel="rbf", width=0.01) :
        PartialLeastSquaresRegressor.PLSRegressor(; n_factors)
    pipe = MLJBase.Pipeline(
        Standardizer(),
        atom,
        prediction_type=:deterministic,
    )
    return MLJBase.TransformedTargetModel(
        pipe;
        transformer=Standardizer(),
    )
end

include("./utils_test.jl")
include("./pls1_test.jl")
include("./pls2_test.jl")
include("./kpls_test.jl")
