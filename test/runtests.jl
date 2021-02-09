using PartialLeastSquaresRegressor
using Test
using MLJBase
using Statistics
using Random

# load some locally defined models needed for testing:
include("./_standardizer.jl")
import .Stand

include("./utils_test.jl")
include("./pls1_test.jl")
include("./pls2_test.jl")
include("./kpls_test.jl")
