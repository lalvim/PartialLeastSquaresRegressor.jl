# Partial Least Squares (PLS1 and PLS2 NIPALS version)
module PartialLeastSquaresRegressor

using LinearAlgebra
using Statistics

include("utils.jl")
include("types.jl")
include("pls1.jl")
include("pls2.jl")
include("kpls.jl")
include("method.jl")
include("mlj_interface.jl")

dir(path...) = joinpath(dirname(dirname(@__FILE__)),path...)

end
