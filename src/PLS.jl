module PLS

include("method.jl")

dir(path...) = joinpath(dirname(dirname(@__FILE__)),path...)

end # module
