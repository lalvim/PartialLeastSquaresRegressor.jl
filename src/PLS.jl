module PLS

#include("utils.jl")
#include("types.jl")

include("method.jl")

dir(path...) = joinpath(dirname(dirname(@__FILE__)),path...)

end # module

#module PLSTypes

#include("types.jl")

#dir(path...) = joinpath(dirname(dirname(@__FILE__)),path...)

#end # module
