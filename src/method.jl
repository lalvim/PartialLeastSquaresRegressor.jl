# Partial Least Squares (PLS1 version)
include("utils.jl")

using JLD

export fit,transform,load,save


## constants
const NFACT          = 10              # default number of factors if it is not informed by the user
const MODEL_FILENAME = "pls_model.jld" # jld filename for storing the model
const MODEL_ID       = "pls_model"     # if od the model in the filesystem jld data


#### PLS type
mutable struct Model{T<:AbstractFloat}
    W::Matrix{T}        # a set of vectors representing correlation weights of input data (X) with the target (Y)
    R::Matrix{T}        # a set of projection vectors of input data (X) into w
    b::Matrix{T}        # a set of scalar values representing a latent value for dependent variables or target (Y)
    P::Matrix{T}        # a set of latent vetors for the input data (X)
    nfactors::Int       # a scalar value representing the number of latent variables
    mx::Matrix{T}       # mean stat after for z-scoring input data (X)
    my::T               # mean stat after for z-scoring target data (Y)
    sx::Matrix{T}       # standard deviation stat after z-scoring input data (X)
    sy::T               # standard deviation stat after z-scoring target data (X)
    nfeatures::Int      # number of input (X) features columns
    centralize::Bool    # store information of centralization of data. if true, tehn it is passed to transform function
end

## constructor
function Model{T<:AbstractFloat}(nrows::Int,
                                 ncols::Int,
                                 nfactors::Int,
                                 mx::Matrix{T},
                                 my::T,
                                 sx::Matrix{T},
                                 sy::T,
                                 nfeatures::Int,
                                 centralize::Bool)

    ## Allocation
    return Model(zeros(T,ncols,nfactors), ## W
        zeros(T,nrows,nfactors), ## R
        zeros(T,1,nfactors),       ## b
        zeros(T,ncols,nfactors), ## P
        nfactors,
        mx,
        my,
        sx,
        sy,
        nfeatures,
        centralize)::Model{T}
end

## Load and Store models (good for production)
function load(; filename::AbstractString = MODEL_FILENAME, modelname::AbstractString = MODEL_ID)
    local M#::Model{Float64}
    jldopen(filename, "r") do file
        M = read(file, modelname)
    end
    M
end

function save{T<:AbstractFloat}(M::Model{T}; filename::AbstractString = MODEL_FILENAME, modelname::AbstractString = MODEL_ID)
    jldopen(filename, "w") do file
        #addrequire(file, method)
        write(file, modelname, M)
    end
end


## the learning algorithm
function pls1_trainer{T<:AbstractFloat}(model::Model{T},
                                X::Matrix{T}, Y::Vector{T})
    W,R,b,P  = model.W,model.R,model.b,model.P
    nfactors = model.nfactors
    for i = 1:nfactors
        W[:,i] = X'Y
        W[:,i] /= norm(W[:,i])#sqrt.(W[:,i]'*W[:,i])
        R[:,i] = X*W[:,i]
        Rn     = R[:,i]'/(R[:,i]'R[:,i]) # change to use function...
        P[:,i] = Rn*X
        b[i]   = Rn * Y
        if b[i] == 0
           print("PLS converged. No need learning with more than $(i) factors")
           model.nfactors = i
           break
        end
        X      = X - R[:,i] * P[:,i]'
        Y      = Y - R[:,i] * b[i]
    end

    return model
end

"""
    fit(X::Matrix{:<AbstractFloat},Y::Vector{:<AbstractFloat}; nfactors::Int=10,copydata::Bool=true,centralize::Bool=true)

A Partial Least Squares learning algorithm.

# Arguments
- `nfactors::Int = 10`: The number of latent variables to explain the data.
- `copydata::Bool = true`: If you want to use the same input matrix or a copy.
- `centralize::Bool = true`: If you want to z-score columns. Recommended if not z-scored yet.
"""
function fit{T<:AbstractFloat}(X::Matrix{T}, Y::Vector{T};
                              nfactors::Int=NFACT,
                              copydata::Bool=true,
                              centralize::Bool=true)

    check_constant_cols(X)
    check_constant_cols(Y)

    check_params(nfactors, size(X,2))

    check_data(X, Y)

    Xi =  (copydata ? deepcopy(X) : X)
    Yi =  (copydata ? deepcopy(Y) : Y)

    model = Model(size(X,1),size(X,2),
                 nfactors,
                 mean(X,1),mean(Y),
                 std(X,1),std(Y),
                 size(X,2),
                 centralize)

    Xi =  (centralize ? centralize_data(Xi,model.mx,model.sx) : Xi)
    Yi =  (centralize ? centralize_data(Yi,model.my,model.sy) : Yi)
    model.centralize  = (centralize ? true: false)

    pls1_trainer(model,Xi,Yi)

    return model
end

function pls1_predictor{T<:AbstractFloat}(model::Model{T},
                                          X::DenseMatrix{T})
    W,b,P    = model.W,model.b,model.P
    nfactors = model.nfactors
    nrows    = size(X,1)
    R = zeros(T,nrows)
    Y = zeros(T,nrows)

    for i = 1:nfactors
        R      = X*W[:,i]
        Y      = Y + R*b[i]
        X      = X - R*P[:,i]'
    end

    return Y
end

"""
    transform(model::PLS.Model; X::Matrix{:<AbstractFloat}; copydata::Bool=true)

A Partial Least Squares predictor.

# Arguments
- `copydata::Bool = true`: If you want to use the same input matrix or a copy.
"""
function transform{T<:AbstractFloat}(model::Model{T},
                                    X::Matrix{T};
                                    copydata::Bool=true)
    check_data(X,model.nfeatures)

    Xi =  (copydata ? deepcopy(X) : X)
    Xi =  (model.centralize ? centralize_data(Xi,model.mx,model.sx) : Xi)

    Yi =  pls1_predictor(model,Xi)
    Yi =  decentralize_data(Yi,model.my,model.sy)

    return Yi
end
