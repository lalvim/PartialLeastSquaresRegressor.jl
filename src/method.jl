# Partial Least Squares (PLS1 version)
include("utils.jl")

using JLD

## constants
const NFACT          = 10              # default number of factors if it is not informed by the user
const MODEL_FILENAME = "pls_model.jld" # jld filename for storing the model
const MODEL_ID       = "pls_model"     # if od the model in the filesystem jld data


#### PLS1 type
mutable struct PLS1Model{T<:AbstractFloat}
    W::Matrix{T}           # a set of vectors representing correlation weights of input data (X) with the target (Y)
    b::Matrix{T}           # a set of scalar values representing a latent value for dependent variables or target (Y)
    P::Matrix{T}           # a set of latent vetors for the input data (X)
    nfactors::Int          # a scalar value representing the number of latent variables
    mx::Matrix{T}          # mean stat after for z-scoring input data (X)
    my::T                  # mean stat after for z-scoring target data (Y)
    sx::Matrix{T}          # standard deviation stat after z-scoring input data (X)
    sy::T                  # standard deviation stat after z-scoring target data (X)
    nfeatures::Int         # number of input (X) features columns
    centralize::Bool       # store information of centralization of data. if true, tehn it is passed to transform function
end

#### PLS2 type
mutable struct PLS2Model{T<:AbstractFloat}
    W::Matrix{T}           # a set of vectors representing correlation weights of input data (X) with the target (Y)
    Q::Matrix{T}           #
    b::Matrix{T}           # a set of scalar values representing a latent value for dependent variables or target (Y)
    P::Matrix{T}           # a set of latent vetors for the input data (X)
    nfactors::Int          # a scalar value representing the number of latent variables
    mx::Matrix{T}          # mean stat after for z-scoring input data (X)
    my::Matrix{T}          # mean stat after for z-scoring target data (Y)
    sx::Matrix{T}          # standard deviation stat after z-scoring input data (X)
    sy::Matrix{T}          # standard deviation stat after z-scoring target data (X)
    nfeatures::Int         # number of input (X) features columns
    centralize::Bool       # store information of centralization of data. if true, tehn it is passed to transform function
end

## PLS1: constructor
function Model{T<:AbstractFloat}(X::Matrix{T},
                                 Y::Vector{T},
                                 nfactors::Int,
                                 centralize::Bool)
    (nrows,ncols) = size(X)
    ## Allocation
    return PLS1Model(zeros(T,ncols,nfactors), ## W
            zeros(T,1,nfactors),       ## b
            zeros(T,ncols,nfactors), ## P
            nfactors,
            mean(X,1),
            mean(Y),
            std(X,1),
            std(Y),
            ncols,
            centralize)
end

## PLS2: constructor
function Model{T<:AbstractFloat}(X::Matrix{T},
                                 Y::Matrix{T}, # this is the diference from PLS1 param constructor!
                                 nfactors::Int,
                                 centralize::Bool)
    (nrows,ncols) = size(X)
    (n,m)         = size(Y)
    ## Allocation
    return PLS2Model(zeros(T,ncols,nfactors), ## W
            zeros(T,m,nfactors),       ## Q
            zeros(T,n,nfactors),       ## b
            zeros(T,ncols,nfactors),   ## P
            nfactors,
            mean(X,1),
            mean(Y,1),
            std(X,1),
            std(Y,1),
            ncols,
            centralize)
end



## the learning algorithm: PLS1 - single target
function trainer{T<:AbstractFloat}(model::PLS1Model{T},
                                   X::Matrix{T}, Y::Vector{T})
    W,b,P  = model.W,model.b,model.P
    nfactors = model.nfactors
    for i = 1:nfactors
        W[:,i] = X'Y
        W[:,i] /= norm(W[:,i])#sqrt.(W[:,i]'*W[:,i])
        R      = X*W[:,i]
        Rn     = R'/(R'R) # change to use function...
        P[:,i] = Rn*X
        b[i]   = Rn * Y
        if b[i] == 0
           print("PLS1 converged. No need learning with more than $(i) factors")
           model.nfactors = i
           break
        end
        X      = X - R * P[:,i]'
        Y      = Y - R * b[i]
    end

    return model
end


## the learning algorithm: PLS2 - multiple targets
function trainer{T<:AbstractFloat}(model::PLS2Model{T},
                                   X::Matrix{T}, Y::Matrix{T})
    W,b,P,Q  = model.W,model.b,model.P,model.Q
    nfactors = model.nfactors

    for i = 1:nfactors
        b[:,i]    = Y[:,1] #u: arbitrary col. Thus, I set to the first.
        Rold = b[:,i]
        local R::Vector{T}
        while true

            W[:,i]  = X'b[:,i]
            W[:,i] /= norm(W[:,i])#sqrt.(W[:,i]'*W[:,i])
            R       = X*W[:,i]

            Rold      = R
            Q[:,i]    = Y'R
            Q[:,i]   /= norm(Q[:,i])
            b[:,i]    = Y*Q[:,i]
            if  all(abs.(R - Rold) .<= 1.0e-3)
                break
            end

        end
        Rn     = R'/(R'R) # change to use function...
        P[:,i] = Rn*X

        X      = X - R * P[:,i]'
        c      = Rn'b[:,i:i]'./(R'R)
        Yp     = c*R*Q[:,i]'
        Y      = Y - Yp
        if  all(abs.(Y - Yp) .<= 1.0e-3)
            print("PLS2 converged. No need learning with more than $(i) factors")
            model.nfactors = i
            break
        end

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
function fit{T<:AbstractFloat}(X::Matrix{T}, Y::Union{Vector{T},Matrix{T}};
                              nfactors::Int=NFACT,
                              copydata::Bool=true,
                              centralize::Bool=true)

    check_constant_cols(X)
    check_constant_cols(Y)

    check_params(nfactors, size(X,2))

    check_data(X, Y)

    Xi =  (copydata ? deepcopy(X) : X)
    Yi =  (copydata ? deepcopy(Y) : Y)

    model = Model(Xi,Yi,
                 nfactors,
                 centralize)

    Xi =  (centralize ? centralize_data(Xi,model.mx,model.sx) : Xi)
    Yi =  (centralize ? centralize_data(Yi,model.my,model.sy) : Yi)
    model.centralize  = (centralize ? true: false)

    trainer(model,Xi,Yi)

    return model
end

function predictor{T<:AbstractFloat}(model::PLS1Model{T},
                                     X::DenseMatrix{T})

    W,b,P    = model.W,model.b,model.P
    nfactors = model.nfactors
    nrows    = size(X,1)
    Y = zeros(T,nrows)

    for i = 1:nfactors
        R      = X*W[:,i]
        Y      = Y + R*b[i]
        X      = X - R*P[:,i]'
    end

    return Y
end

function predictor{T<:AbstractFloat}(model::PLS2Model{T},
                                     X::DenseMatrix{T})

    W,Q,b,P    = model.W,model.Q,model.b,model.P
    nfactors   = model.nfactors
    nrows      = size(X,1)
    Y = zeros(T,nrows)

    for i = 1:nfactors

        R      = X*W[:,i]
        X      = X - R * P[:,i]'
        #c      = Rn'b[:,i]/(R'R)
        c      = R'b[:,i]
        Y      = Y + c*R * Q'

    end

    return Y
end


"""
    transform(model::PLS.Model; X::Matrix{:<AbstractFloat}; copydata::Bool=true)

A Partial Least Squares predictor.

# Arguments
- `copydata::Bool = true`: If you want to use the same input matrix or a copy.
"""
function transform{T<:AbstractFloat}(model::Union{PLS1Model{T},PLS2Model{T}},
                                    X::Matrix{T};
                                    copydata::Bool=true)
    check_data(X,model.nfeatures)

    Xi =  (copydata ? deepcopy(X) : X)
    Xi =  (model.centralize ? centralize_data(Xi,model.mx,model.sx) : Xi)

    Yi =  predictor(model,Xi)
    Yi =  decentralize_data(Yi,model.my,model.sy)

    return Yi
end



## Load and Store models (good for production)
function load(; filename::AbstractString = MODEL_FILENAME, modelname::AbstractString = MODEL_ID)
    local M
    jldopen(filename, "r") do file
        M = read(file, modelname)
    end
    M
end

function save{T<:AbstractFloat}(M::Union{PLS1Model{T},PLS2Model{T}}; filename::AbstractString = MODEL_FILENAME, modelname::AbstractString = MODEL_ID)
    jldopen(filename, "w") do file
        write(file, modelname, M)
    end
end
