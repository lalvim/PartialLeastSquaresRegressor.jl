# Partial Least Squares (PLS1 version)

#### PLS type
export fit,transform

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

## constants
const NFACT = 10        # default number of factors if it is not informed by the user

## Auxiliary functions

## checks PLS input data and params
function check_data{T<:AbstractFloat}(X::Matrix{T},Y::Vector{T})
    !isempty(X) ||
        throw(DimensionMismatch("Empty input data (X)."))
    !isempty(Y) ||
        throw(DimensionMismatch("Empty target data (Y)."))
    size(X, 1) == size(Y, 1) ||
        throw(DimensionMismatch("Incompatible number of rows of input data (X) and target data (Y)."))
end

function check_data{T<:AbstractFloat}(X::Matrix{T},nplscols::Int)
    !isempty(X) ||
        throw(DimensionMismatch("Empty input data (X)."))
    size(X, 2) == nplscols ||
        throw(DimensionMismatch("Incompatible number of columns of input data (X) and original training X columns."))
end

function check_params(nfactors::Int, ncols::Int)
    nfactors >= 1 || error("nfactors must be a positive integer.")
    nfactors <= ncols || error("nfactors must be less or equal to the number of columns of input data (X).")
end

## checks constant columns
check_constant_cols{T<:AbstractFloat}(X::Matrix{T}) = size(X,1)==1 || (i=find(all(X .== X[1,:]',1))) == [] || error("You must remove constant columns $i of input data (X) before train")
check_constant_cols{T<:AbstractFloat}(Y::Vector{T}) = length(Y)==1 || length(unique(Y)) != 1 || error("Your target values are constant. All values are equal to $(Y[1])")

## Preprocessing data using z-score statistics. this is due to the fact that if X and Y are z-scored, than X'Y returns for W vector a pearson correlation for each element! :)
centralize_data{T<:AbstractFloat}(D::Matrix{T}, m::Matrix{T}, s::Matrix{T})   = (D .-m)./s
centralize_data{T<:AbstractFloat}(D::Vector{T}, m::T, s::T)                        = (D -m)/s

decentralize_data{T<:AbstractFloat}(D::Matrix{T}, m::Matrix{T}, s::Matrix{T}) = D .*s .+m
decentralize_data{T<:AbstractFloat}(D::Vector{T}, m::T, s::T)                      = D *s +m

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
