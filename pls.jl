# Partial Least Squares (PLS1 version)

#### PLS type

mutable struct PLS{T<:AbstractFloat}
    W::Matrix{T}        # a set of vectors representing correlation weights of input data (X) with the target (Y)
    R::Matrix{T}        # a set of projection vectors of input data (X) into w
    b::Matrix{T}        # a set of scalar values representing a latent value for dependent variables or target (Y)
    P::Matrix{T}        # a set of latent vetors for the input data (X)
    nfactors::Int     # a scalar value representing the number of latent variables
    mx::Matrix{T}       # mean stat after for z-scoring input data (X)
    my::T               # mean stat after for z-scoring target data (Y)
    sx::Matrix{T}       # standard deviation stat after z-scoring input data (X)
    sy::T               # standard deviation stat after z-scoring target data (X)
    nfeatures::Int      # number of input (X) features columns
end

## constructor

function PLS{T<:AbstractFloat}(nrows::Int, ncols::Int,
                               nfactors::Int,
                               mx::Matrix{T},my::T,
                               sx::Matrix{T},sy::T,
                               nfeatures::Int)

    ## Allocation
    PLS(zeros(T,ncols,nfactors), ## W
        zeros(T,nrows,nfactors), ## R
        zeros(T,nfactors),       ## b
        zeros(T,ncols,nfactors), ## P
        nfactors,mx,my,sx,sy,
        nfeatures)
end

## constants
const NFACT = 10        # default number of factors if it is not informed by the user

## Auxiliary functions

## checks PLS input data and params
function check_plsdata{T<:AbstractFloat}(X::Matrix{T},Y::Vector{T})
    !isempty(X) ||
        throw(DimensionMismatch("Empty input data (X)."))
    !isempty(Y) ||
        throw(DimensionMismatch("Empty target data (Y)."))
    size(Y,2) == 1 ||
        throw(DimensionMismatch("target data (Y) must be equal to 1."))
    size(X, 1) == size(Y, 1) ||
        throw(DimensionMismatch("Incompatible number of rows of input data (X) and target data (Y)."))
end

function check_plsdata{T<:AbstractFloat}(X::Matrix{T},nplscols::Int)
    !isempty(X) ||
        throw(DimensionMismatch("Empty input data (X)."))
    size(X, 2) == nplscols ||
        throw(DimensionMismatch("Incompatible number of columns of input data (X) and original training X columns."))
end

function check_plsparams(nfactors::Int, ncols::Int)
    nfactors >= 1 || error("nfactors must be a positive integer.")
    nfactors <= ncols || error("nfactors must be less or equal to the number of columns of input data (X).")
end

## Preprocessing data using z-score statistics. this is due to the fact that if X and Y are z-scored, than X'Y returns for W vector a pearson correlation for each element! :)
centralize_data{T<:AbstractFloat}(D::DenseMatrix{T}, m::Vector{T}, s::Vector{T})   = (D .-m)./s
centralize_data{T<:AbstractFloat}(D::Vector{T}, m::T, s::T)                        = (D .-m)./s

decentralize_data{T<:AbstractFloat}(D::DenseMatrix{T}, m::Vector{T}, s::Vector{T}) = D .*s .+m
decentralize_data{T<:AbstractFloat}(D::Vector{T}, m::T, s::T)                      = D .*s .+m

## the learning algorithm
function pls1_trainer{T<:AbstractFloat}(pls::Type{PLS},
                                X::DenseMatrix{T}, Y::Vector{T})

    W,R,b,P  = pls.W,pls.R,pls.b,pls.P
    nfactors = pls.nfactors
    for i = 1:nfactors
        W[:,i] = X'Y
        W[:,i] /= norm(W[:,i])
        R[:,i] = X*W[:,i]
        Rn     = R[:,i]/norm(T[:,i])
        b[i]   = Rn * Y
        P[:,i] = Rn * X
        X      = X - Rn*P[:,i]
        Y      = Y - Rn.*b[i]
    end

    return pls

end

## the learning algorithm
function pls1_predictor{T<:AbstractFloat}(pls::Type{PLS},
                                          X::DenseMatrix{T})

    W,b,P  = pls.W,pls.b,pls.P
    nfactors = pls.nfactors

    R = zeros(T,nrows,nfactors)
    Y = zeros(T,nrows,nfactors)

    for i = 1:nfactors
        R[:,i] = X'W[:,i]
        Y      = Y + R.*b[i]
        X      = X - R*P[:,i]

    end

    return Y

end


## this function checks for validity of data and calls pls1 regressor
function fit{T<:AbstractFloat}(X::DenseMatrix{T}, Y::Vector{T}; nfactors::Int=NFACT, copydata::Bool=true)

    check_plsparams(nfactors, size(X,2))

    check_plsdata(X, Y)

    Xi =  (copydata ? deepcopy(X) : X)
    Yi =  (copydata ? deepcopy(Y) : Y)

    pls = PLS(size(X,1),size(X,2),
                 nfactors,
                 mean(X,2),mean(Y),
                 std(X,2),std(Y),
                 size(X,2))

    Xi =  centralize_data(Xi,pls.mx,pls.sx)
    Yi =  centralize_data(Yi,pls.my,pls.sy)

    pls1_trainer(pls,Xi,Yi)

    return pls::PLS

end


## this function checks for validity of data and calls pls1 regressor
function transform{T<:AbstractFloat}(pls::Type{PLS}, X::DenseMatrix{T})


    check_plsdata(X,pls.nfeatures)

    Xi =  (copydata ? deepcopy(X) : X)

    Xi =  centralize_data(Xi,pls.mx,pls.sx)

    return pls1_predictor(pls,Xi,Yi)::Vector{AbstractFloat}

end


X        = [1 2; 2 4.5; 4.7 9.3]
Y        = [2.1; 4.6; 9.4]

#fit(X, Y, nfactors=1,copydata=true)
fit(X, Y, nfactors=1)

#, 10, copydata=true)
