

## Auxiliary functions
# A gaussian kernel function
@inline function Φ(x::Vector{Float64},
                   y::Vector{Float64},
                   r::Float64=1.0)
    norm  = 1.0 / sqrt(2π*r)
    scale = 1.0 / (2r^2)
    return norm * exp(-scale * sum((x.-y).^2))
end
# A kernel matrix
function ΦΦ(X::Matrix{Float64})
    n = size(X,1)
    K = zeros(n,n)
    for i=1:n
        for j=1:i
            K[i, j] = Φ(X[i, :], X[j, :])
            K[j, i] = K[i, j]
        end
        K[i, i] = Φ(X[i, :], X[i, :])
    end
    K
end

#gaussian_kernel([1.0; 2 ; 20],[1.0; 8; 3])

## checks PLS input data and params
function check_data{T<:AbstractFloat}(X::Matrix{T},Y::Union{Vector{T},Matrix{T}})
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
centralize_data{T<:AbstractFloat}(D::Vector{T}, m::T, s::T)                   = (D -m)/s

decentralize_data{T<:AbstractFloat}(D::Matrix{T}, m::Matrix{T}, s::Matrix{T}) = D .*s .+m
decentralize_data{T<:AbstractFloat}(D::Vector{T}, m::T, s::T)                 = D *s +m
