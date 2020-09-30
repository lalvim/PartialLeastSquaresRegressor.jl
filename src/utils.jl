## checks PLS input data and params
## checks PLS input data and params
function check_data(X::Matrix{T},Y::Union{Vector{T},Matrix{T}}) where T<:AbstractFloat
    !isempty(X) ||
        throw(DimensionMismatch("Empty input data (X)."))
    !isempty(Y) ||
        throw(DimensionMismatch("Empty target data (Y)."))
    size(X, 1) == size(Y, 1) ||
        throw(DimensionMismatch("Incompatible number of rows of input data (X) and target data (Y)."))
end

function check_data(X::Matrix{T},nfeatures::Int) where T<:AbstractFloat
    !isempty(X) ||
        throw(DimensionMismatch("Empty input data (X)."))
    size(X, 2) == nfeatures ||
        throw(DimensionMismatch("Incompatible number of columns of input data (X) and original training X columns."))
end

function check_params(nfactors::Int, ncols::Int, kernel::AbstractString)
    nfactors >= 1 || throw(ErrorException("nfactors must be a positive integer."))
    nfactors <= ncols || @warn("nfactors greater than ncols of input data (X) must generate numerical problems. However, can improve results if ok.")
    kernel == "rbf" || kernel == "linear" || throw(ErrorException("kernel must be kernel='linear' or 'kernel=rbf'"))
end

## checks constant columns
check_constant_cols(X::Matrix{T}) where {T<:AbstractFloat} = size(X,1)>1 && !any(all(X .== X[1,:]',dims=1)) || throw(ErrorException("You must remove constant columns of input data (X) before train"))
check_constant_cols(Y::Vector{T}) where {T<:AbstractFloat} = length(Y)>1 && length(unique(Y)) > 1 || throw(ErrorException("Your target values are constant. All values are equal to $(Y[1])"))
