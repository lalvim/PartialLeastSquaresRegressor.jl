## constants
const NFACT          = 10              # default number of factors if it is not informed by the user



"""
    fit(X::Matrix{:<AbstractFloat},Y::Vector{:<AbstractFloat}; nfactors::Int=10,copydata::Bool=true,centralize::Bool=true,kernel="",width=1.0)

A Partial Least Squares learning algorithm.

# Arguments
- `nfactors::Int = 10`: The number of latent variables to explain the data.
- `copydata::Bool = true`: If you want to use the same input matrix or a copy.
- `centralize::Bool = true`: If you want to z-score columns. Recommended if not z-scored yet.
- `kernel::AbstractString = "gaussian"`: If you want to apply a nonlinear PLS with gaussian Kernel.
- `width::AbstractFloat = 1.0`: Gaussian Kernel width (Only if kernel="gaussian").
"""
function fit(X::AbstractArray{T},
             Y::AbstractArray{T};
             nfactors::Int          = NFACT,
             copydata::Bool         = true,
             centralize::Bool       = true,
             kernel                 = "linear",
             width                  = 1.0) where T<:AbstractFloat
    X = X[:,:]
    check_constant_cols(X)
    check_constant_cols(Y)

    check_params(nfactors, size(X,2),kernel)

    check_data(X, Y)

    Xi =  (copydata ? deepcopy(X) : X)
    Yi =  (copydata ? deepcopy(Y) : Y)
    if kernel == "rbf"
       model = Model(Xi,Yi,
                 nfactors,
                 centralize,
                 kernel,
                 width)
    else
       model = Model(Xi,Yi,
                 nfactors,
                 centralize)
    end

    Xi =  (centralize ? centralize_data(Xi,model.mx,model.sx) : Xi)
    Yi =  (centralize ? centralize_data(Yi,model.my,model.sy) : Yi)
    model.centralize  = (centralize ? true : false)

    trainer(model,Xi,Yi)

    return model
end




"""
    transform(model::PLSRegressor.Model; X::Matrix{:<AbstractFloat}; copydata::Bool=true)

A Partial Least Squares predictor.

# Arguments
- `copydata::Bool = true`: If you want to use the same input matrix or a copy.
"""
function predict(model::PLSModel{T},
                X::AbstractArray{T};
                copydata::Bool=true) where T<:AbstractFloat

    X = X[:,:]
    check_data(X,model.nfeatures)

    Xi =  (copydata ? deepcopy(X) : X)
    Xi =  (model.centralize ? centralize_data(Xi,model.mx,model.sx) : Xi)

    Yi =  predictor(model,Xi)
    Yi =  decentralize_data(Yi,model.my,model.sy)

    return Yi
end
