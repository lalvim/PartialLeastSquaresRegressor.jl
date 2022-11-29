using MLJModelInterface
using Random

const MMI = MLJModelInterface

const PLSRegressor_Desc  = "A Partial Least Squares Regressor. Contains PLS1, PLS2 (multi target) algorithms. Can be used mainly for regression."
const KPLSRegressor_Desc = "A Kernel Partial Least Squares Regressor. A Kernel PLS2 NIPALS algorithms. Can be used mainly for regression."

const MLJDICT = Dict(:Pls1 => PLS1Model,:Pls2 => PLS2Model,:Kpls => KPLSModel)

mutable struct PLSRegressor <: MMI.Deterministic
    n_factors::Int
end

mutable struct KPLSRegressor <: MMI.Deterministic
    n_factors::Integer
    kernel::String
    width::Real
end

function PLSRegressor(; n_factors=1)
    model   = PLSRegressor(n_factors)
    message = MLJModelInterface.clean!(model)
    isempty(message) || @warn message
    return model
end

function MMI.clean!(m::PLSRegressor)
    warning = ""
    if m.n_factors <= 0
        warning *= "Parameter `n_factors` expected to be positive, resetting to 1"
        m.n_factors = 1
    end
    return warning
end

function KPLSRegressor(; n_factors=1,kernel="rbf",width=1.0)
    model   = KPLSRegressor(n_factors,kernel,width)
    message = MLJModelInterface.clean!(model)
    isempty(message) || @warn message
    return model
end

function MMI.clean!(m::KPLSRegressor)
    warning = ""
    if m.n_factors <= 0
        warning *= "Parameter `n_factors` expected to be positive, resetting to 1"
        m.n_factors = 1
    end
    if m.kernel != "rbf"
        warning *= "Parameter `kernel` expected to be rbf, resetting to rbf"
        m.kernel = "rbf"
    end

    return warning
end

# Because PLSModel type and it's relatives cannot accept subarrays (views) in their
# constructors, we need to modify `MMI.matrix`:
_concretify(X::Array) = X
_concretify(X::AbstractArray) = convert(Array, X)
_matrix(X::AbstractArray) = _concretify(X)
_matrix(table) = _concretify(MMI.matrix(table))

function MMI.fit(m::PLSRegressor, verbosity, _X, _Y)

    X =  _matrix(_X)
    Y = _matrix(_Y)

    check_constant_cols(X)
    check_constant_cols(Y)
    check_params(m.n_factors, size(X,2),"linear")
    check_data(X, Y)

    model                    = PLSModel(X,Y,m.n_factors)

    (fitresult,cache,report) = fitting(model,X,Y)

    return (fitresult,cache,report)

end


function MMI.fit(m::KPLSRegressor, verbosity, _X, _Y)

    X = _matrix(_X)
    Y = _matrix(_Y)

    check_constant_cols(X)
    check_constant_cols(Y)
    check_params(m.n_factors, size(X,2),m.kernel)
    check_data(X, Y)

    model = PLSModel(X,Y,
                 m.n_factors,
                 m.kernel,
                 m.width)

    (fitresult,cache,report) = fitting(model,X,Y)

    return (fitresult,cache,report)
end

function MMI.predict(m::Union{PLSRegressor,KPLSRegressor}, fitresult, _X)

    X = _matrix(_X)
    check_data(X,fitresult.nfeatures)

    Y =  predicting(fitresult,X)
    Y = (length(size(Y)) > 1 ? MMI.table(Y) : Y)

    return Y
end

MMI.metadata_pkg.(
    (PLSRegressor, KPLSRegressor),
    name       = "PartialLeastSquaresRegressor",
    uuid       = "f4b1acfe-f311-436c-bb79-8483f53c17d5",
    url        = "https://github.com/lalvim/PartialLeastSquaresRegressor.jl",
    julia      = true,
    license    = "MIT",
    is_wrapper = false)

MMI.metadata_model(PLSRegressor,
    path = "PartialLeastSquaresRegressor.PLSRegressor",
    input   = Table(Continuous),
    target  = Union{AbstractVector{<:Continuous},Table(Continuous)},
    weights = false,
    descr   = PLSRegressor_Desc)

MMI.metadata_model(KPLSRegressor,
    path = "PartialLeastSquaresRegressor.KPLSRegressor",
    input   = Table(Continuous),
    target  = Union{AbstractVector{<:Continuous},Table(Continuous)},
    weights = false,
    descr   = KPLSRegressor_Desc)
