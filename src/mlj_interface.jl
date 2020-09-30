using MLJModelInterface
using Random

const MMI = MLJModelInterface

const PLSRegressor_Desc  = "A Partial Least Squares Regressor. Contains PLS1, PLS2 (multi target) algorithms. Can be used mainly for regression."
const KPLSRegressor_Desc = "A Kernel Partial Least Squares Regressor. A Kernel PLS2 NIPALS algorithms. Can be used mainly for regression."

const MLJDICT = Dict(:Pls1 => PLS1Model,:Pls2 => PLS2Model,:Kpls => KPLSModel)

mutable struct PLS <: MMI.Deterministic
    n_factors::Int
end

mutable struct KPLS <: MMI.Deterministic
    n_factors::Integer
    kernel::String
    width::Real
end

function PLS(; n_factors=1)
    model   = PLS(n_factors)
    message = MLJModelInterface.clean!(model)
    isempty(message) || @warn message
    return model
end

function MMI.clean!(m::PLS)
    warning = ""
    if m.n_factors <= 0
        warning *= "Parameter `n_factors` expected to be positive, resetting to 1"
        m.n_factors = 1
    end
    return warning
end

function KPLS(; n_factors=1,kernel="rbf",width=1.0)
    model   = KPLS(n_factors,kernel,width)
    message = MLJModelInterface.clean!(model)
    isempty(message) || @warn message
    return model
end

function MMI.clean!(m::KPLS)
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

function MMI.fit(m::PLS, verbosity::Int, X,Y)

    X =  MMI.matrix(X)
    Y = (MMI.scitype(Y) == Table{AbstractArray{Continuous,1}} ? MMI.matrix(Y) : Y)
    Y = (MMI.scitype(Y) == Table{AbstractArray{Continuous,2}} ? MMI.matrix(Y) : Y)

    check_constant_cols(X)
    check_constant_cols(Y)
    check_params(m.n_factors, size(X,2),"linear")
    check_data(X, Y)

    model                    = PLSModel(X,Y,m.n_factors)

    (fitresult,cache,report) = fitting(model,X,Y)

    return (fitresult,cache,report)

end


function MMI.fit(m::KPLS, verbosity::Int, X,Y)

    X = MMI.matrix(X)
    Y = (MMI.scitype(Y) == Table{AbstractArray{Continuous,1}} ? MMI.matrix(Y) : Y)
    Y = (MMI.scitype(Y) == Table{AbstractArray{Continuous,2}} ? MMI.matrix(Y) : Y)

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

function MMI.predict(m::Union{PLS,KPLS}, fitresult, X)

    X = MMI.matrix(X)
    check_data(X,fitresult.nfeatures)

    Y =  predicting(fitresult,X)
    Y = (length(size(Y)) > 1 ? MMI.table(Y) : Y)

    return Y
end

MMI.metadata_pkg.(
    (PLS, KPLS),
    name       = "Partial Least Squares Regressor",
    uuid       = "e010f91f-06b9-52b3-bed3-bb1da186bddc",
    url        = "https://github.com/lalvim/PLSRegressor.jl",
    julia      = true,
    license    = "MIT",
    is_wrapper = false)

MMI.metadata_model(PLS,
    input   = Table(Continuous),
    target  = Union{AbstractVector{<:Continuous},Table(Continuous)},
    weights = false,
    descr   = PLSRegressor_Desc)

MMI.metadata_model(KPLS,
    input   = Table(Continuous),
    target  = Union{AbstractVector{<:Continuous},Table(Continuous)},
    weights = false,
    descr   = KPLSRegressor_Desc)
