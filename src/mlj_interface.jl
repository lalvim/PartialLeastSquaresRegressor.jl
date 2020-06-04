using MLJModelInterface 
using Random

const MMI = MLJModelInterface

const PLSRegressor_Desc = "A Partial Least Squares Regressor. Contains PLS1, PLS2 (multi target) algorithms. Can be used mainly for regression."
const KPLSRegressor_Desc = "A Kernel Partial Least Squares Regressor. A Kernel PLS2 NIPALS algorithms. Can be used mainly for regression."

const MLJDICT = Dict(:Pls1 => PLS1Model,:Pls2 => PLS2Model,:Kpls => KPLSModel)

mutable struct PLS <: MMI.Deterministic
    n_factors::Int             
    centralize::Bool           
    rng::Int                   
end

mutable struct KPLS <: MMI.Deterministic
    n_factors::Integer           # = 10
    centralize::Bool             # = false
    kernel::String               # = :rbf
    width::Real                  # = 1.0
    rng::Union{AbstractRNG, Integer} # = Random.GLOBAL_RNG
end

function PLS(; n_factors=1,centralize=false,rng=42)
    model   = PLS(n_factors,centralize,rng)
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

function KPLS(; n_factors=1,centralize=false,kernel="rbf",width=1.0,rng=42)
    model   = KPLS(n_factors,centralize,kernel,width,rng)
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
    
    #X = convert(Array{Float64, 2}, MMI.matrix(X))
    if typeof(X) != Array{Float64,2}
        X = MMI.matrix(X)
    end
 
    check_constant_cols(X)
    check_constant_cols(Y)

    check_params(m.n_factors, size(X,2),"linear")

    check_data(X, Y)

    Xi =  X #(m.copy_data ? deepcopy(X) : X)
    Yi =  Y #(m.copy_data ? deepcopy(Y) : Y)


    fitresult = PLSModel(Xi,Yi,m.n_factors, m.centralize)

    Xi =  (m.centralize ? centralize_data(Xi,fitresult.mx,fitresult.sx) : Xi)
    Yi =  (m.centralize ? centralize_data(Yi,fitresult.my,fitresult.sy) : Yi)
    fitresult.centralize  = (m.centralize ? true : false)

    trainer(fitresult,Xi,Yi)

    report = nothing    
    cache  = nothing
    return (fitresult,cache,report)

end

function MMI.fit(m::KPLS, verbosity::Int, X,Y)
    
    #X = convert(Array{Float64, 2}, MMI.matrix(X))
    if typeof(X) != Array{Float64,2}
        X = MMI.matrix(X)
    end
 
    check_constant_cols(X)
    check_constant_cols(Y)

    check_params(m.n_factors, size(X,2),m.kernel)

    check_data(X, Y)

    Xi =  X #(m.copy_data ? deepcopy(X) : X)
    Yi =  Y #(m.copy_data ? deepcopy(Y) : Y)

    fitresult = PLSModel(Xi,Yi,
                 m.n_factors,
                 m.centralize,
                 m.kernel,
                 m.width)

    Xi =  (m.centralize ? centralize_data(Xi,fitresult.mx,fitresult.sx) : Xi)
    Yi =  (m.centralize ? centralize_data(Yi,fitresult.my,fitresult.sy) : Yi)
    fitresult.centralize  = (m.centralize ? true : false)

    trainer(fitresult,Xi,Yi)
    report = nothing    
    cache  = nothing

    return (fitresult,cache,report)
end

function MMI.predict(m::Union{PLS,KPLS}, fitresult, X) 
    
    #X = convert(Array{Float64, 2}, MMI.matrix(X))
    if typeof(X) != Array{Float64,2}
       X = MMI.matrix(X)
    end
    check_data(X,fitresult.nfeatures)

    Xi =  X #(m.copy_data ? deepcopy(X) : X)
    Xi =  (m.centralize ? centralize_data(Xi,fitresult.mx,fitresult.sx) : Xi)
    Yi =  predictor(fitresult,Xi)
    Yi =  (m.centralize ? decentralize_data(Yi,fitresult.my,fitresult.sy) : Yi)

    return Yi
end

MMI.metadata_pkg.(
    (PLS, KPLS),
    name       = "Partial Least Squares Regressor",
    uuid       = "e010f91f-06b9-52b3-bed3-bb1da186bddc",
    url        = "https://github.com/lalvim/PLSRegressor.jl",
    julia      = true,
    license    = "MIT",
    is_wrapper = false) # ?

MMI.metadata_model(PLS,
    input   = Table(Continuous),
    target  = AbstractVector{<:Continuous},
    weights = false,
    descr   = PLSRegressor_Desc)

MMI.metadata_model(KPLS,
    input   = Table(Continuous),
    target  = AbstractVector{<:Continuous},
    weights = false,
    descr   = KPLSRegressor_Desc)
