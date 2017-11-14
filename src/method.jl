# Partial Least Squares (PLS1 and PLS2 NIPALS version)
include("utils.jl")
include("src/types.jl")
#using PLSTypes
#reload("PLSTypes")
## constants
const NFACT          = 10              # default number of factors if it is not informed by the user


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
    model.ntargetcols = size(Y,2)
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
            return model
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
function fit{T<:AbstractFloat}(X::Matrix{T}, Y::AbstractArray{T};
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

    W,Q,P    = model.W,model.Q,model.P
    nfactors   = model.nfactors
    Y          = zeros(T,size(X,1),model.ntargetcols)
    #println("nfactors: ",nfactors)
    for i = 1:nfactors
        R      = X*W[:,i]
        X      = X - R * P[:,i]'
        Y      = Y + R * Q[:,i]'
    end

    return Y
end


"""
    transform(model::PLS.Model; X::Matrix{:<AbstractFloat}; copydata::Bool=true)

A Partial Least Squares predictor.

# Arguments
- `copydata::Bool = true`: If you want to use the same input matrix or a copy.
"""
function transform{T<:AbstractFloat}(model::PLSModel{T}},
                                    X::Matrix{T};
                                    copydata::Bool=true)
    check_data(X,model.nfeatures)

    Xi =  (copydata ? deepcopy(X) : X)
    Xi =  (model.centralize ? centralize_data(Xi,model.mx,model.sx) : Xi)

    Yi =  predictor(model,Xi)
    Yi =  decentralize_data(Yi,model.my,model.sy)

    return Yi
end
