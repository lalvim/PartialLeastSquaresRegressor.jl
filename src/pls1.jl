## the learning algorithm: PLS1 - single target
function trainer(model::PLS1Model{T},
                 X::AbstractArray{T}, Y::Vector{T}) where T<:AbstractFloat
   
    W,b,P  = model.W,model.b,model.P
    nfactors = model.nfactors
    for i = 1:nfactors
        W[:,i] = X'Y
        W[:,i] /= norm(W[:,i])#sqrt.(W[:,i]'*W[:,i])
        R      = X*W[:,i]
        Rn     = R'/(R'R) # change to use function...
        P[:,i] = Rn*X
        b[i]   = Rn * Y
        if abs(b[i]) <= 1e-3
           print("PLS1 converged. No need learning with more than $(i) factors")
           model.nfactors = i
           break
        end
        X      = X - R * P[:,i]'
        Y      = Y - R * b[i]
    end

    return model
end


function predictor(model::PLS1Model{T},
                   X::AbstractArray{T}) where T<:AbstractFloat

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

function filter!(model::PLS1Model{T},
    X::AbstractArray{T}) where T<:AbstractFloat

    nrows  = size(X,1)
    X_proj = zeros(T, nrows, model.nfactors)
    Y_pred = zeros(size(X,1))

    for i = 1:model.nfactors
        X_proj[:,i] = X*model_pls.W[:,i]
        X           = X - X_proj[:,i]*model_pls.P[:,i]'
        Y_pred      = Y_pred + X_proj[:,i]*model_pls.b[i] # building prediction
    end

    return X, X_proj, Y_pred
end

function component(model::PLS1Model{T},i) where T<:AbstractFloat
    return model_pls.W[:,i]
end