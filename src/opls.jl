# based on the code from https://github.com/BiRG/pyopls
# reference:  Johan Trygg and Svante Wold. Orthogonal projections to latent structures (O-PLS). 
#             J. Chemometrics 2002; 16: 119-128. DOI: 10.1002/cem.695



##filtering and learning - Ortogonal PLS, one feature
function fitting(model::OPLS1Model,
    X::AbstractArray{T}, 
    Y::Vector{T}) where T<:AbstractFloat
    
    w=X'*Y         # calculate weight vector
    w=w/norm(w)    # normalization

    for i in 1:model.n_ortho_components
        t=X*w                     # calculate scores vector  nrows
        p=X'*t/(t'*t)             # calculate loadings of X  ncols
        wosc=p-(w'*p)/(w'*w)*w    # orthogonal weight        ncols
        wosc=wosc/norm(wosc)      # normalization            ncols
        tosc=X*wosc               # orthogonal components    nrows
        posc=X'*tosc/(tosc'*tosc) # loadings                 ncols
        
        X=X-tosc*posc'             # remove orthogonal components

        model.W_ortho[:,i]=wosc    # weights orthogonal to y
        model.P_ortho[:,i]=posc    # loadings orthogonal to y
        model.T_ortho[:,i]=tosc    # scores orthogonal to y
    end

    # X is now with orthogonal signal components removed
    return X, model.T_ortho
end

# remove orthogonal components from Xt
# return filtered Xt and weightings of orthogonal components
function filter!(model::OPLS1Model,
    X::AbstractArray{T}) where T<:AbstractFloat
    nrow = size(X,1)
    # ortogonal weights
    ortho = zeros(T, nrow, model.n_ortho_components)

    for i in 1:model.n_ortho_components
        R = X * model.W_ortho[:,i]
        X = X - R * model.P_ortho[:,i]'
        ortho[:,i] = R
    end

    return X, ortho
end

function component(model::OPLS1Model{T},i) where T<:AbstractFloat
    return model.W_ortho[:,i]
end