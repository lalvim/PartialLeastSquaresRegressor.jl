

## the learning algorithm: PLS2 - multiple targets
function trainer{T<:AbstractFloat}(model::PLS2Model{T},
                                   X::AbstractArray{T}, Y::Matrix{T})
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



function predictor{T<:AbstractFloat}(model::PLS2Model{T},
                                     X::AbstractArray{T})

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
