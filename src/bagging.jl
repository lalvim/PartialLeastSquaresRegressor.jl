
function bagging{T<:AbstractFloat}(X::AbstractArray{T},
                                   Y::AbstractArray{T};
                                   nfactors               = range(1,5,10),
                                   copydata::Bool         = true,
                                   centralize::Bool       = true,
                                   kernel                 = "rbf",
                                   width                  = linspace(0.01,3,10),
                                   nspaces::Int           = 100,
                                   reduce_func(x)         = sqrt(x),
                                  loss(yp,y)              = mean(abs.(yp .- y)))

    Y       = Y[:,:] # change to matrix
    n,m     = size(X)

    sub_n   = Int(floor(reduce_func(n)))
    sub_m   = Int(floor(reduce_func(m)))

    model_list = []
    for i=1:nspaces
        print("|")
        # mask for samples and features
        ind_samples   = rand(1:n,sub_n)
        ind_features  = rand(1:n,sub_m)
        ind_samplest  = setdiff(1:n,ind_samples) # out of sample data for validation

        # the sub samples
        X_train,Y_train = X[ind_samples,ind_features],Y[ind_samples,ind_features]
        X_test,Y_test   = X[ind_samplest,:],Y[ind_samplest,:]

        # getting the best model for the subsample
        min_l   = Inf#typemax(Float64)
        best_g  =  0
        best_w  = .0
        for g in nfactors, w in width
            model          = PLS.fit(X_train,Y_train,nfactors=g,copydata=copydata,centralize=centralze,kernel=kernel,width=w)
            Y_pred         = PLS.predict(model,X_test)
            l              = loss(Y_test,Y_pred)
            if l < min_l
               min_l     = l
               best_g    = g
               best_w    = w
           end
           print(".")
        end
        model          = PLS.fit(X_train,Y_train,nfactors=best_g,copydata=copydata,centralize=centralze,kernel=kernel,width=best_w)

        push!(model_list,model)
        print("|\n")
    end

    model_list

end


function vote{T<:AbstractFloat}(model_list::AbstractArray{PLSModel},
                                        X::AbstractArray{T},
                                        vote_func(x) = mean(x,2))

    m    = length(model_list)
    n    = size(X)
    r    = zeros(T,n,m)
    i    = 1
    for model in model_list
        r[:,i]  = PLS.predict(model,X)
        i += 1
    end

    return vec(vote_func(r))

end
