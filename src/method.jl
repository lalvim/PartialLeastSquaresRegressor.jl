"""
fit(model, X::AbstractArray{T},Y::AbstractArray{T})
A Partial Least Squares learning algorithm.
"""
function fit(model,
             X::AbstractArray{T},
             Y::AbstractArray{T}) where T<:AbstractFloat

    X = (model.centralize ? centralize_data(X,model.mx,model.sx) : X)
    Y = (model.centralize ? centralize_data(Y,model.my,model.sy) : Y)

    fitresult = trainer(model,X,Y)
    report    = nothing
    cache     = nothing

    return (fitresult,cache,report)
end

"""
transform(model, X::AbstractArray{T})
A Partial Least Squares predictor.
"""
function predict(model,X::AbstractArray{T}) where T<:AbstractFloat

   X =  (model.centralize ? centralize_data(X,model.mx,model.sx) : X)
   Y =  predictor(model,X)
   Y =  (model.centralize ? decentralize_data(Y,model.my,model.sy) : Y)

   return Y
end
