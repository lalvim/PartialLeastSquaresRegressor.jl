"""
fit(model, X::AbstractArray{T},Y::AbstractArray{T})
A Partial Least Squares learning algorithm.
"""
function fitting(model,
             X::AbstractArray{T},
             Y::AbstractArray{T}) where T<:AbstractFloat

    X = (model.standardize ? standardize_data(X,model.mx,model.sx) : X)
    Y = (model.standardize ? standardize_data(Y,model.my,model.sy) : Y)

    fitresult = trainer(model,X,Y)
    report    = nothing
    cache     = nothing

    return (fitresult,cache,report)
end

"""
transform(model, X::AbstractArray{T})
A Partial Least Squares predictor.
"""
function predicting(model,X::AbstractArray{T}) where T<:AbstractFloat

   X =  (model.standardize ? standardize_data(X,model.mx,model.sx) : X)
   Y =  predictor(model,X)
   Y =  (model.standardize ? destandardize_data(Y,model.my,model.sy) : Y)

   return Y
end
