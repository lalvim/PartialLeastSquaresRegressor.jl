"""
fit(model, X::AbstractArray{T},Y::AbstractArray{T})
A Partial Least Squares learning algorithm.
"""
function fitting(model,
             X::AbstractArray{T},
             Y::AbstractArray{T}) where T<:AbstractFloat

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
   predictor(model,X)
end
