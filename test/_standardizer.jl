module Stand

export Standardizer

using MLJModelInterface
using ..MLJBase.Tables
const MMI = MLJModelInterface
using Statistics
using ScientificTypes
#import ..MLJBase.MLJScientificTypes.coerce
#import ScientificTypes.coerce

const UNIVARIATE_STD_DESCR = "Standardize (whiten) univariate data."
const STANDARDIZER_DESCR = "Standardize (whiten) features (columns) "*
"of a table."

## UNIVARIATE STANDARDIZATION

"""
    UnivariateStandardizer()

Unsupervised model for standardizing (whitening) univariate data.
"""
mutable struct UnivariateStandardizer <: Unsupervised end

function MMI.fit(transformer::UnivariateStandardizer, verbosity::Int,
             v::AbstractVector{T}) where T<:Real
    std(v) > eps(Float64) ||
        @warn "Extremely small standard deviation encountered in standardization."
    fitresult = (mean(v), std(v))
    cache = nothing
    report = NamedTuple()
    return fitresult, cache, report
end


MMI.fitted_params(::UnivariateStandardizer, fitresult) =
    (mean_and_std = fitresult, )


# for transforming single value:
function MMI.transform(transformer::UnivariateStandardizer, fitresult, x::Real)
    mu, sigma = fitresult
    return (x - mu)/sigma
end

# for transforming vector:
MMI.transform(transformer::UnivariateStandardizer, fitresult, v) =
              [transform(transformer, fitresult, x) for x in v]

# for single values:
function MMI.inverse_transform(transformer::UnivariateStandardizer, fitresult, y::Real)
    mu, sigma = fitresult
    return mu + y*sigma
end

# for vectors:
MMI.inverse_transform(transformer::UnivariateStandardizer, fitresult, w) =
    [inverse_transform(transformer, fitresult, y) for y in w]

## STANDARDIZATION OF ORDINAL FEATURES OF TABULAR DATA

"""
    Standardizer(; features=Symbol[],
                   ignore=false,
                   ordered_factor=false,
                   count=false)

Unsupervised model for standardizing (whitening) the columns of
tabular data.  If `features` is unspecified then all columns
having `Continuous` element scitype are standardized. Otherwise, the
features standardized are the `Continuous` features named in
`features` (`ignore=false`) or `Continuous` features not named in
`features` (`ignore=true`). To allow standarization of `Count` or
`OrderedFactor` features as well, set the appropriate flag to true.

Instead of supplying a features vector, a Bool-valued callable with one 
argument can be also be specified. For example, specifying 
`Standardizer(features = name -> name in [:x1, :x3], ignore = true, count=true)`  
has the same effect as `Standardizer(features = [:x1, :x3], ignore = true,
count=true)`, namely to standardise all `Continuous` and `Count` features, 
with the exception of `:x1` and `:x3`.

The `inverse_tranform` method is supported provided `count=false` and
`ordered_factor=false` at time of fit.

# Example

```
X = (ordinal1 = [1, 2, 3],
     ordinal2 = coerce([:x, :y, :x], OrderedFactor),
     ordinal3 = [10.0, 20.0, 30.0],
     ordinal4 = [-20.0, -30.0, -40.0],
     nominal = coerce(["Your father", "he", "is"], Multiclass));
stand1 = Standardizer();
julia> transform(fit!(machine(stand1, X)), X)
[ Info: Training Machine{Standardizer} @ 7…97.
(ordinal1 = [1, 2, 3],
 ordinal2 = CategoricalValue{Symbol,UInt32}[:x, :y, :x],
 ordinal3 = [-1.0, 0.0, 1.0],
 ordinal4 = [1.0, 0.0, -1.0],
 nominal = CategoricalVale{String,UInt32}["Your father", "he", "is"],)

stand2 = Standardizer(features=[:ordinal3, ], ignore=true, count=true);
julia> transform(fit!(machine(stand2, X)), X)
[ Info: Training Machine{Standardizer} @ 1…87.
(ordinal1 = [-1.0, 0.0, 1.0],
 ordinal2 = CategoricalValue{Symbol,UInt32}[:x, :y, :x],
 ordinal3 = [10.0, 20.0, 30.0],
 ordinal4 = [1.0, 0.0, -1.0],
 nominal = CategoricalValue{String,UInt32}["Your father", "he", "is"],)
```

"""
mutable struct Standardizer <: Unsupervised
    # features to be standardized; empty means all
    features::Union{AbstractVector{Symbol}, Function}
    ignore::Bool # features to be ignored
    ordered_factor::Bool
    count::Bool
end

# keyword constructor
function Standardizer(
    ;
    features::Union{AbstractVector{Symbol}, Function}=Symbol[],
    ignore::Bool=false,
    ordered_factor::Bool=false,
    count::Bool=false
)
    transformer = Standardizer(features, ignore, ordered_factor, count)
    message = MMI.clean!(transformer)
    isempty(message) || throw(ArgumentError(message))
    return transformer
end

function MMI.clean!(transformer::Standardizer)
    err = ""
    if (
        typeof(transformer.features) <: AbstractVector{Symbol} &&
        isempty(transformer.features) &&
        transformer.ignore
    )
        err *= "Features to be ignored must be specified in features field."
    end
    return err
end

function MMI.fit(transformer::Standardizer, verbosity::Int, X)

    # if not a table, it must be an abstract vector, eltpye AbstractFloat:
    is_univariate = !Tables.istable(X)

    # are we attempting to standardize Count or OrderedFactor?
    is_invertible = !transformer.count && !transformer.ordered_factor

    # initialize fitresult:
    fitresult_given_feature = Dict{Symbol,Tuple{Float64,Float64}}()

    # special univariate case:
    if is_univariate
        fitresult_given_feature[:unnamed] =
            MMI.fit(UnivariateStandardizer(), verbosity - 1, X)[1]
        return (is_univariate=true,
                is_invertible=true,
                fitresult_given_feature=fitresult_given_feature),
        nothing, nothing
    end

    all_features = Tables.schema(X).names
    feature_scitypes =
        collect(elscitype(selectcols(X, c)) for c in all_features)
    scitypes = Vector{Type}([Continuous])
    transformer.ordered_factor && push!(scitypes, OrderedFactor)
    transformer.count && push!(scitypes, Count)
    AllowedScitype = Union{scitypes...}

    # determine indices of all_features to be transformed
    if transformer.features isa AbstractVector{Symbol}
        if isempty(transformer.features)
            cols_to_fit = filter!(eachindex(all_features) |> collect) do j
                feature_scitypes[j] <: AllowedScitype
            end
        else
            !issubset(transformer.features, all_features) && verbosity > -1 &&
                @warn "Some specified features not present in table to be fit. "
            cols_to_fit = filter!(eachindex(all_features) |> collect) do j
                ifelse(
                    transformer.ignore,
                    !(all_features[j] in transformer.features) &&
                        feature_scitypes[j] <: AllowedScitype,
                    (all_features[j] in transformer.features) &&
                        feature_scitypes[j] <: AllowedScitype
                )
            end
        end
    else
        cols_to_fit = filter!(eachindex(all_features) |> collect) do j
            ifelse(
                transformer.ignore,
                !(transformer.features(all_features[j])) &&
                    feature_scitypes[j] <: AllowedScitype,
                (transformer.features(all_features[j])) &&
                    feature_scitypes[j] <: AllowedScitype
            )
        end
    end
    fitresult_given_feature = Dict{Symbol,Tuple{Float64,Float64}}()

    isempty(cols_to_fit) && verbosity > -1 &&
        @warn "No features to standarize."

    # fit each feature and add result to above dict
    verbosity < 2 || @info "Features standarized: "
    for j in cols_to_fit
        col_data = if (feature_scitypes[j] <: OrderedFactor)
            coerce(selectcols(X, j), Continuous)
        else
            selectcols(X, j)
        end
        col_fitresult, cache, report =
            MMI.fit(UnivariateStandardizer(), verbosity - 1, col_data)
        fitresult_given_feature[all_features[j]] = col_fitresult
        verbosity < 2 ||
            @info "  :$(all_features[j])    mu=$(col_fitresult[1])  sigma=$(col_fitresult[2])"
    end

    fitresult = (is_univariate=false, is_invertible=is_invertible,
                 fitresult_given_feature=fitresult_given_feature)
    cache = nothing
    report = (features_fit=keys(fitresult_given_feature),)

    return fitresult, cache, report
end

function MMI.fitted_params(::Standardizer, fitresult)
    is_univariate, _, dic = fitresult
    is_univariate &&
        return fitted_params(UnivariateStandardizer(), dic[:unnamed])
    return (mean_and_std_given_feature=dic)
end

MMI.transform(::Standardizer, fitresult, X) =
    _standardize(transform, fitresult, X)

function MMI.inverse_transform(::Standardizer, fitresult, X)
    fitresult.is_invertible ||
        error("Inverse standardization is not supported when `count=true` "*
              "or `ordered_factor=true` during fit. ")
    return _standardize(inverse_transform, fitresult, X)
end

function _standardize(operation, fitresult, X)

    # `fitresult` is dict of column fitresults, keyed on feature names
    is_univariate, _, fitresult_given_feature = fitresult

    if is_univariate
        univariate_fitresult = fitresult_given_feature[:unnamed]
        return operation(UnivariateStandardizer(), univariate_fitresult, X)
    end

    features_to_be_transformed = keys(fitresult_given_feature)

    all_features = Tables.schema(X).names

    all(e -> e in all_features, features_to_be_transformed) ||
        error("Attempting to transform data with incompatible feature labels.")

    col_transformer = UnivariateStandardizer()

    cols = map(all_features) do ftr
        ftr_data = selectcols(X, ftr)
        if ftr in features_to_be_transformed
            col_to_transform = coerce(ftr_data, Continuous)
            operation(col_transformer,
                      fitresult_given_feature[ftr],
                      col_to_transform)
        else
            ftr_data
        end
    end

    named_cols = NamedTuple{all_features}(tuple(cols...))

    return MMI.table(named_cols, prototype=X)

    metadata_model(UnivariateStandardizer,
               input=AbstractVector{<:MLJBase.Infinite},
               output=AbstractVector{MLJBase.Continuous},
               weights=false,
               descr=UNIVARIATE_STD_DESCR,
               path="MLJBase.UnivariateStandardizer")

    metadata_model(Standardizer,
               input=MLJBase.Table,
               output=MLJBase.Table,
               weights=false,
               descr=STANDARDIZER_DESCR,
               path="MLJBase.Standardizer")

end

end
