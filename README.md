# PartialLeastSquaresRegressor.jl
[![][travis-img]][travis-url] [![][codecov-img]][codecov-url] [![][coverage-img]][coverage-url]

The PartialLeastSquaresRegressor.jl package is a package with Partial Least Squares Regressor methods. Contains PLS1, PLS2 and Kernel PLS2 NIPALS algorithms.
Can be used mainly for regression. However, for classification task, binarizing targets and then obtaining multiple targets, you can apply KPLS.

## Install

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add PartialLeastSquaresRegressor
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("PartialLeastSquaresRegressor")
```

## Using

PartialLeastSquaresRegressor is used with [MLJ](https://github.com/alan-turing-institute/MLJ.jl) machine learning framework. Here are a few examples to show the Package functionalities:

### Example 1

```julia
using MLJBase, RDatasets, MLJModels
PLSRegressor = @load PLSRegressor pkg=PartialLeastSquaresRegressor

# loading data and selecting some features
data = dataset("datasets", "longley")[:, 2:5]

# unpacking the target
y, X = unpack(data, ==(:GNP))

# loading the model
regressor = PLSRegressor(n_factors=2)

# building a pipeline with scaling on data
pipe = Standardizer |> regressor
model = TransformedTargetModel(pipe, transformer=Standardizer())

# a simple hould out
(Xtrain, Xtest), (ytrain, ytest) = partition((X, y), 0.7, rng=123, multi=true)

mach = machine(model, Xtest, ytest)

fit!(mach)
yhat = predict(mach, Xtest)

mae(yhat, ytest) |> mean
```

### Example 2

```julia
using MLJBase, RDatasets, MLJTuning, MLJModels
@load KPLSRegressor pkg=PartialLeastSquaresRegressor

# loading data and selecting some features
data = dataset("datasets", "longley")[:, 2:5]

# unpacking the target
y, X = unpack(data, ==(:GNP), colname -> true)

# loading the model
pls_model = KPLSRegressor()

# defining hyperparams for tunning
r1 = range(pls_model, :width, lower=0.001, upper=100.0, scale=:log)

# attaching tune
self_tuning_pls_model = TunedModel(model =          pls_model,
                                   resampling = CV(nfolds = 10),
                                   tuning = Grid(resolution = 100),
                                   range = [r1],
                                   measure = mae)

# putting into the machine
self_tuning_pls = machine(self_tuning_pls_model, X, y)

# fitting with tunning
fit!(self_tuning_pls, verbosity=0)

# getting the report
report(self_tuning_pls)
```

## What is Implemented

* A fast linear algorithm for single targets (PLS1 - NIPALS)
* A linear algorithm for multiple targets (PLS2 - NIPALS)
* A non linear algorithm for multiple targets (Kernel PLS2 - NIPALS)

## Model Description

* PLS - PLS MLJ model (PLS1 or PLS2)
    * n_factors::Int = 10 - The number of latent variables to explain the data.

* KPLS - Kernel PLS MLJ model
    * nfactors::Int = 10 - The number of latent variables to explain the data.
    * kernel::AbstractString = "rbf" - use a non linear kernel.
    * width::AbstractFloat   = 1.0 - If you want to z-score columns. Recommended if not z-scored yet.

## References

* PLS1 and PLS2 based on
   * Bob Collins Slides, LPAC Group. http://vision.cse.psu.edu/seminars/talks/PLSpresentation.pdf
* A Kernel PLS2 based on
   * Kernel Partial Least Squares Regression in Reproducing Kernel Hilbert Space" by Roman Rosipal and Leonard J Trejo. Journal of Machine Learning Research 2 (2001) 97-123 http://www.jmlr.org/papers/volume2/rosipal01a/rosipal01a.pdf

* NIPALS: Nonlinear Iterative Partial Least Squares
    * Wold, H. (1966). Estimation of principal components and related models
by iterative least squares. In P.R. Krishnaiaah (Ed.). Multivariate Analysis.
(pp.391-420) New York: Academic Press.

* SIMPLS: more efficient, optimal result
    * Supports multivariate Y
    * De Jong, S., 1993. SIMPLS: an alternative approach to partial least squares
regression. Chemometrics and Intelligent Laboratory Systems, 18: 251–
263

## License

The PartialLeastSquaresRegressor.jl is free software: you can redistribute it and/or modify it under the terms of the MIT "Expat"
License. A copy of this license is provided in ``LICENSE``

[travis-img]: https://travis-ci.com/lalvim/PartialLeastSquaresRegressor.jl.svg?branch=master
[travis-url]: https://travis-ci.com/lalvim/PartialLeastSquaresRegressor.jl

[codecov-img]: https://codecov.io/gh/lalvim/PartialLeastSquaresRegressor.jl/branch/master/graph/badge.svg?token=13TrPsgakO
[codecov-url]: https://codecov.io/gh/lalvim/PartialLeastSquaresRegressor.jl

[coverage-img]: https://coveralls.io/repos/github/lalvim/PartialLeastSquaresRegressor.jl/badge.svg?branch=master
[coverage-url]: https://coveralls.io/github/lalvim/PartialLeastSquaresRegressor.jl?branch=master
