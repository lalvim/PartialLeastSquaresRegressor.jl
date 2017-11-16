PLS.jl
======

A Partial Least Squares Regressor package


| **PackageEvaluator**            | **Build Status**                          |
|:-------------------------------:|:-----------------------------------------:|
| [![][pkg-0.6-img]][pkg-0.6-url] | [![][travis-img]][travis-url] [![][codecov-img]][codecov-url] |

[travis-img]: https://travis-ci.org/lalvim/PLS.jl.svg?branch=master
[travis-url]: https://travis-ci.org/lalvim/PLS.jl

[codecov-img]: http://codecov.io/github/lalvim/PLS.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/lalvim/PLS.jl?branch=master

[issues-url]: https://github.com/lalvim/PLS.jl/issues

[pkg-0.6-img]: http://pkg.julialang.org/badges/PLS_0.6.svg
[pkg-0.6-url]: http://pkg.julialang.org/?pkg=PLS&ver=0.6
[pkg-0.7-img]: http://pkg.julialang.org/badges/PLS_0.7.svg
[pkg-0.7-url]: http://pkg.julialang.org/?pkg=PLS&ver=0.7

Install
=======

    Pkg.add("PLS")

Using
=====

    using PLS

Examples
========

    using PLS

    # learning a single target
    X_train        = [1 2; 2 4; 4 6.0]
    Y_train        = [4; 6; 8.0]
    X_test         = [6 8; 8 10; 10 12.0]

    model          = PLS.fit(X_train,Y_train,nfactors=2)
    Y_test         = PLS.transform(model,X_test)

    print("[PLS1] mae error : $(mean(abs.(Y_test .- Y_pred)))")


    # learning multiples targets
    X_train        = [1 2; 2 4; 4 6.0]
    Y_train        = [2 4;4 6;6 8.0]
    X_test         = [6 8; 8 10; 10 12.0]

    model          = PLS.fit(X_train,Y_train,nfactors=2)
    Y_test         = PLS.transform(model,X_test)

    print("[PLS2] mae error : $(mean(abs.(Y_test .- Y_pred)))")


    # if you want to save your model
    PLS.save(model,filename="/tmp/pls_model.jld")

    # if you want to load back your model
    model = PLS.load(filename="/tmp/pls_model.jld")


What is Implemented
======
* PLS.fit - learns from input data and its related single target
    * X::Matrix{:<AbstractFloat} - A matrix that columns are the features and rows are the samples
    * Y::Vector{:<AbstractFloat} - A vector with float values.
    * nfactors::Int = 10 - The number of latent variables to explain the data.
    * copydata::Bool = true - If you want to use the same input matrix or a copy.
    * centralize::Bool = true - If you want to z-score columns. Recommended if not z-scored yet.
* PLS.transform - predicts using the learnt model extracted from fit.
    * model::PLS.Model - A PLS model learnt from fit.
    * X::Matrix{:<AbstractFloat} - A matrix that columns are the features and rows are the samples.
    * copydata::Bool = true - If you want to use the same input matrix or a copy.

What is not ready yet
=======
* A non linear Kernel PLS
* An automatic validation inside fit function


References
=======
* PLS1 and PLS2 NIPALS Algorithm (this implemented version) http://vision.cse.psu.edu/seminars/talks/PLSpresentation.pdf
* A Kernel PLS2 (this implemented version) http://www.jmlr.org/papers/volume2/rosipal01a/rosipal01a.pdf

* NIPALS: Nonlinear Iterative Partial Least Squares
    * Wold, H. (1966). Estimation of principal components and related models
by iterative least squares. In P.R. Krishnaiaah (Ed.). Multivariate Analysis.
(pp.391-420) New York: Academic Press.
* SIMPLS: more efficient, optimal result
    * Supports multivariate Y
    * De Jong, S., 1993. SIMPLS: an alternative approach to partial least squares
regression. Chemometrics and Intelligent Laboratory Systems, 18: 251–
263

License
=======

The PLS.jl is free software: you can redistribute it and/or modify it under the terms of the MIT "Expat"
License. A copy of this license is provided in ``LICENSE.md``
