import RDatasets

using MLJ
using MLJBase

import PLSRegressor: PLS


data = RDatasets.dataset("datasets", "longley");

y, X = unpack(data, ==(:GNP), colname -> true);

# algorothm
pls_model      = PLS(n_factors=1,centralize=true,copy_data=true,rng=42)

# associating algo. and data
pls_machine    = MLJ.machine(pls_model, X, y)

# evaluate you regressor using cross validation
MLJ.evaluate!(pls_machine, resampling=CV(shuffle=true), measure=mae, verbosity=0)


# you can use hould out
train, test    = partition(eachindex(y), 0.7, shuffle=true); 

fit!(pls_machine, rows=train)

yhat = predict(pls_machine, rows=test);

mae(yhat, y[test]) |> mean



