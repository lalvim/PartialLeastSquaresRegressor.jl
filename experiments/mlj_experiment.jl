import RDatasets

using MLJ
using MLJBase

import PLSRegressor: PLS


data = RDatasets.dataset("datasets", "longley");

y, X = unpack(data, ==(:GNP), colname -> true);

# algorothm
pls_model      = PLS(n_factors=10,centralize=true,copy_data=true,rng=42)

# associating algo. and data
pls_machine    = MLJ.machine(pls_model, X, y)

# evaluate you regressor using cross validation
MLJ.evaluate!(pls_machine, resampling=CV(shuffle=true), measure=[mae], verbosity=0)


# you can use hould out
train, test    = MLJ.partition(eachindex(y), 0.7, shuffle=true); 

MLJ.fit!(pls_machine, rows=train)

yhat = MLJ.predict(pls_machine, rows=test);

MLJ.mae(yhat, y[test]) |> mean



