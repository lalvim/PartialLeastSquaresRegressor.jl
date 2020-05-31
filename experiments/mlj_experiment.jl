import RDatasets
using PLSRegressor


data = RDatasets.dataset("datasets", "longley");

y, X = unpack(data, ==(:GNP), colname -> true);

pls_model      = PLS(n_factors=1,centralize=true,copy_data=true,rng=42)
pls_machine    = machine(pls_model, X, y)

# evaluate you regressor
evaluate!(pls_machine, resampling=CV(shuffle=true), measure=mae, verbosity=0)



# or you can use withou evalutate
train, test    = partition(eachindex(y), 0.7, shuffle=true); 

fit!(pls_machine, rows=train)

yhat = predict(pls_machine, rows=test);

mae(yhat, y[test]) |> mean



