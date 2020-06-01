import RDatasets

using MLJ
using MLJBase

import PLSRegressor: PLS,KPLS


data = RDatasets.dataset("Ecdat", "Housing")#[:,2:5];
data


y, X = unpack(data, ==(:Price), colname -> true);

# algorithm
#pls_model      = PLS(n_factors=3,centralize=true,copy_data=true,rng=42)
pls_model      = KPLS(n_factors=1,kernel="rbf",width=0.01,centralize=true,rng=42)

#@pipeline MyPipe(X -> coerce(X, :Price=>Continuous,:LotSize=>Continuous,:LotSize=>Continuous,:Bathrms=>Continuous,:Stories=>Continuous,:GaragePl=>Continuous),
#                      hot = OneHotEncoder(),
#                      regressor = pls_model)


# associating algo. and data
pls_machine    = MLJ.machine(pls_model, X, y)

# you can use hould out
train, test    = MLJ.partition(eachindex(y), 0.7, shuffle=true); 

MLJ.fit!(MyPipe, rows=train)

yhat = MLJ.predict(pls_machine, rows=train);

MLJ.mae(yhat, y[train]) |> mean

yhat = MLJ.predict(pls_machine, rows=test);

MLJ.mae(yhat, y[test]) |> mean
