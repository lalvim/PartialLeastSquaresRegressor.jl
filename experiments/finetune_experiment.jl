import RDatasets

using MLJ
using MLJBase

import PLSRegressor: PLS,KPLS


data = RDatasets.dataset("datasets", "longley")[:,2:5];

y, X = unpack(data, ==(:GNP), colname -> true);


pls_model      = KPLS(n_factors=1,kernel="rbf",width=0.01,centralize=true,rng=42)


r1 = MLJ.range(pls_model, :width, lower=0.001, upper=100.0)#, scale=:log);
r2 = MLJ.range(pls_model, :n_factors, lower=1, upper=3);


self_tuning_pls_model = TunedModel(model=pls_model,
                              resampling = CV(nfolds=10),
                              tuning = Grid(resolution=100),
                              range = [r1,r2],
                              measure = mae);

self_tuning_pls = machine(self_tuning_pls_model, X, y);

MLJ.fit!(self_tuning_pls, verbosity=0)

MLJ.report(self_tuning_pls).best_result
MLJ.report(self_tuning_pls).best_model


#using Plots
#plot(self_tuning_pls)


