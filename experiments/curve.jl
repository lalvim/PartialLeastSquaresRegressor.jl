### This is a validation test
### the objective is to show that KPLS can easily fit any curve using the same matrix
### If not, the regressor could be wrong

### Example extracted from a python KPLS implementation: https://github.com/jhumphry/regressions/blob/master/examples/kpls_example.py


using MLJ
using MLJBase

import PLSRegressor: KPLS, PLS


import Random

Random.seed!(1)

X = MLJ.table(rand(100, 2));
X = coerce(X,:x1=>Continuous,:x2=>Continuous) 
y = 2X.x1 + X.x2 - 0.05*rand(100)  

pls_model      = KPLS(n_factors=2,kernel="rbf",width=1.0,centralize=true,rng=42)


r1 = MLJ.range(pls_model, :width, lower=0.001, upper=10.0, scale=:log);
r2 = MLJ.range(pls_model, :n_factors, lower=1, upper=2);


self_tuning_pls_model = TunedModel(model=pls_model,
                              resampling = CV(nfolds=10),
                              tuning = Grid(resolution=100),
                              range = [r1,r2],
                              measure = mae);

self_tuning_pls = machine(self_tuning_pls_model, X, y);

MLJ.fit!(self_tuning_pls, verbosity=0)

MLJ.report(self_tuning_pls).best_result
MLJ.report(self_tuning_pls).best_model

