import RDatasets

using MLJ
using MLJBase

import PLSRegressor: PLS,KPLS


data = RDatasets.dataset("datasets", "longley")[:,2:5];

y, X = unpack(data, ==(:GNP), colname -> true);

# algorothm
pls_model      = PLS(n_factors=3,centralize=true,copy_data=true,rng=42)

# associating algo. and data
pls_machine    = MLJ.machine(pls_model, X, y)


# evaluate you regressor using cross validation
MLJ.evaluate!(pls_machine, resampling=CV(shuffle=true), measure=[mae], verbosity=0)





