### This is a validation test
### the objective is to show that KPLS can easily fit any curve using the same matrix
### If not, the regressor could be wrong

### Example extracted from a python KPLS implementation: https://github.com/jhumphry/regressions/blob/master/examples/kpls_example.py

using PLSRegressor
using Gadfly

srand(1)

z(x)     = 4.26 * (exp.(-x) - 4 * exp.(-2.0*x) + 3 * exp.(-3.0*x))
x_values = linspace(0.0,3.5,100)
z_pure   = z(x_values)
noise    = randn(100)
z_noisy  = z_pure + noise
X        = collect(x_values)
Y        = z_noisy #z_pure

min_mae = 10
global best_pred
global best_w = 10
global best_g = 10

for g in [1,2],
    w in linspace(0.01,3,10)
    print(".")
    model      = PLSRegressor.fit(X,Y,centralize=true,nfactors=g,kernel="rbf",width=w)
    Y_pred     = PLSRegressor.predict(model,X)
    mae = mean(abs.(Y .- Y_pred))
    if mae < min_mae
       min_mae = mae
       best_pred = Y_pred[:]
       best_g    = g
       best_w    = w
   end
end

print("[KPLS] min mae error : $(min_mae)")
print("[KPLS] best factor : $(best_g)")
print("[KPLS] best width : $(best_w)")

plot([Y,best_pred ], x=Row.index, y=Col.value, color=Col.index, Geom.line)
