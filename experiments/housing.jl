using Knet
using MultivariateStats
using PLS

function loaddata(test=0.1)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    file=Knet.dir("data","housing.data")
    nfeatures = 13
    target_idx = 14
    if !isfile(file)
        info("Downloading $url to $file")
        download(url, file)
    end
    data = readdlm(file)'
    @show size(data) # (14,506)
    x = data[1:nfeatures,:]
    y = data[target_idx:target_idx,:]
    if test == 0
        xtrn = xtst = x
        ytrn = ytst = y
    else
        r = randperm(size(x,2))          # trn/tst split
        n = round(Int, (1-test) * size(x,2))
        xtrn=x[:,r[1:n]]
        ytrn=y[:,r[1:n]]
        xtst=x[:,r[n+1:end]]
        ytst=y[:,r[n+1:end]]
    end
    (xtrn, ytrn, xtst, ytst)
end

ds = loaddata()


model    = PLS.fit(ds[1]',[ds[2]'...],nfactors=8)
pred     = PLS.transform(model,ds[3]')

print("[PLS] mae error :",mean(abs.([ds[4]'...] .- pred)))

# linear least squares from MultiVariateStats
sol = llsq(ds[1]',[ds[2]'...])
a, b = sol[1:end-1], sol[end]
yp = ds[3]' * a + b
print("[LLS] mae error :",mean(abs.([ds[4]'...] .- yp)))
