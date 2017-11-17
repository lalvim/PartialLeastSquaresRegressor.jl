

@testset "Test KPLS Model" begin


    X        = [1 2; 2 4; 4.0 6]
    Y        = [2; 4; 6.0]

	#Xtr        = [1 -2; 2 -4; 4.0 -6]
	#Ytr        = [-2 -2; -4 -4; -6.0 -6.0]
	#Xt         = [6 -8; 8 -10; 10.0 -12]
	model    = PLS.fit(X,Y,nfactors=2,kernel="gaussian",width=1.0)
	pred     = PLS.transform(model,X)
    print(pred)


end
