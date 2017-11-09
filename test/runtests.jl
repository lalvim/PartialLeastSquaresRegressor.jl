using PLS
using Base.Test

# write your own tests here
reload("PLS")

@testset "One Column Prediction Test" begin

	X        = reshape([1; 2; 3.0],(3,1))
	Y        = [1; 2; 3.0]
	@time model = PLS.fit(X,Y,nfactors=1)
	pred = PLS.transform(model,X)
	print(pred)
	print([1; 2; 3.0])
    print( pred .= [1; 2; 3.0] )
	#@test isequal(pred,[1; 2; 3.0])

end;


@testset "Constant Values Prediction Tests" begin

	X        = [1 1;2 1;3 1.0]
	Y        = [1; 1; 1.0]
	@time model = PLS.fit(X,Y,nfactors=2)
	pred = PLS.transform(model,X)
	@test isequal(pred,[1; 1; 1.0])

	X        = reshape([1; 1; 1.0],(3,1))
	Y        = [1; 1; 1.0]
	@time model = PLS.fit(X,Y,nfactors=1)
	pred = PLS.transform(model,X)
	@test isequal(pred,[1; 1; 1.0])

end;

@testset "Linear Prediction Tests" begin


	X        = [1 2; 2 4; 4.0 6]
	Y        = [2; 4; 6.0]
	@time model = PLS.fit(X,Y,nfactors=2)
	pred = PLS.transform(model,X)
	@test isequal(pred,[2; 4; 6.0])

	X        = [1 -2; 2 -4; 4.0 -6]
	Y        = [-2; -4; -6.0]
	@time model = PLS.fit(X,Y,nfactors=2)
	pred = PLS.transform(model,X)
	@test isequal(pred,[-2; -4; -6.0])

end;
