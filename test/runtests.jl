using PLS
using Base.Test

# write your own tests here

X        = [1 2; 2 4.5; 4.7 9.3]
Y        = [2.1; 4.6; 9.4]

#fit(X, Y, nfactors=1,copydata=true)
PLS.fit(X, Y, nfactors=1)

#, 10, copydata=true)
