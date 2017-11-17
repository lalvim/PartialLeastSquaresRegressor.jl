
# A gaussian kernel function
@inline function Φ{T<:AbstractFloat}(x::Vector{T},
                                     y::Vector{T},
                                     r::T=1.0)
    n  = 1.0 / sqrt(2π*r)
    s  = 1.0 / (2r^2)
    return n*exp(-s*sum((x.-y).^2))
end

# A kernel matrix
function ΦΦ{T<:AbstractFloat}(X::AbstractArray{T},
                              r::T=1.0)
    n = size(X,1)
    K = zeros(n,n)
    for i=1:n
        for j=1:i
            K[i, j] = Φ(X[i, :], X[j, :],r)
            K[j, i] = K[i, j]
        end
        K[i, i] = Φ(X[i, :], X[i, :],r)
    end
    K
end

# A kernel matrix for test data
function ΦΦ{T<:AbstractFloat}(X::AbstractArray{T},
                              Z::AbstractArray{T},
                              r::T=1.0)
    (nx,mx)    = size(X)
    (nz,mz)    = size(Z)
    K          = zeros(T,nz, nx)
    for i=1:nz
        for j=1:nx
            K[i, j] = Φ(Z[i, :], X[j, :],r)
        end
    end
    K
end

## the learning algorithm: KPLS2 - multiple targets
function trainer{T<:AbstractFloat}(model::KPLSModel{T},
                                       X::AbstractArray{T},
                                       Y::AbstractArray{T};
                                       ignore_failures       = true,
                                       tol                   = 1e-6,
                                       max_iterations        = 250
                                       )

    kernel,width        = model.kernel,model.width
    Y                   = Y[:,:]
    model.ntargetcols   = size(Y,2)
    nfactors            = model.nfactors

    n = size(X,1)

    Tj = zeros(T,n, nfactors)
    Q  = zeros(T,model.ntargetcols, nfactors)
    U  = zeros(T,n, nfactors)
    P  = zeros(T,n, nfactors)

    K = ΦΦ(X,width)

    # centralize kernel
    c = eye(n) -  ones(Float64,n,n)*1.0/n
    K = c * K * c

    K_j = K[:,:]

    for j=1:nfactors

        u = Y[:,1]

        iteration_count  = 0
        iteration_change = tol * 10.0
        local w,t,q,old_u
        while iteration_count < max_iterations && iteration_change > tol

            w = K * u
            t = w / norm(w, 2)

            q = Y' * t

            old_u = u
            u = Y * q
            u /= norm(u, 2)
            iteration_change = norm(u - old_u)
            iteration_count += 1
        end

        if iteration_count >= max_iterations
            if ignore_failures
                nfactors = j
                warn("KPLS: Found with less factors. Overall factors = $(nfactors)")
                break
            else
                error("KPLS: failed to converge for component: $(nfactors+1)")
            end
        end
        Tj[:, j] = t
        Q[:, j] = q
        U[:, j] = u

        P[:, j]  = (K_j' * w) / (w'w)
        deflator = eye(n) - t'*t
        K_j      = deflator * K_j * deflator
        Y        = Y - t * q'

    end
    # If iteration stopped early because of failed convergence, only
    # the actual components will be copied

    Tj = Tj[:, 1:nfactors]
    Q = Q[:, 1:nfactors]
    U = U[:, 1:nfactors]
    #P = P[:, 1:nfactors]

    model.nfactors = nfactors
    model.X        = X # unfortunately it is necessary on the prediction phase
    model.K        = K # unfortunately it is necessary on the prediction phase
    try
       model.B        = U * inv(Tj' * K * U) * Q'
    catch
       error("KPLS: Not able to compute inverse.
             Maybe nfactors is greater than ncols of input data (X) or
             this matrix is not invertible.
             ")
    end

    return model

end

function predictor{T<:AbstractFloat}(model::KPLSModel{T},
                                         Z::AbstractArray{T})

    X,K,B,w      = model.X,model.K,model.B,model.width
    (nx,mx)    = size(X)
    (nz,mz)    = size(Z)

    Kt         = ΦΦ(X,Z,w) # kernel matrix

    # centralize
    c = (1.0 / nx) * ones(T,nz,nx)

    Kt = (Kt - c * K) * (eye(T,nx) - (1.0 / nx) * ones(T,nx,nx))

    Y = Kt * B

    return Y

end
