
# A gaussian kernel function
@inline function Φ(x::Vector{Float64},
                   y::Vector{Float64};
                   r::Float64=1.0)
    norm  = 1.0 / sqrt(2π*r)
    scale = 1.0 / (2r^2)
    return norm * exp(-scale * sum((x.-y).^2))
end
# A kernel matrix
function ΦΦ(X::Matrix{Float64})
    n = size(X,1)
    K = zeros(n,n)
    for i=1:n
        for j=1:i
            K[i, j] = Φ(X[i, :], X[j, :])
            K[j, i] = K[i, j]
        end
        K[i, i] = Φ(X[i, :], X[i, :])
    end
    K
end
