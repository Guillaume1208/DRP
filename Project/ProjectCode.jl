using LinearAlgebra
using FiniteDifferences
using Symbolics
using BenchmarkTools 
using ForwardDiff
using Zygote

#First, we need to implement the kernel function as it is a part of the Energy function
function kernel(alpha, lambda)
    #Check that lambda has good values:
    if !(0 <= lambda <= 3)
        throw(ArgumentError("lambda must be in the range (0, 3)"))
    end
    if alpha < 0
        throw(ArgumentError("Alpha must be greater than or equal to 0"))
    end 
    #Now, we implement the function kernel, as given in the notes.
    return r -> (1/alpha)*r^alpha + (1/lambda)*r^(-lambda)
end 

#Implement the distance function:
function dist(Xi, Xj)
    return sqrt(sum((Xi[i]-Xj[i])^2 for i in 1:3))
end

#Implement the energy function:
function energy(X, alpha, lambda)
    #Make Kernel and get number of points
    K = kernel(alpha, lambda)
    N = length(X)
    #Get all the values of distances
    r = [dist(X[i], X[j]) for i in 1:N for j in 1:N if i != j]
    #Get the sum
    E = sum(K(r))

    return (1/(2*N))*E
end