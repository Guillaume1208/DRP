using LinearAlgebra
using FiniteDifferences
using Symbolics
using BenchmarkTools 
using ForwardDiff
import Zygote

#Do it using autodiff, NOT FINITE FiniteDifferences

#Implement the function
function fM(f, M)
    F = eigen(M)
    return F.vectors * Diagonal(f.(F.values)) / F.vectors
end
