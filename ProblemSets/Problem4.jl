using LinearAlgebra
using FiniteDifferences
using Symbolics
using BenchmarkTools 
using ForwardDiff
import Zygote

#Make a relative error function: ReError = ||a-b||/||b||
function reError(a, b)
    return norm(a-b)/norm(b)
end
#Do it using autodiff, NOT FINITE FiniteDifferences

#Implement the function:
function fM(f, M)
    F = eigen(M)
    return F.vectors * Diagonal(f.(F.values)) / F.vectors
end
#Because of Zygote, I have to do something like: Jexp = Zygote.jacobian(M -> fM(exp, M), M + [0 0 1e-13; 0 0 0; 0 0 0])[1]
# "It turns out that this is a bug in Zygote for computing derivatives of eigenvalues 
#when the matrix happen to be real-symmetric/Hermitian."

# Initialize M:
M = [0 1 4; 1 0 1; 4 1 0]

#So, to find the Jacobian determinant for f being exp:
JExp = Zygote.jacobian(M -> fM(exp, M), M + [0 0 1e-13; 0 0 0; 0 0 0])[1]

#To find the Jacobian determinant for f being ^2:
Jsquare = Zygote.jacobian(M -> M^2, M)[1]

#To find the jacobian determinant for f being sin:
Jsin = Zygote.jacobian(M -> fM(sin, M), M + [0 0 1e-13; 0 0 0; 0 0 0])[1]

#Part ii) We implement the analytical formula:
