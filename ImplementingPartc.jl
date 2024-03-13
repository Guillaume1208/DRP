using LinearAlgebra
using FiniteDifferences
using Symbolics
using BenchmarkTools 
using ForwardDiff

#Make some epsilon variables to be the small δp

eps = 1e-8 #This is the suggested value in the lectures

#Need a way to create the tridiagonal matrix A
function create_tridiagonal_matrix(a, p)

    #Now, we create the matrix a
    A = SymTridiagonal(a, p)

    return A
end

#Found this tridiagonal solve algorithm online: (for efficient solve) it is called the Thomas algorithm http://www.industrial-maths.com/ms6021_thomas.pdf 
function tridiagonal_solve(A, b)
    #Extract the diagonals
    a = copy(diag(A))
    p = copy(diag(A, 1)) 

    n = length(a) #size of the matrix
    
    #initialize vectors for the solution
    x = similar(b)
    gamma = similar(p)
    rho = similar(b)

    #first values:
    gamma[1] = a[1]/p[1]
    rho[1] = b[1]/a[1]
    
    #Forward elimination
    for i = 2:n-1
        gamma[i] = p[i] / (a[i] - p[i-1] * gamma[i-1])
        rho[i] = (b[i] - p[i-1] * rho[i-1]) / (a[i] - p[i-1] * gamma[i-1])
    end
    rho[n] = (b[n] - p[n-1] * rho[n-1]) / (a[n] - p[n-1] * gamma[n-1])

    #Backward substitution
    x[n] = rho[n]
    for i = n-1:-1:1
        x[i] = rho[i] - gamma[i]*x[i+1]
    end

    return x
end

#Now, we implement the function f of the problem:
function f(a, b, c, p)
    #First, we compute the matrix A
    A = create_tridiagonal_matrix(a, p)

    #Then, we compute A^-1 * b, we can do that since c^TA^1b is a scalar, we can commute it as we want
    #A_inv_b = tridiagonal_solve(A, b)
    A_inv_b = A\b

    #Compute c^T * A_inv_b
    product = dot(c, A_inv_b)

    #Square it
    result = product^2

    return result
end

#Now, we implement how we can compute v
function computeV(A, c, x)
    #Then, like we said in part b), we can solve doing v = A^-1[-2(c^Tx)c], which is a single tridiagonal solve
    block = -2((c'x)c)
    #A^-1(block)
    #v = tridiagonal_solve(A, block)
    v = A\block

    return v
end

#Now the procedure to find gradient f: ∂f/∂pk = vk/xk+1 + vk+1/xk
function gradF(a, b, c, p)
    #Compute the matrix A
    A = create_tridiagonal_matrix(a, p)

    #Then, we compute x, like in the question assumption: x = A^1b
    #x = tridiagonal_solve(A, b)
    x = A\b

    #Then we compute v 
    v = computeV(A, c, x)

    grad = similar(p)
    n = length(p)

    for k = 1:n
        grad[k] = v[k]*x[k+1] + v[k+1]*x[k]
    end

    return grad
end
