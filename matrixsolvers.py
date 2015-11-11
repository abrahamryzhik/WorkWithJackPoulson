import numpy as np
from numpy import linalg as LA


def CholeskyFactorization(C):
	#assume that C is already symmetric and positive definite

    B = C.copy()
    m, n = B.shape

    for k in range(0, n):
        B[k, k] = np.sqrt(B[k, k])
        for i in range(k+1, n):
            B[i, k] = (B[i, k] / B[k, k])
        for j in range(k+1, n):
            for i in range(j, n):
                B[i, j] -= B[i, k] * B[j, k]

    for a in range(0, n):
        for b in range(a+1, n):
            B[a, b] = 0.0

    return B




#This uses inverses and has been replaced
def SolveSPD1(A, b):
    L = CholeskyFactorization(A)
    
    LT = L.transpose()
    y = np.dot(LA.inv(L), b)
    
    return np.dot(LA.inv(LT), y)

#Minimize the two-norm of Ax=b using AtA=Atb and Gaussian elimination of lower and upper triangular matrices
def SolveSPD(A, b):
    C = np.dot(A.transpose(), A)

    L = CholeskyFactorization(C)

    LT = L.transpose()

    y = np.dot(A.transpose(), b)

    z = LowerTriangularGaussianElimination(L, y)

    x = UpperTriangularGaussianElimination(LT, z)

    return x



def RelativeResidual(A, b, x):
    
    return (LA.norm(b-(np.dot(A, x))))/LA.norm(b)

def LowerTriangularGaussianElimination(L, y):
    x = y.copy()

    m, n = L.shape

    for k in range(0, n):
        sum = 0
        for i in range(0, k):
            sum += L[k, i]*x[i, 0]
        x[k, 0] = (y[k, 0]-sum)/L[k, k]   

    return x



def UpperTriangularGaussianElimination(L, y):
    x = y.copy()

    m, n = L.shape

    for k in range(0, n):
        sum = 0
        for i in range(0, k):
            sum += L[n-k-1, n-1-i]*x[n-i-1, 0]
        x[n-k-1, 0] = (y[n-k-1, 0]-sum)/L[n-k-1, n-k-1]

    return x




m = 1000
n = 1000
A = np.random.randn(m,n)


b = np.random.randn(m, 1)


r = RelativeResidual(A, b, SolveSPD(A, b))

print(r)






