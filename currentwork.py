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



#x = np.matrix(((4.0, 12.0, -16.0), (12.0, 37.0, -43.0), (-16, -43, 98)))


def SolveSPD(A, b):
	L = CholeskyFactorization(A)
	#print("A")
	#print(A)
	LT = L.transpose()
	y = np.dot(LA.inv(L), b)
	#print("y")
	#print(y)
	return np.dot(LA.inv(LT), y)


def RelativeResidual(A, b, x):
	#print(A)
	#print(b)
	#print(x)
	return (LA.norm(b-(np.dot(A, x))))/LA.norm(b)

m = 100
n = 100
B = np.random.randn(m,n)
A = np.dot(B, B.transpose())

#print("original matrix")
#print(A)
#y = CholeskyFactorization(A)
#print("L")
#print(y)
#print("LT")
#print(y.transpose())
#print("LLT")
#print(np.dot(y, y.transpose()))

b = np.random.randn(m, 1)

#print(b)

r = RelativeResidual(A, b, SolveSPD(A, b))

print(r)



