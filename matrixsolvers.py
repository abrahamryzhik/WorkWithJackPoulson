import numpy as np
from numpy import linalg as LA
import scipy.linalg.blas as BLAS
from scipy.linalg import solve_triangular
import ctypes
from ctypes import byref, c_char, c_int, c_double

try:
    _blaslib = ctypes.cdll.LoadLibrary(np.core._dotblas.__file__) # @UndefinedVariable
    dsyrk = _blaslib.dsyrk_
    dsyr = _blaslib.dsyr_
    blas_available = True
except AttributeError as e:
     _blas_available = False
     warnings.warn("warning: caught this exception:" + str(e))

def DSYR_blas(A, x, alpha=1.):
    """
    Performs a symmetric rank-1 update operation:
    A <- A + alpha * np.dot(x,x.T)
    :param A: Symmetric NxN np.array
    :param x: Nx1 np.array
    :param alpha: scalar
    """
    N = c_int(A.shape[0])
    LDA = c_int(A.strides[1]/8)
    UPLO = c_char('l')
    ALPHA = c_double(alpha)
    A_ = A.ctypes.data_as(ctypes.c_void_p)
    x_ = x.ctypes.data_as(ctypes.c_void_p)
    INCX = c_int(1)
    dsyr(byref(UPLO), byref(N), byref(ALPHA),
            x_, byref(INCX), A_, byref(LDA))

def DSYRK_blas(A, C, alpha=1., beta=1.):
    """
    Performs a symmetric rank-k update operation:
    C <- alpha * np.dot(A,A.T) + beta C
    :param C: Symmetric NxN np.array
    :param A: Nxk np.array
    :param alpha: scalar
    :param beta: scalar
    """
    N = c_int(C.shape[0])
    K = c_int(A.shape[1])
    LDA = c_int(A.strides[1]/8)
    LDC = c_int(C.strides[1]/8)
    UPLO = c_char('l')
    TRANS = c_char('n')
    ALPHA = c_double(alpha)
    BETA = c_double(beta)
    A_ = A.ctypes.data_as(ctypes.c_void_p)
    C_ = C.ctypes.data_as(ctypes.c_void_p)
    dsyrk(byref(UPLO), byref(TRANS), byref(N), byref(K), byref(ALPHA),
            A_, byref(LDA), byref(BETA), C_, byref(LDC))

def UnblockedCholesky(A,explicitlyZero=False):
	#assume that A is already symmetric and positive definite
    m, n = A.shape

    for k in xrange(0, n):
        A[k, k] = np.sqrt(A[k, k])
        
        for i in xrange(k+1, n):
            A[i, k] = (A[i, k] / A[k, k])

        DSYR_blas(A[k+1:n,k+1:n],A[k+1:n,k],-1.)
        #BLAS.dsyr( -1., B[k+1:n,k], B[k+1:n,k+1:n], lower=1)
        #for j in xrange(k+1, n):
        #    for i in xrange(j, n):
        #        B[i, j] -= B[i, k] * B[j, k]

    if explicitlyZero:
        for a in xrange(0, n):
            for b in xrange(a+1, n):
                A[a, b] = 0.0

def Cholesky(A,blocksize=64,explicitlyZero=False):
    #assume that A is already symmetric and positive definite

    m, n = A.shape

    for k in xrange(0, n, blocksize):
        nb = min(blocksize,n-k)

        A11 = A[k:k+nb,k:k+nb]
        A21 = A[k+nb:n,k:k+nb]
        A22 = A[k+nb:n,k+nb:n]

        # Generalized square-root of B[k,k]
        UnblockedCholesky(A11)
        
        # Generalized division by B[k,k]
        # Hacked version which avoids the lack of a 'right' solve and instead
        # transposes the expression X := X inv(L)^T to X^T := inv(L) X^T
        XT = np.transpose(A21).copy(order='F')
        XT = solve_triangular(A11,XT,trans=0,lower=1)
        # Assign back to A21 (sorry, this is ugly)
        A[k+nb:n,k:k+nb] = np.transpose(XT).copy(order='F')

        DSYRK_blas(A21,A22,-1.,1.)

    if explicitlyZero:
        for a in xrange(0, n):
            for b in xrange(a+1, n):
                A[a, b] = 0.0


#This uses inverses and has been replaced
def SolveSPD1(A, b, blocksize=64):
    L = A.copy(order='F')
    Cholesky(L,blocksize)
    
    LT = L.transpose()
    y = np.dot(LA.inv(L), b)
    
    return np.dot(LA.inv(LT), y)

#Minimize the two-norm of Ax=b using AtA=Atb and Gaussian elimination of lower and upper triangular matrices
def SolveNormalEquations(A, b, blocksize=64):
    C = np.dot(A.transpose(), A)

    L = C.copy(order='F')
    Cholesky(L,blocksize)
    #UnblockedCholesky(L)

    #LT = L.transpose()

    y = np.dot(A.transpose(), b)

    #z = LowerTriangularSolve(L, y)

    #x = UpperTriangularSolve(LT, z)
    x = solve_triangular(L,y,lower=1,trans=0)
    x = solve_triangular(L,x,lower=1,trans=1)

    return x



def RelativeResidual(A, b, x):
    
    return (LA.norm(b-(np.dot(A, x))))/LA.norm(b)

def LowerTriangularSolve(L, y):
    x = y.copy(order='F')

    m, n = L.shape

    for k in xrange(0, n):
        sum = 0
        for i in xrange(0, k):
            sum += L[k, i]*x[i, 0]
        x[k, 0] = (y[k, 0]-sum)/L[k, k]   

    return x



def UpperTriangularSolve(L, y):
    x = y.copy(order='F')

    m, n = L.shape

    for k in xrange(0, n):
        sum = 0
        for i in xrange(0, k):
            sum += L[n-k-1, n-1-i]*x[n-i-1, 0]
        x[n-k-1, 0] = (y[n-k-1, 0]-sum)/L[n-k-1, n-k-1]

    return x




m = 1000
n = 1000
blocksize = 64
A = np.random.randn(m,n).copy(order='F')

b = np.random.randn(m, 1).copy(order='F')

r = RelativeResidual(A, b, SolveNormalEquations(A, b, blocksize))

print(r)






