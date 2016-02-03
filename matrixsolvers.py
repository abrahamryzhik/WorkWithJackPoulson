import numpy as np
from numpy import linalg as LA
import scipy.linalg.blas as BLAS
from scipy.linalg import solve_triangular
import ctypes
from ctypes import byref, c_char, c_int, c_double
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
from scipy.optimize import nnls
from random import randint

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
def SolveNormalEquations(A, B, blocksize=64):
    C = np.dot(A.transpose(), A)

    L = C.copy(order='F')
    Cholesky(L,blocksize)
    #UnblockedCholesky(L)

    #LT = L.transpose()

    Y = np.dot(A.transpose(), B)

    #z = LowerTriangularSolve(L, y)

    #print(L)
    #print(Y)

    #x = UpperTriangularSolve(LT, z)
    X = solve_triangular(L,Y,lower=1,trans=0)
    X = solve_triangular(L,X,lower=1,trans=1)

    return X





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




def ApproxNNLS1(A, B):
    #X = SolveNormalEquations(A, B)
    X = LA.lstsq(A,B)[0]
    m, n = X.shape

    for j in xrange(n):
        for i in xrange(m):
            if X[i, j] <= 0:
                X[i, j] = 0

    return X

def ApproxNNLS(A, B):

    m, n = A.shape
    #print("m: ", m)
    #print("n: ", n)

    m, numcols = B.shape
    #print("numcols: ", numcols)

    X = np.zeros((n, numcols)).copy(order='F')

    for j in xrange(numcols):
        x, rnorm = nnls(A, B[:,j])
        X[:,j] = x.copy(order='F')

    return X


def NMF(A, rank, numIts=20):

    m, n = A.shape

    F = np.random.uniform(low, high, (m, rank)).copy(order='F')
    
    AT = A.transpose().copy(order='F')

    #Use a better stoping criteria than 20 iterations
    for it in xrange(numIts):
        #Solve for G and fix F
        print(it)
        G = ApproxNNLS(F, A)


        #print(G)

        #Solve for F and fix G

        F = ApproxNNLS(G.transpose().copy(order='F'), AT).transpose().copy(order='F')
        #print(F)

    return F, G


def corsen(A):
    pass

def interpolate(A):
    pass

def shorten_digits(A, numSamples = 4000):
    m, n = A.shape

    if numSamples >= n:
        return A

    s = np.zeros((numSamples, 1), dtype = int)
    for i in xrange(numSamples):
        s[i] = randint(0,n-1)

    sUnique = np.unique(s)

    return A[:, sUnique]


dimension = 2
numcenters = 10
#centers = np.array([[1, 2], [2, 1], [2, 3]])



#A = np.random.uniform(low, high, (m, n)).copy(order='F')

result = np.array(list(csv.reader(open("train.csv","rb"),delimiter=','))).astype('float')
#digits = result[:,0]
#data = result[:, 1:]
A = result.transpose().copy(order='F')

A = shorten_digits(A)

m, n = A.shape

Dnum = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
D = [np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0))]
for i in xrange(n):
    Dnum[int(A[0, i])] += 1

for i in xrange(10):
    D[i] = np.zeros((28*28, Dnum[i]))



perm = np.argsort(A[0,:n])

A = A[:,perm]
Dindex = np.zeros((10,1))
for i in xrange(11):
    if i == 0:
        Dindex[i] = 0

    else:
        Dindex[i] = Dindex[i-1] + Dnum[i-1] 

for i in xrange(10):
    D[i] = A[1:m, Dindex[i]:Dindex[i+1]]



#dimensions of matrix that is to be clustered

A = D[5]


m, n = A.shape
blocksize = 64

#range of values initiated in A
low = 0
high = 10

#create perturbation with which to offset A from the cluster centers
perturbation = 5
gaussian = True
if gaussian:
    P = np.random.normal(0, perturbation, (m, n)).copy(order='F')
else: P = np.random.uniform(-perturbation, perturbation, (m, n)).copy(order='F')

#A=A+P
"""
for i in xrange(m):
    for j in xrange(n):
        if A[i,j] == 0:
            A[i,j] += P[i,j]
"""

print(A)
#create cluster centers that are copies of the first 3 columns of A, but with the first two rows fixed



#centers = A[:, :numcenters].copy() IMPORTANT LINE IF NOT READING DATA



#centers = A.copy(m, numcenters)
#centers = A[:, 0:numcenters]
#centers[0,:] = [1,2,1,3,4]
#centers[1,:] = [2,1,4,3,5]
#print(centers)

#change A so that every column is now a perturbed copy of a random cluster center
"""
for i in xrange(n):
    center_index = np.random.randint(numcenters)
    center = centers[:,center_index]
    A[:,i] = center + P[:,i]

"""


#b = np.random.randn(m, 1).copy(order='F')

#r = RelativeResidual(A, b, SolveNormalEquations(A, b, blocksize))

#print(r)

Anorm = LA.norm(A, ord='fro')

numIts=80

maxRank = 10

"""
for rank in xrange(1, maxRank):
    F,G = NMF(A, rank, numIts)
    #print(F)
    #print(G)
    E = A-np.dot(F, G)
    Enorm = LA.norm(E, ord='fro')
    print("||E||_F = %f" %Enorm)
    print(Enorm/Anorm)
"""

rank = numcenters
F,G = NMF(A, rank, numIts)
E = A-np.dot(F, G)
Enorm = LA.norm(E, ord='fro')
print("||E||_F = %f" %Enorm)
print(Enorm/Anorm)
print(F)

#Find matrix d that will be used to normalize G and find the correct cluster centers

dsums = np.zeros(rank)
dcount = np.zeros(rank)

for j in xrange(n):
    for i in xrange(rank):
        index =  np.argmax(G[:,j])
        dcount[index] += 1
        dsums[index] += G[index, j]

d = np.zeros((rank, rank))

for k in xrange(rank):
    d[k, k] = dsums[k]/dcount[k]


dinverse = d.copy()

for k in xrange(rank):
    dinverse[k, k] = 1/dinverse[k, k]

#change F and G by multiplying by d and d inverse so that F is the cluster centers and G is normalized

F = np.dot(F, d)
G = np.dot(dinverse, G)


print ("Find d and change F and G")

E = A-np.dot(F, G)
Enorm = LA.norm(E, ord='fro')
print("||E||_F = %f" %Enorm)
print(Enorm/Anorm)
print(F)



#which points from the matrix to show on the 2d graph
coord_1 = 500
coord_2 = 600

colors = ['r.', 'g.', 'y.', 'b.', 'm.', 'c.']
#plot each point of A, coloring it based on which cluster center it corresponds to using G

"""
for j in xrange(n):
    index = np.argmax(G[:,j]) % 6
    plt.plot(A[coord_1, j], A[coord_2, j], colors[index])
"""

indices = np.zeros((n, 1))

for j in xrange(n):
    distances = []
    
    for k in xrange(rank):
        distances.append(LA.norm(A[:,j]-F[:,k]))
    index = distances.index(min(distances))
    indices[j] = index

permutation = np.argsort(indices)

sorted_indices = indices[permutation]

sorted_A = A[:,permutation]

for i in xrange(20):
    digit = np.zeros((28,28))
    
for i in xrange(10):
    for k in xrange(28*28):
        row = int(k/28)
        column = k % 28
        digit[row,column] = A[k,i]


    #imgplot = plt.imshow(digit, cmap=cm.Greys_r)

    print('Sorted index: ', sorted_indices[i])

   # plt.show()






#First, sort data by index
#Second, find % of data with same actual digit 

    #plt.plot(A[coord_1, j], A[coord_2, j], colors[index])

"""
for k in xrange(rank):
    plt.plot(F[coord_1, k], F[coord_2, k], "ko")
"""
#firstdigit = F[:,0]



digit = np.zeros((28,28))
    
for i in xrange(10):
    for k in xrange(28*28):
        row = int(k/28)
        column = k % 28
        digit[row,column] = F[k,i]




    imgplot = plt.imshow(digit, cmap=cm.Greys_r)
    plt.show()







"""
#plt.plot(A[0,:], A[1,:], 'k.')
plt.axis([-1, 11, -1, 11])
plt.xlabel("Coordinate %d" %coord_1)
plt.ylabel("Coordinate %d" %coord_2)
plt.show()
"""