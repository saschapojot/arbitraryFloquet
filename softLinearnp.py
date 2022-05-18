import numpy as np
from datetime import datetime
from scipy.sparse import csc_matrix
import mpmath
from mpmath import mp
mp.dpd=20
# from scipy.sparse.linalg import expm
from scipy.linalg import expm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pathlib import Path
#this script computes spectrum along n2 with soft boundary along x, and MBCX

J1=0.5*np.pi
J3=0.2*np.pi
J2Coef=0.5
J2=J2Coef*np.pi
M = 1
sigma3 = np.array([[1, 0], [0, -1]])
sigma2 = np.array([[0, -1j], [1j, 0]])
sigma1 = np.array([[0, 1], [1, 0]])
sigma0=np.eye(2,dtype=complex)

N1 = 100
N2 = 100
dk = 2 * np.pi / N2

mu=0.1
def V(n1):
    """

    :param n1: 0,1,...,N1-1
    :return: soft boundary
    """
    # mu=0.0125
    return mpmath.exp(mu*np.abs(n1-N1/2))

#V values on each n1
diagV=[V(n1) for n1 in range(0,N1)]
#assemble H10


S1 = np.zeros((N1, N1), dtype=complex)
for n1 in range(0, N1 - 1):
    S1[n1, n1 + 1] = 1
    S1[n1 + 1, n1] = -1

S1 *= 3 * J1 / (2 * 1j)

H10=np.kron(S1, sigma1)+np.kron(np.diag(diagV),sigma0)

def count0(mat):
    """

    :param mat:
    :return: percentage of 0 in mat
    """
    nRow, nCol=mat.shape
    nCount=0
    for j in range(0,nRow):
        rowTmpAbs=list(np.abs(mat[j,:]))
        nCount+=rowTmpAbs.count(0)
    return nCount/(nRow*nCol)



# H1Sparse=csc_matrix(H10)
U1Mat=expm(-1j*1/3*H10)


# U1MatDense=U1Mat.toarray()
# print(count0(U1MatDense))
# we can see that U1Mat is sparse


def U2(b):
    k2=b*dk
    S2=np.eye(N2,dtype=complex)
    H20 = 3 * J2 * np.sin(k2) * np.kron(S2, sigma2)+np.kron(np.diag(diagV),sigma0)
    # H2Sparse=csc_matrix(H2)
    return expm(-1j*1/3*H20)

#
# print(count0(U2(10).toarray()))
# we can see that U2 is sparse


def U3(b):
    k2=b*dk
    S3 = np.eye(N1, dtype=complex) * (2 * M + 2 * np.cos(k2))
    for n1 in range(0, N1 - 1):
        S3[n1, n1 + 1] = 1
        S3[n1 + 1, n1] = 1
    S3 *= 3 * J3 / 2
    H30 = np.kron(S3, sigma3)+np.kron(np.diag(diagV),sigma0)
    # H3Sparse=csc_matrix(H3)
    return expm(-1j*1/3*H30)


# print(count0(U3(20).toarray()))
# we can see that U3 is sparse


def U(b):
    return U3(b)@U2(b)@U1Mat

# print(count0(U(13).toarray()))
# we can see that U is dense



def eigValsAndVecs(b):
    UMat=U(b)
    vals,vecs=np.linalg.eig(UMat)
    phases=np.angle(vals)
    inds=np.argsort(phases)
    phasesSorted=[phases[ind] for ind in inds]
    vecsSorted=[vecs[:,ind] for ind in inds]

    return [phasesSorted,vecsSorted]

localLengthRatio = 0.1
localLength = int(2 * N1 * localLengthRatio)
weight = 0.7

def categorizeVec(vec):
    """

    :param vec:
    :return: 2: left
             1: right
            0: middle
    """

    leftVec = vec[:localLength]
    rightVec = vec[-localLength:]

    leftNorm = np.linalg.norm(leftVec, 2)
    rightNorm = np.linalg.norm(rightVec, 2)

    totalNorm = np.linalg.norm(vec, 2)

    if leftNorm / totalNorm >= weight:
        return 2
    elif rightNorm / totalNorm >= weight:
        return 1
    else:
        return 0



def partitionData(b):
    phases, vecs = eigValsAndVecs(b)

    leftPhases = []
    leftVecs = []
    rightPhases = []
    rightVecs = []
    middlePhases=[]
    middleVecs=[]
    for j in range(0, len(phases)):
        kind = categorizeVec(vecs[j])
        if kind==2:
            leftPhases.append(phases[j])
            leftVecs.append(vecs[j])
        elif kind==1:
            rightPhases.append(phases[j])
            rightVecs.append(vecs[j])

        else:
            middlePhases.append(phases[j])
            middleVecs.append(vecs[j])
        # return [b,leftPhases,leftVecs,rightPhases,rightVecs,middlePhases,middleVecs]
    return [b,leftPhases,rightPhases,middlePhases]


tPartitionStart=datetime.now()
threaNum=24
pool0=Pool(threaNum)
retAll=pool0.map(partitionData,range(0,N2))
tPartitionEnd=datetime.now()

print("computation time: ",tPartitionEnd-tPartitionStart)

#data serialization
pltLeftk2=[]
pltRightk2=[]
pltMiddlek2=[]
pltLeftPhases=[]
pltRightPhases=[]
pltMiddlePhases=[]
tPltStart=datetime.now()
for itemTmp in retAll:
    b,leftPhasesTmp,rightPhasesTmp,middlePhasesTmp=itemTmp
    k2=b*dk/np.pi

    if len(leftPhasesTmp)>0:
        for elem in leftPhasesTmp:
            pltLeftk2.append(k2)
            pltLeftPhases.append(elem/np.pi)
    if len(rightPhasesTmp)>0:
        for elem in rightPhasesTmp:
            pltRightk2.append(k2)
            pltRightPhases.append(elem/np.pi)
    if len(middlePhasesTmp)>0:
        for elem in middlePhasesTmp:
            pltMiddlek2.append(k2)
            pltMiddlePhases.append(elem/np.pi)


sVal=5
ftSize=5
fig=plt.figure()
ax=fig.add_subplot(111)
plt.scatter(pltLeftk2,pltLeftPhases,color="green",s=sVal,label="upper")
plt.scatter(pltRightk2,pltRightPhases,color="red",s=sVal,label="down")
plt.scatter(pltMiddlek2,pltMiddlePhases,color="black",s=sVal,label="bulk",alpha=0.2)
plt.legend()
lgnd =ax.legend(loc='upper right',fontsize=ftSize)
plt.xlabel("$k_{y}/\pi$")
plt.ylabel("$\epsilon/\pi$")
plt.title("$\mu=$"+str(mu)+", $J_{2}/\pi=$"+str(J2/np.pi))

# plt.hlines(y=-1,xmin=0,xmax=2,linewidth=0.5,color="blue",linestyles="--")
#
# plt.hlines(y=0,xmin=0,xmax=2,linewidth=0.5,color="blue",linestyles="--")
#
# plt.hlines(y=1,xmin=0,xmax=2,linewidth=0.5,color="blue",linestyles="--")

outDir="./soft/exp/J2"+str(J2Coef)+"/"
Path(outDir).mkdir(parents=True,exist_ok=True)
plt.savefig(outDir+"mu"+str(mu)+"J2"+str(J2Coef)+"SoftExpMbcx.png")

tPltEnd=datetime.now()
print("plt time: ",tPltEnd-tPltStart)