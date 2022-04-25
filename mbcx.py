import numpy as np
from datetime import datetime
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm
import matplotlib.pyplot as plt
from multiprocessing import Pool
# from scipy.linalg import expm


J1 = 0.5 * np.pi
J3 = 0.2 * np.pi
J2 = 6 * np.pi
M = 1
sigma3 = np.array([[1, 0], [0, -1]])
sigma2 = np.array([[0, -1j], [1j, 0]])
sigma1 = np.array([[0, 1], [1, 0]])

N1 = 200
N2 = 200
dk = 2 * np.pi / N2

# spatial part of H1
S1 = np.zeros((N1, N1), dtype=complex)
for n1 in range(0, N1 - 1):
    S1[n1, n1 + 1] = 1
    S1[n1 + 1, n1] = -1

S1 *= 3 * J1 / (2 * 1j)
H1 = np.kron(S1, sigma1)

H1Sparse = csc_matrix(H1)
U1Mat = expm(-1j * 1 / 3 * H1Sparse)

U1MatDense = U1Mat.toarray()


def U2(b):
    # Spatial part of H2
    k2 = b * dk
    S2 = np.eye(N1, dtype=complex)
    H2 = 3 * J2 * np.sin(k2) * np.kron(S2, sigma2)
    H2Sparse = csc_matrix(H2)
    return expm(-1j * 1 / 3 * H2Sparse)


def U3(b):
    k2 = b * dk
    # spatial part of H3
    S3 = np.eye(N1, dtype=complex) * (2 * M + 2 * np.cos(k2))
    for n1 in range(0, N1 - 1):
        S3[n1, n1 + 1] = 1
        S3[n1 + 1, n1] = 1
    S3 *= 3 * J3 / 2
    H3 = np.kron(S3, sigma3)
    H3Sparse = csc_matrix(H3)
    return expm(-1j * 1 / 3 * H3Sparse)


def U(b):
    return (U3(b).toarray()) @ (U2(b).toarray()) @ U1MatDense


def eigValsAndVecs(b):
    UMat = U(b)
    vals, vecs = np.linalg.eig(UMat)
    phases = np.angle(vals)
    inds = np.argsort(phases)

    phasesSorted = [phases[ind] for ind in inds]
    vecsSorted = [vecs[:, ind] for ind in inds]

    return [phasesSorted, vecsSorted]


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
plt.scatter(pltLeftk2,pltLeftPhases,color="green",s=sVal,label="left")
plt.scatter(pltRightk2,pltRightPhases,color="red",s=sVal,label="right")
plt.scatter(pltMiddlek2,pltMiddlePhases,color="black",s=sVal,label="bulk")
plt.legend()
lgnd =ax.legend(loc='upper right',fontsize=ftSize)
plt.xlabel("$k_{y}/\pi$")
plt.ylabel("$\epsilon/\pi$")
plt.title("$J_{2}/\pi=$"+str(J2/np.pi))

plt.hlines(y=-1,xmin=0,xmax=2,linewidth=0.5,color="blue",linestyles="--")

plt.hlines(y=0,xmin=0,xmax=2,linewidth=0.5,color="blue",linestyles="--")

plt.hlines(y=1,xmin=0,xmax=2,linewidth=0.5,color="blue",linestyles="--")




plt.savefig("J2"+str(J2/np.pi)+"mbcx.png")

tPltEnd=datetime.now()
print("plt time: ",tPltEnd-tPltStart)