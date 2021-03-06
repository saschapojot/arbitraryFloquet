import numpy as np
from datetime import datetime
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pathlib import Path
import shutil
import pandas as pd
import os
# from scipy.linalg import expm

J1 = 0.5 * np.pi
J3 = 0.2 * np.pi
J2Coef=0.5
J2 = J2Coef * np.pi
M = 1
sigma3 = np.array([[1, 0], [0, -1]])
sigma2 = np.array([[0, -1j], [1j, 0]])
sigma1 = np.array([[0, 1], [1, 0]])

N1 = 40
N2 =40
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
    return [b,leftPhases,rightPhases,middlePhases,leftVecs,rightVecs,middleVecs]


tPartitionStart=datetime.now()
threaNum=24
pool0=Pool(threaNum)
retAll=pool0.map(partitionData,range(0,N2))
tPartitionEnd=datetime.now()

print("computation time: ",tPartitionEnd-tPartitionStart)

j2indLeftPhaseVec=[]
j2indRightPhaseVec=[]
for itemTmp in retAll:
    b,leftPhasesTmp,rightPhasesTmp,middlePhasesTmp,\
        leftVecsTmp,rightVecsTmp,middleVecsTmp=itemTmp
    if len(leftPhasesTmp)>0:
        for j in range(0,len(leftPhasesTmp)):
            j2indLeftPhaseVec.append([b,leftPhasesTmp[j],leftVecsTmp[j]])
    if len(rightPhasesTmp)>0:
        for j in range(0,len(rightPhasesTmp)):
            j2indRightPhaseVec.append([b,rightPhasesTmp[j],rightVecsTmp[j]])


#plot 1 figure
# bTmp,phaseTmp,vecTmp=j2indLeftPhaseVec[1]
#
# outMat=np.zeros((N1,N2),dtype=float)
# for n1 in range(0,N1):
#     for n2 in range(0,N2):
#         outMat[n1,n2]=1/N2*(np.abs(vecTmp[2*n1])**2+np.abs(vecTmp[2*n1+1])**2)
#
#
# ax=sns.heatmap(outMat)
# plt.savefig("hm.png")
outDirLeft="./leftLinearJ2"+str(J2Coef)+"N2"+str(N2)+"/"
outDirRight="./rightLinearJ2"+str(J2Coef)+"N2"+str(N2)+"/"
if os.path.exists(outDirLeft) and os.path.isdir(outDirLeft):
    shutil.rmtree(outDirLeft)
if os.path.exists(outDirRight) and os.path.isdir(outDirRight):
    shutil.rmtree(outDirRight)
Path(outDirLeft).mkdir(parents=True,exist_ok=True)
Path(outDirRight).mkdir(parents=True,exist_ok=True)
#plt left
outTableLeft=[]
for j in range(0,len(j2indLeftPhaseVec)):
    b,phaseTmp,vecLeftTmp=j2indLeftPhaseVec[j]
    oneRowTmp=[N2,b,phaseTmp]
    oneRowTmp.extend(vecLeftTmp)
    outTableLeft.append(oneRowTmp)
    outMat=np.zeros((N1,N2),dtype=float)
    for n1 in range(0,N1):
        for n2 in range(0,N2):
            outMat[n1,n2]=1/N2*(np.abs(vecLeftTmp[2*n1])**2+np.abs(vecLeftTmp[2*n1+1])**2)

    plt.figure()
    im=plt.imshow(outMat,cmap=plt.cm.RdBu,interpolation="bilinear")
    plt.colorbar(im)
    plt.savefig(outDirLeft+"leftb"+str(b)+"j"+str(j)+".png")
    plt.close()

leftDtfrm=pd.DataFrame(data=outTableLeft)
leftDtfrm.to_csv(outDirLeft+"leftVecsJ2"+str(J2Coef)+".csv",index=False,header=False)

#plt right
outTableRight=[]
for j in range(0,len(j2indRightPhaseVec)):
    b,phaseTmp,vecRightTmp=j2indRightPhaseVec[j]
    oneRowTmp=[N2,b,phaseTmp]
    oneRowTmp.extend(vecRightTmp)
    outTableRight.append(oneRowTmp)
    outMat=np.zeros((N1,N2),dtype=float)
    for n1 in range(0,N1):
        for n2 in range(0,N2):
            outMat[n1,n2]=1/N2*(np.abs(vecRightTmp[2*n1])**2+np.abs(vecRightTmp[2*n1+1])**2)
    plt.figure()
    im=plt.imshow(outMat,cmap=plt.cm.RdBu,interpolation="bilinear")
    plt.colorbar(im)
    plt.savefig(outDirRight+"rightb"+str(b)+"j"+str(j)+".png")
    plt.close()

rightDtfrm=pd.DataFrame(data=outTableRight)
rightDtfrm.to_csv(outDirRight+"rightVecsJ2"+str(J2Coef)+".csv",index=False,header=False)