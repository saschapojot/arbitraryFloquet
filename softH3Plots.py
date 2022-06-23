import numpy as np
from datetime import datetime
from scipy.sparse import csc_matrix
# import mpmath
# from mpmath import mp
# mp.dpd=20
from scipy.sparse.linalg import expm
# from scipy.linalg import expm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pathlib import Path

#this script plots wavefunction pattern for soft edge during H3, and plots the
#configuration of each sublattice, for one value of slope

J1=0.5*np.pi
J3=0.2*np.pi
J2Coef=0.5
J2=J2Coef*np.pi
M = 1
sigma3 = np.array([[1, 0], [0, -1]])
sigma2 = np.array([[0, -1j], [1j, 0]])
sigma1 = np.array([[0, 1], [1, 0]])
sigma0=np.eye(2,dtype=complex)

N1 = 200
N2 = 200
dk = 2 * np.pi / N2


length=10
leftStartingPoint=length
rightStartingPoint=N1-length-1

slopes=np.arange(0,0.1,0.7)#symmetrical slopes
def vec2Mat(vec):
    """

    :param vec: length=2*N1, in momentum space along k2
    :return: matrix representation of wavefunction on each sublattice
    """
    mat=np.zeros((2*N1,N2),dtype=float)
    for n2 in range(0,N2):
        for j in range(0,len(vec)):
            mat[j,n2]=1/N2*np.abs(vec[j])**2
    return mat


tTotStart=datetime.now()
for slVal in slopes:
    leftSlope = slVal
    rightSlope = slVal

    def V(n1):
        """

        :param n1: 0,1,...,N1-1
        :return: soft boundary potential, linear
        """
        if n1 <= leftStartingPoint:
            return leftSlope * np.abs(n1 - leftStartingPoint)
        elif n1 > leftStartingPoint and n1 < rightStartingPoint:
            return 0
        else:
            return rightSlope * np.abs(n1 - rightStartingPoint)# V values on each n1

    diagV = [V(n1) for n1 in range(0, N1)]
    # assemble H10
    S1 = np.zeros((N1, N1), dtype=complex)
    for n1 in range(0, N1 - 1):
        S1[n1, n1 + 1] = 1
        S1[n1 + 1, n1] = -1

    S1 *= 3 * J1 / (2 * 1j)
    H10 = np.kron(S1, sigma1)  # + np.kron(np.diag(diagV), sigma0)
    H10Sparse = csc_matrix(-1j * 1 / 3 * H10)
    U1Mat = expm(H10Sparse)

    def U2(b):
        k2 = b * dk
        S2 = np.eye(N1, dtype=complex)
        H20 = 3 * J2 * np.sin(k2) * np.kron(S2, sigma2)# + np.kron(np.diag(diagV), sigma0)
        H20Sparse = csc_matrix(-1j * 1 / 3 *H20)
        return expm( H20Sparse)

    def U3(b):
        k2 = b * dk
        S3 = np.eye(N1, dtype=complex) * (2 * M + 2 * np.cos(k2))
        for n1 in range(0, N1 - 1):
            S3[n1, n1 + 1] = 1
            S3[n1 + 1, n1] = 1
        S3 *= 3 * J3 / 2
        H30 = np.kron(S3, sigma3) + np.kron(np.diag(diagV), sigma0)
        H30Sparse = csc_matrix(-1j * 1 / 3 * H30)
        return expm(H30Sparse)

    def U(b):
        return (U3(b) @ U2(b) @ U1Mat).toarray()

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
        middlePhases = []
        middleVecs = []
        for j in range(0, len(phases)):
            kind = categorizeVec(vecs[j])
            if kind == 2:
                leftPhases.append(phases[j])
                leftVecs.append(vecs[j])
            elif kind == 1:
                rightPhases.append(phases[j])
                rightVecs.append(vecs[j])

            else:
                middlePhases.append(phases[j])
                middleVecs.append(vecs[j])
        # return [b,leftPhases,leftVecs,rightPhases,rightVecs,middlePhases,middleVecs]
        return [b, leftPhases, rightPhases, middlePhases, leftVecs, rightVecs, middleVecs]


    tPartitionStart = datetime.now()
    threaNum = 48
    pool0 = Pool(threaNum)
    retAll = pool0.map(partitionData, range(0, N2))
    tPartitionEnd = datetime.now()

    print("computation time for " + str(slVal) + ": ", tPartitionEnd - tPartitionStart)

    # data serialization
    pltLeftk2 = []
    pltRightk2 = []
    pltMiddlek2 = []
    pltLeftPhases = []
    pltRightPhases = []
    pltMiddlePhases = []
    pltLeftVecs=[]
    pltRightVecs=[]
    pltMiddleVecs=[]
    for itemTmp in retAll:
        b, leftPhasesTmp, rightPhasesTmp, middlePhasesTmp, leftVecsTmp, rightVecsTmp, middleVecsTmp = itemTmp
        k2=b/N2
        if len(leftPhasesTmp)>0:
            for j in range(0,len(leftPhasesTmp)):
                pltLeftk2.append(k2)
                pltLeftPhases.append(leftPhasesTmp[j]/np.pi)
                pltLeftVecs.append(leftVecsTmp[j])
        if len(rightPhasesTmp)>0:
            for j in range(0,len(rightPhasesTmp)):
                pltRightk2.append(k2)
                pltRightPhases.append(rightPhasesTmp[j]/np.pi)
                pltRightVecs.append(rightVecsTmp[j])

        if len(middlePhasesTmp)>0:
            for j in range(0,len(middlePhasesTmp)):
                pltMiddlek2.append(k2)
                pltMiddlePhases.append(middlePhasesTmp[j]/np.pi)
                pltMiddlek2.append(middleVecsTmp[j])

    outDir="./softH3plts/"+str(slVal)+"/"
    tPltLeftStart=datetime.now()
    if len(pltLeftk2)>0:
        outLeft=outDir+"left/"
        Path(outLeft).mkdir(parents=True,exist_ok=True)
        for j in range(0,len(pltLeftk2)):
            fig,ax=plt.subplots(figsize=(100,100))
            vecTmp=pltLeftVecs[j]
            outMatTmp=vec2Mat(vecTmp)
            # imTmp=plt.imshow(outMatTmp,cmap=plt.cm.RdBu,interpolation="bilinear")
            # plt.colorbar(imTmp)

            imTmp=plt.pcolormesh(outMatTmp, edgecolors='k', linewidth=1,cmap=plt.cm.RdBu)
            cbar=fig.colorbar(imTmp)
            cbar.ax.tick_params(labelsize=150)
            plt.xlabel("$n_{2}$",fontsize=200)
            plt.ylabel("$n_{1}$",fontsize=200)
            phLeftRounded=round(pltLeftPhases[j],3)
            plt.title("left edge, $k_{2}=$"+str(pltLeftk2[j])+"$\pi$, phase="+str(phLeftRounded)+"$\pi$",fontsize=200)
            plt.savefig(outLeft+"left"+str(j)+".png")
            plt.close()
    tPltLeftEnd=datetime.now()
    print("plotting left time: ",tPltLeftEnd-tPltLeftStart)
    tPltRightStart=datetime.now()
    if len(pltRightk2)>0:
        outRight=outDir+"right/"
        Path(outRight).mkdir(parents=True,exist_ok=True)
        for j in range(0,len(pltRightk2)):
            fig,ax=plt.subplots(figsize=(100,100))
            vecTmp=pltRightVecs[j]
            outMatTmp=vec2Mat(vecTmp)
            imTmp = plt.pcolormesh(outMatTmp, edgecolors='k', linewidth=1, cmap=plt.cm.RdBu)
            cbar = fig.colorbar(imTmp)
            cbar.ax.tick_params(labelsize=150)
            plt.xlabel("$n_{2}$", fontsize=200)
            plt.ylabel("$n_{1}$", fontsize=200)
            phRightRounded = round(pltRightPhases[j], 3)
            plt.title("right edge, $k_{2}=$" + str(pltRightk2[j]) + "$\pi$, phase=" + str(phRightRounded) + "$\pi$",
                      fontsize=200)
            plt.savefig(outRight + "left" + str(j) + ".png")
            plt.close()
    tPltRightEnd=datetime.now()
    print("plotting right time: ",tPltRightEnd-tPltRightStart)