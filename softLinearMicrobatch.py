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
#this script computes spectrum along n2 with soft boundary along x, and MBCX, using np, repeat for different values of slopes

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

mu=0.003
###########################################exp
# def V(n1):
#     """
#
#     :param n1: 0,1,...,N1-1
#     :return: soft boundary potential, exp
#     """
#     # mu=0.0125
#     return np.exp(mu*np.abs(n1-N1/2))
#############################################

############################################half exp
# def V(n1):
#     """
#
#     :param n1: 0,1,...,N1-1
#     :return: soft boundary potential, exp
#     """
#     # mu=0.0125
#     return np.exp(mu*(n1-N1/2))
############################################

############################################linear slope
leftStartingPoint=2
rightStartingPoint=N1-3

slopes=np.arange(0,20,1)#symmetrical slopes

tTotStart=datetime.now()
for slVal in slopes:
    leftSlope=slVal
    rightSlope=slVal


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
            return rightSlope * np.abs(n1 - rightStartingPoint)


    # V values on each n1
    diagV = [V(n1) for n1 in range(0, N1)]

    # assemble H10
    S1 = np.zeros((N1, N1), dtype=complex)
    for n1 in range(0, N1 - 1):
        S1[n1, n1 + 1] = 1
        S1[n1 + 1, n1] = -1

    S1 *= 3 * J1 / (2 * 1j)

    H10 = np.kron(S1, sigma1) + np.kron(np.diag(diagV), sigma0)
    H10Sparse = csc_matrix(-1j * 1 / 3 *H10)
    U1Mat = expm( H10Sparse)


    def U2(b):
        k2 = b * dk
        S2 = np.eye(N1, dtype=complex)
        H20 = 3 * J2 * np.sin(k2) * np.kron(S2, sigma2) + np.kron(np.diag(diagV), sigma0)
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
    threaNum = 24
    pool0 = Pool(threaNum)
    retAll = pool0.map(partitionData, range(0, N2))
    tPartitionEnd = datetime.now()

    print("computation time for "+str(slVal)+": ", tPartitionEnd - tPartitionStart)

    # data serialization
    pltLeftk2 = []
    pltRightk2 = []
    pltMiddlek2 = []
    pltLeftPhases = []
    pltRightPhases = []
    pltMiddlePhases = []
    # pltLeftVecs=[]
    # tPltStart = datetime.now()

    for itemTmp in retAll:
        b, leftPhasesTmp, rightPhasesTmp, middlePhasesTmp, _, _, _ = itemTmp
        k2 = b * dk / np.pi

        if len(leftPhasesTmp) > 0:
            # pltLeftVecs.append(leftVecsTmp)
            for elem in leftPhasesTmp:
                pltLeftk2.append(k2)
                pltLeftPhases.append(elem / np.pi)
        if len(rightPhasesTmp) > 0:
            for elem in rightPhasesTmp:
                pltRightk2.append(k2)
                pltRightPhases.append(elem / np.pi)
        if len(middlePhasesTmp) > 0:
            for elem in middlePhasesTmp:
                pltMiddlek2.append(k2)
                pltMiddlePhases.append(elem / np.pi)

    pltLeftPhases = np.array(pltLeftPhases)
    pltRightPhases = np.array(pltRightPhases)
    pltMiddlePhases = np.array(pltMiddlePhases)

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    sVal = 5
    ftSize = 5
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(pltLeftk2, pltLeftPhases, color="green", s=sVal * 2, label="upper", alpha=0.4)
    plt.scatter(pltRightk2, pltRightPhases, color="red", s=sVal * 2, label="lower", alpha=0.4)
    plt.scatter(pltMiddlek2, pltMiddlePhases, color="black", s=sVal, label="bulk", alpha=0.2)
    plt.legend()
    lgnd = ax.legend(loc='upper right', fontsize=ftSize)
    plt.xlabel("$k_{y}/\pi$")
    plt.ylabel("$\epsilon/\pi$")
    # plt.title("$\mu=$"+str(mu)+", $J_{2}/\pi=$"+str(J2Coef))
    leftSc = "{:.4e}".format(leftSlope)
    rightSc = "{:.4e}".format(rightSlope)
    plt.title("upper slope=" + leftSc + ", lower slope=" + rightSc + ", $J_{2}/\pi=$" + str(J2Coef))
    plt.hlines(y=-1, xmin=0, xmax=2, linewidth=0.5, color="blue", linestyles="--")
    #
    # plt.hlines(y=0,xmin=0,xmax=2,linewidth=0.5,color="blue",linestyles="--")
    #
    plt.hlines(y=1, xmin=0, xmax=2, linewidth=0.5, color="blue", linestyles="--")

    outDir = "./soft/lin/J2" + str(J2Coef) + "/slopeStart"+str(slopes[0])+"slopeEnd"+str(slopes[-1])+"/"
    Path(outDir).mkdir(parents=True, exist_ok=True)
    # plt.savefig(outDir+"mu"+str(mu)+"J2"+str(J2Coef)+"SoftExpMbcx.png")
    plt.savefig(
        outDir + "leftSlope" + str(leftSlope) + "rightSlope" + str(rightSlope) + "J2" + str(J2Coef) + "soft.png")

    # tPltEnd = datetime.now()
    # print("plt time: ", tPltEnd - tPltStart)


tTotEnd=datetime.now()

print("Total time: ",tTotEnd-tTotStart)