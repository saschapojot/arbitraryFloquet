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

#this script computes all eigenvectors for the linear problem

J1 = 0.5 * np.pi
J3 = 0.2 * np.pi
J2Coef=0.5
J2 = J2Coef * np.pi
M = 1
sigma3 = np.array([[1, 0], [0, -1]])
sigma2 = np.array([[0, -1j], [1j, 0]])
sigma1 = np.array([[0, 1], [1, 0]])

N1 = 30
N2 =30
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

    return [b,phasesSorted, vecsSorted]


tEigStart=datetime.now()
threadNum=24
pool0=Pool(threadNum)

ret0=pool0.map(eigValsAndVecs,range(0,N2))
tEigEnd=datetime.now()
ret0=sorted(ret0,key=lambda elem: elem[0])
print(f"computation time: {tEigEnd-tEigStart}")

outTable=[]
for elem in ret0:
    b=elem[0]
    phases=elem[1]
    vecs=elem[2]
    for j in range(0,len(phases)):
        oneRow=[N2,b,phases[j]]

        oneRow.extend(vecs[j])
        outTable.append(oneRow)

outDtFrm=pd.DataFrame(data=outTable)

outDir="./linearEigs/"
Path(outDir).mkdir(parents=True,exist_ok=True)

outDtFrm.to_csv(outDir+f"AllJ2{J2Coef}.csv", index=False,header=False)