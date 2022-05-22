import numpy as np
from datetime import datetime
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm
import matplotlib.pyplot as plt
from multiprocessing import Pool
import torch

J1=0.5*np.pi
J3=0.2*np.pi

J2Coef=0.5
J2=J2Coef*np.pi


M = 1
sigma0=np.array([[1,0],[0,1]],dtype=complex)
sigma3 = np.array([[1, 0], [0, -1]],dtype=complex)
sigma2 = np.array([[0, -1j], [1j, 0]],dtype=complex)
sigma1 = np.array([[0, 1], [1, 0]],dtype=complex)

N1 = 70
N2 = 70
slopesAll=np.arange(0,1,0.5)

#spatial part of H10
S1=np.zeros((N1*N2,N1*N2),dtype=complex)
for ny in range(0,N2):
    for nx in range(0,N1-1):
        S1[nx*N2+ny,(nx+1)*N2+ny]=1
        S1[(nx+1)*N2+ny,nx*N2+ny]=-1

H10Sharp=np.kron(3*J1/(2*1j)*S1,sigma1)
H10SharpSparse=csc_matrix(H10Sharp)


#spatial part of H20
S2=np.zeros((N1*N2,N1*N2),dtype=complex)
for nx in range(0,N1):
    for ny in range(0,N2-1):
        S2[nx*N2+ny,nx*N2+ny+1]=1
        S2[nx*N2+ny+1,nx*N2+ny]=-1

H20Sharp=np.kron(3*J2/(2*1j)*S2,sigma2)
H20SharpSparse=csc_matrix(H20Sharp)

#spatial part of H30
S3=np.zeros((N1*N2,N1*N2),dtype=complex)
#part 1
for nx in range(0,N1):
    for ny in range(0,N2):
        S3[nx*N2+ny,nx*N2+ny]=2*M

#part 2
for ny in range(0,N2):
    for nx in range(0,N1-1):
        S3[nx*N2+ny,(nx+1)*N2+ny]=1
        S3[(nx+1)*N2+ny,nx*N2+ny]=1
#part 3
for nx in range(0,N1):
    for ny in range(0,N2-1):
        S3[nx*N2+ny,nx*N2+ny+1]=1
        S3[nx*N2+ny+1,nx*N2+ny]=1

H30Sharp=np.kron(3*J3/2*S3,sigma3)
H30SharpSparse=csc_matrix(H30Sharp)
leftStartingPoint=2
rightStartingPoint=N1-3


def U(j):
    """

    :param j: index of slopes
    :return: [j,U]
    """
    slope=slopesAll[j]
    leftSlope = slope
    rightSlope = slope

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


    S4=np.zeros((N1*N2,N1*N2),dtype=complex)
    for n1 in range(0,N1):
        for n2 in range(0,N2):
            S4[n1*N2+n2,n1*N2+n2]=diagV[n1]
    H40Soft=np.kron(S4,sigma0)
    H40SoftSparse=csc_matrix(H40Soft)

    H10TotSparse=H10SharpSparse+H40SoftSparse
    H20TotSparse=H20SharpSparse+H40SoftSparse
    H30TotSparse=H30SharpSparse+H40SoftSparse

    U1Sparse=expm(-1j*1/3*H10TotSparse)
    U2Sparse=expm(-1j*1/3*H20TotSparse)
    U3Sparse=expm(-1j*1/3*H30TotSparse)

    UMat=(U3Sparse@U2Sparse@U1Sparse).toarray()

    return [j,UMat]


EValsAll=[0.065*np.pi,0.935*np.pi]
pSp=np.zeros((2*N2,N1*N2),dtype=complex)
for n in range(0,N2):
    pSp[n,n]=1
for n in range(1,N2+1):
    pSp[-n,-n]=1

P=np.kron(pSp,sigma0)
PT=P.T
PTP=PT@P

PTPtorch=torch.from_numpy(PTP)
Ptorch=torch.from_numpy(P)
PTtorch=torch.from_numpy(PT)


tUStart=datetime.now()
procNum=48
pool0=Pool(procNum)
jUAll=pool0.map(U,range(0,len(slopesAll)))
tUEnd=datetime.now()
print("U time: ",tUEnd-tUStart)

EIndjUAll=[]
for EInd in range(0,len(EValsAll)):
    for itemTmp in jUAll:
        newItem=[EInd]+itemTmp
        EIndjUAll.append(newItem)

def toBeInversed(EIndjU):
    """

    :param EIndjU: [EInd, j, U]
    :return: 1-e^{iE}U(1-PTP)
    """

    EInd, j, UMat=EIndjU

    Utensor=torch.from_numpy(UMat)

    identityMat=torch.eye(2*N1*N2,dtype=torch.cfloat)
    ETmp=EValsAll[EInd]

    rst=identityMat-np.exp(1j*ETmp)*Utensor@(identityMat-PTPtorch)

    return [EInd, j, rst]

tInvStart=datetime.now()
procNum=48

pool1=Pool(procNum)

outDataAll=pool1.map(toBeInversed,EIndjUAll)

tensorToBeInvsered=torch.zeros((len(EValsAll),len(slopesAll),2*N1*N2,2*N1*N2),dtype=torch.cfloat)

for itemTmp in outDataAll:
    EInd,j,matTmp=itemTmp
    tensorToBeInvsered[EInd,j,:,:]=matTmp

inversed=np.linalg.inv(tensorToBeInvsered)

tInvEnd=datetime.now()

print("inv time: ",tInvEnd-tInvStart)

tProdStart=datetime.now()
pRow,pCol=P.shape
leftP=torch.zeros((len(EValsAll),len(slopesAll),pRow,pCol),dtype=torch.cfloat)

pTRow, pTCol=PT.shape
rightEUPT=torch.zeros((len(EValsAll),len(slopesAll),2*N1*N2,pTCol),dtype=torch.cfloat)

for EInd in range(0,len(EValsAll)):
    for j in range(0,len(slopesAll)):
        leftP[EInd,j,:,:]=Ptorch

for itemTmp in jUAll:
    j,UMat=itemTmp
    UTensor=torch.from_numpy(UMat)
    for EInd in range(0,len(EValsAll)):
        ETmp=EValsAll[EInd]
        rightEUPT[EInd,j,:,:]=np.exp(1j*ETmp)*UTensor@PTtorch

STensor=leftP@inversed@rightEUPT





tTensor=torch.zeros((len(EValsAll),len(slopesAll),2*N2,2*N2),dtype=torch.cfloat)

tDaggerTensor=torch.zeros((len(EValsAll),len(slopesAll),2*N2,2*N2),dtype=torch.cfloat)

for EInd in range(0,len(EValsAll)):
    for j in range(0,len(slopesAll)):
        tTmp=STensor[EInd,j,0:(2*N2),(-2*N2):]
        tTensor[EInd,j,:,:]=tTmp
        tDaggerTensor[EInd,j,:,:]=torch.transpose(torch.conj(tTmp),0,1)

tProd=tDaggerTensor@tTensor

tProdEnd=datetime.now()

print("prod time: ",tProdEnd-tProdStart)

G=np.zeros((len(EValsAll),len(slopesAll)))

for EInd in range(0,len(EValsAll)):
    for j in range(0,len(slopesAll)):
        G[EInd,j]=torch.trace(tProd[EInd,j,:,:])

print(G)