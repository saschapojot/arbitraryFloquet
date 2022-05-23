import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from multiprocessing import Pool
import torch

#this script calculates conductance with soft edge

J1=0.5*np.pi
J3=0.2*np.pi

J2Coef=0.5
J2=J2Coef*np.pi


M = 1
sigma0=np.array([[1,0],[0,1]],dtype=complex)
sigma3 = np.array([[1, 0], [0, -1]],dtype=complex)
sigma2 = np.array([[0, -1j], [1j, 0]],dtype=complex)
sigma1 = np.array([[0, 1], [1, 0]],dtype=complex)


N1 = 7
N2 = 7
slopesAll=np.arange(0,1,0.5)


#spatial part of H10
S1=np.zeros((N1*N2,N1*N2),dtype=complex)
for ny in range(0,N2):
    for nx in range(0,N1-1):
        S1[nx*N2+ny,(nx+1)*N2+ny]=1
        S1[(nx+1)*N2+ny,nx*N2+ny]=-1

H10Sharp=np.kron(3*J1/(2*1j)*S1,sigma1)


#spatial part of H20
S2=np.zeros((N1*N2,N1*N2),dtype=complex)
for nx in range(0,N1):
    for ny in range(0,N2-1):
        S2[nx*N2+ny,nx*N2+ny+1]=1
        S2[nx*N2+ny+1,nx*N2+ny]=-1


H20Sharp=np.kron(3*J2/(2*1j)*S2,sigma2)


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

leftStartingPoint=2
rightStartingPoint=N1-3
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device is "+str(device))
H1Tensor=torch.zeros((len(slopesAll),2*N1*N2,2*N1*N2),dtype=torch.cfloat)
H2Tensor=torch.zeros((len(slopesAll),2*N1*N2,2*N1*N2),dtype=torch.cfloat)
H3Tensor=torch.zeros((len(slopesAll),2*N1*N2,2*N1*N2),dtype=torch.cfloat)



def H1H2H3(j):
    """

    :param j: index of slopes
    :return: [j, -1/3i*H1, -1/3i*H2, -1/3i*H3]
    """

    slope = slopesAll[j]
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

    S4 = np.zeros((N1 * N2, N1 * N2), dtype=complex)
    for n1 in range(0, N1):
        for n2 in range(0, N2):
            S4[n1 * N2 + n2, n1 * N2 + n2] = diagV[n1]
    H40Soft = np.kron(S4, sigma0)

    H10Tot=H10Sharp+H40Soft
    H20Tot=H20Sharp+H40Soft
    H30Tot=H30Sharp+H40Soft

    return [j,-1/3*1j*H10Tot, -1/3*1j*H20Tot,-1/3*1j*H30Tot]

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

#cast to cfloat
PTPtorch=PTPtorch.type(torch.cfloat)
Ptorch=Ptorch.type(torch.cfloat)
PTtorch=PTtorch.type(torch.cfloat)

t3HStart=datetime.now()
procNum=48
pool0=Pool(procNum)
jHHH=pool0.map(H1H2H3,range(0,len(slopesAll)))

t3HEnd=datetime.now()

print("3H time: ",t3HEnd-t3HStart)


tExpStart=datetime.now()
for itemTmp in jHHH:
    j,mat1,mat2,mat3=itemTmp
    H1Tensor[j,:,:]=torch.from_numpy(mat1)
    H2Tensor[j,:,:]=torch.from_numpy(mat2)
    H3Tensor[j,:,:]=torch.from_numpy(mat3)


U1Tensor=H1Tensor.matrix_exp()
U2Tensor=H2Tensor.matrix_exp()
U3Tensor=H3Tensor.matrix_exp()

tExpEnd=datetime.now()
print("exp time: ",tExpEnd-tExpStart)

tProdStart=datetime.now()
UTensor=U3Tensor@U2Tensor@U1Tensor
tProdEnd=datetime.now()

print("prod time: ",tProdEnd-tProdStart)
identityMat = torch.eye(2 * N1 * N2, dtype=torch.cfloat)
x=identityMat-PTPtorch
def oneToBeInversed(EIndj):
    """

    :param EIndj: [EInd, j]
    :return: 1-e^{iE}U(1-PTP)
    """
    EInd, j=EIndj
    ETmp=EValsAll[EInd]

    UTmp=UTensor[j,:,:]



    rst=identityMat-np.exp(1j*ETmp)*UTmp@x

    return [EInd, j, rst]


tInitStart=datetime.now()
procNum=48
pool1=Pool(procNum)
EIndjAll=[[EInd,j] for EInd in range(0,2) for j in range(0,len(slopesAll))]
toBeInversedAll=pool1.map(oneToBeInversed,EIndjAll)
tensorToBeInversed=torch.zeros((len(EValsAll),len(slopesAll),2*N1*N2,2*N1*N2),dtype=torch.cfloat)

for itemTmp in toBeInversedAll:
    EInd, j,matTmp=itemTmp
    tensorToBeInversed[EInd,j,:,:]=matTmp

tInitEnd=datetime.now()
print("init time: ",tInitEnd-tInitStart)

tInvStart=datetime.now()
# tensorToBeInversed=tensorToBeInversed.cuda()

inversedTensor=torch.linalg.inv(tensorToBeInversed)

tInvEnd=datetime.now()

print("inv time: ",tInvEnd-tInvStart)


tGStart=datetime.now()
pRow, pCol=P.shape

leftP=torch.zeros((len(EValsAll),len(slopesAll),pRow,pCol),dtype=torch.cfloat)

pTRow, pTCol=PT.shape
rightEUPT=torch.zeros((len(EValsAll),len(slopesAll),2*N1*N2,pTCol),dtype=torch.cfloat)

for EInd in range(0,len(EValsAll)):
    for j in range(0,len(slopesAll)):
        leftP[EInd,j,:,:]=Ptorch

pTRow, pTCol=PT.shape
rightEUPT=torch.zeros((len(EValsAll),len(slopesAll),2*N1*N2,pTCol),dtype=torch.cfloat)

for j in range(0,len(slopesAll)):
    for EInd in range(0,len(EValsAll)):
        ETmp=EValsAll[EInd]
        rightEUPT[EInd,j,:,:]=np.exp(1j*ETmp)*UTensor[j,:,:]@PTtorch

STensor=leftP@inversedTensor@rightEUPT

tTensor=torch.zeros((len(EValsAll),len(slopesAll),2*N2,2*N2),dtype=torch.cfloat)

tDaggerTensor=torch.zeros((len(EValsAll),len(slopesAll),2*N2,2*N2),dtype=torch.cfloat)

for EInd in range(0,len(EValsAll)):
    for j in range(0,len(slopesAll)):
        tTmp=STensor[EInd,j,0:(2*N2),(-2*N2):]
        tTensor[EInd,j,:,:]=tTmp
        tDaggerTensor[EInd,j,:,:]=torch.transpose(torch.conj(tTmp),0,1)

tProd=tDaggerTensor@tTensor



G=np.zeros((len(EValsAll),len(slopesAll)))

for EInd in range(0,len(EValsAll)):
    for j in range(0,len(slopesAll)):
        G[EInd,j]=torch.trace(tProd[EInd,j,:,:])

tGEnd=datetime.now()

print("G time: ",tGEnd-tGStart)

print(G)

