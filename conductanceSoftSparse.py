import numpy as np
from datetime import datetime

import pandas as pd
# from multiprocessing import Pool
import torch
from scipy.sparse import lil_matrix
#this script calculates conductance with soft edge
from scipy.sparse import kron
from scipy.sparse.linalg import expm


J1=0.5*np.pi
J3=0.2*np.pi

J2Coef=0.5
J2=J2Coef*np.pi


M = 1
sigma0=np.array([[1,0],[0,1]],dtype=complex)
sigma0Lil=lil_matrix(sigma0)
sigma3 = np.array([[1, 0], [0, -1]],dtype=complex)
sigma2 = np.array([[0, -1j], [1j, 0]],dtype=complex)
sigma1 = np.array([[0, 1], [1, 0]],dtype=complex)


N1 = 70
N2 = 70
slopeStart=0
slopeEndPast=20
slopesAll=np.arange(slopeStart,slopeEndPast,0.1)


#spatial part of H10
S1=np.zeros((N1*N2,N1*N2),dtype=complex)
for ny in range(0,N2):
    for nx in range(0,N1-1):
        S1[nx*N2+ny,(nx+1)*N2+ny]=1
        S1[(nx+1)*N2+ny,nx*N2+ny]=-1

H10Sharp=np.kron(3*J1/(2*1j)*S1,sigma1)
H10SharpLil=lil_matrix(H10Sharp)

#spatial part of H20
S2=np.zeros((N1*N2,N1*N2),dtype=complex)
for nx in range(0,N1):
    for ny in range(0,N2-1):
        S2[nx*N2+ny,nx*N2+ny+1]=1
        S2[nx*N2+ny+1,nx*N2+ny]=-1


H20Sharp=np.kron(3*J2/(2*1j)*S2,sigma2)
H20SharpLil=lil_matrix(H20Sharp)

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
H30SharpLil=lil_matrix(H30Sharp)
leftStartingPoint=2
rightStartingPoint=N1-3

UTensor=torch.zeros((len(slopesAll),2*N1*N2,2*N1*N2),dtype=torch.cfloat)
# UTensor=UTensor.cuda()

def U1U2U3U(j):
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

    # S4 = np.zeros((N1 * N2, N1 * N2), dtype=complex)
    S4Lil=lil_matrix((N1*N2,N1*N2),dtype=complex)
    for n1 in range(0, N1):
        for n2 in range(0, N2):
            S4Lil[n1 * N2 + n2, n1 * N2 + n2] = diagV[n1]
    H40SoftLil = kron(S4Lil, sigma0Lil)

    # H10Tot=torch.from_numpy(((H10SharpLil+H40SoftLil)*(-1/3*1j)).toarray())#.cuda()
    # H20Tot=torch.from_numpy(((H20SharpLil+H40SoftLil)*(-1/3*1j)).toarray())#.cuda()
    # H30Tot=torch.from_numpy(((H30SharpLil+H40SoftLil)*(-1/3*1j)).toarray())#.cuda()
    # tmp=H30Tot.matrix_exp()
    # UTensor[j,:,:]=tmp@H20Tot.matrix_exp()@H10Tot.matrix_exp()
    H10Tot=-1/3*1j*(H10SharpLil+H40SoftLil).tocsc()
    H20Tot=-1/3*1j*(H20SharpLil+H40SoftLil).tocsc()
    H30Tot=-1/3*1j*(H30SharpLil+H40SoftLil).tocsc()
    tmp=(expm(H30Tot)@expm(H20Tot)@expm(H10Tot)).toarray()
    UTensor[j,:,:]=torch.from_numpy(tmp)
    # return tmp


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



tUStart=datetime.now()
#10min each
for j in range(0,len(slopesAll)):
    U1U2U3U(j)
tUEnd=datetime.now()

print("U time: ",tUEnd-tUStart)
#each matrix in UTensor is dense
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

identityMat = torch.eye(2 * N1 * N2, dtype=torch.cfloat)
x=identityMat-PTPtorch

tensorToBeInversed=torch.zeros((len(EValsAll),len(slopesAll),2*N1*N2,2*N1*N2),dtype=torch.cfloat)


def oneToBeInversed(EIndj):
    """

    :param EIndj: [EInd, j]
    :return: insert 1-e^{iE}U(1-PTP)
    """
    EInd, j=EIndj
    ETmp=EValsAll[EInd]

    UTmp=UTensor[j,:,:]



    rst=identityMat-np.exp(1j*ETmp)*UTmp@x

    tensorToBeInversed[EInd,j,:,:]=rst



EIndjAll=[[EInd,j] for EInd in range(0,2) for j in range(0,len(slopesAll))]
tInitStart=datetime.now()
for pair in EIndjAll:
    oneToBeInversed(pair)

tInitEnd=datetime.now()

print("init time: ",tInitEnd-tInitStart)#32min

# tFillStart=datetime.now()
# for EInd in range(0,2):
#     for j in range(0,len(slopesAll)):
#         tensorToBeInversed[EInd,j,:,:]=torch.from_numpy(np.random.randn(2*N1*N2,2*N1*N2))
# tFillEnd=datetime.now()
# print("filling time: ",tFillEnd-tFillStart)#30min

tInvStart=datetime.now()
# tensorToBeInversed=tensorToBeInversed.cuda()

inversedTensor=torch.linalg.inv(tensorToBeInversed)

tInvEnd=datetime.now()
print("inv time: ",tInvEnd-tInvStart)#51min

tGStart=datetime.now()
pRow, pCol=P.shape

leftP=torch.zeros((len(EValsAll),len(slopesAll),pRow,pCol),dtype=torch.cfloat)

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

#to csv
#data serialization
EIndCsv=[]
slopeCsv=[]
GCsv=[]
for EInd in range(0,2):
    for j in range(0,len(slopesAll)):
        EIndCsv.append(EInd)
        slopeCsv.append(j)
        GCsv.append(G[EInd,j])


dataOut=np.array([EIndCsv,slopeCsv,GCsv]).T
dtFrm=pd.DataFrame(data=dataOut,columns=["EInd","slope","G"])

dtFrm.to_csv("J2"+str(J2Coef)+f"slopes{slopeStart}To{slopeEndPast}.csv")