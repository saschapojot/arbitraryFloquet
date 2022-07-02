import numpy as np
from datetime import datetime
import tensorflow as tf
import pandas as pd
from scipy.sparse.linalg import expm
from scipy import sparse
# from multiprocessing import Pool
# import torch

#this script calculates conductance with soft edge in series

#this script computes conductance for soft potential in H3 with GPU

tTotStart=datetime.now()
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
slopeStart=9.4
slopeEndPast=9.5
slopesAll=np.arange(slopeStart,slopeEndPast,0.1)

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

leftStartingPoint=10
rightStartingPoint=N1-11


def U1U2U3(j):
    """

        :param j: index of slopes
        :return: U
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
    S4=np.zeros((N1*N2,N1*N2),dtype=complex)
    for n1 in range(0,N1):
        for n2 in range(0,N2):
            S4[n1 * N2 + n2, n1 * N2 + n2] = diagV[n1]
    H40Soft=np.kron(S4,sigma0)
    tExpStart=datetime.now()
    H10SharpCsc=sparse.csc_matrix(-1/3*1j*H10Sharp)
    H20SharpCsc=sparse.csc_matrix(-1/3*1j*H20Sharp)
    H30Csc=sparse.csc_matrix(-1/3*1j*(H30Sharp+H40Soft))

    U1=expm(H10SharpCsc).toarray()
    U2=expm(H20SharpCsc).toarray()
    U3=expm(H30Csc).toarray()

    tExpEnd=datetime.now()

    print("exp time: ",tExpEnd-tExpStart)
    return [U1,U2,U3]


EValsAll=[0.065*np.pi]

pSp=np.zeros((2*N2,N1*N2),dtype=complex)
for n in range(0,N2):
    pSp[n,n]=1
for n in range(1,N2+1):
    pSp[-n,-n]=1

P=np.kron(pSp,sigma0)
PT=P.T
PTP=PT@P

identityMat=np.eye(2*N1*N2,dtype=complex)
x=identityMat-PTP
with tf.device("/device:GPU:0"):
    Ptf=tf.convert_to_tensor(P)
    PTtf=tf.convert_to_tensor(PT)
    xtf=tf.convert_to_tensor(x)
    identityMattf=tf.convert_to_tensor(identityMat)


G=np.zeros((len(EValsAll),len(slopesAll)))
tGStart=datetime.now()
for EInd in range(0,len(EValsAll)):
    for j in range(0,len(slopesAll)):
        ETmp=EValsAll[EInd]
        U1,U2,U3=U1U2U3(j)
        with tf.device("/device:GPU:0"):
            U1tf=tf.convert_to_tensor(U1)
            U2tf=tf.convert_to_tensor(U2)
            U3tf=tf.convert_to_tensor(U3)
            tProdAndInvStart=datetime.now()
            Utf=U3tf@U2tf@U1tf
            toBeInvsersedTf=identityMattf-tf.math.exp(1j*ETmp)*Utf@xtf
            Stf=Ptf@tf.linalg.inv(toBeInvsersedTf)*tf.math.exp(1j*ETmp)@Utf@PTtf
            tProdAndInvEnd=datetime.now()
            print("prod and inv time: ",tProdAndInvEnd-tProdAndInvStart)
            ttf=Stf[0:(2*N2),(-2*N2):]
            tDaggertf=tf.linalg.adjoint(ttf)
            tProdtf=tDaggertf@ttf
            G[EInd,j]=np.float64(tf.linalg.trace(tProdtf))

tGEnd=datetime.now()
print("G time: ",tGEnd-tGStart)
#to csv
#data serialization
EIndCsv=[]
slopeCsv=[]
GCsv=[]
for EInd in range(0,len(EValsAll)):
    for j in range(0,len(slopesAll)):
        EIndCsv.append(EInd)
        slopeCsv.append(slopesAll[j])
        GCsv.append(G[EInd,j])

dataOut=np.array([EIndCsv,slopeCsv,GCsv]).T
dtFrm=pd.DataFrame(data=dataOut,columns=["EInd","slope","G"])

dtFrm.to_csv("J2"+str(J2Coef)+"slopeStart"+str(slopesAll[0])+"slopeEnd"+str(slopesAll[-1])+".csv")

tTotEnd=datetime.now()

print("total time: ",tTotEnd-tTotStart)