from datetime import datetime
import numpy as np
# from scipy.linalg import  expm
from scipy.sparse import csc_matrix
# import pandas as pd
# from scipy.sparse import identity
from scipy.sparse.linalg import expm
from scipy.sparse import diags
from multiprocessing import Pool
import torch
import sys
# this script calculates the nonlinear eigenvalue problem for part of the table

if len(sys.argv)!=3:
    print("pls enter the correct parameters")
    sys.exit(1)

sampleNum=int(sys.argv[1])
currPart=int(sys.argv[2])


J1=0.5*np.pi

J3=0.2*np.pi
J2Coef=0.5
J2=J2Coef*np.pi

M=1
# print(torch.cuda.is_available())
sigma3 = np.array([[1, 0], [0, -1]],dtype=complex)
sigma2 = np.array([[0, -1j], [1j, 0]],dtype=complex)
sigma1 = np.array([[0, 1], [1, 0]],dtype=complex)

N1 = 30
N2 =  30
Q=60
dt=1/(3*Q)
g=1




nRowStart=currPart*sampleNum
nRowEndPastOne=(currPart+1)*sampleNum

tInitStart=datetime.now()
#spatial part of H10
S10=np.zeros((N1*N2,N1*N2),dtype=complex)

for nx in range(0,N1-1):
    for ny in range(0,N2):
        S10[nx*N2+ny,nx*N2+N2+ny]=1
        S10[nx*N2+N2+ny,nx*N2+ny]=-1

H10=3*J1/(2*1j)*np.kron(S10,sigma1)
# rowN,colN=H10.shape
# print(np.count_nonzero(H10)/(rowN*colN))
H10Sparse=csc_matrix(H10)
U1Sparse=expm(-1j*dt*H10Sparse)

#spatial part of H20
S20=np.zeros((N1*N2,N1*N2),dtype=complex)
for nx in range(0,N1):
    for ny in range(0,N2):
        S20[nx*N2+ny,nx*N2+(ny+1)%N2]=1
        S20[nx*N2+(ny+1)%N2,nx*N2+ny]=-1

H20=3*J2/(2*1j)*np.kron(S20,sigma2)
# rowN,colN=H20.shape
# print(np.count_nonzero(H20)/(rowN*colN))
H20Sparse=csc_matrix(H20)
U2Sparse=expm(-1j*dt*H20Sparse)

#spatial part of H30
#part 0
S300=np.zeros((N1*N2,N1*N2),dtype=complex)
for nx in range(0,N1):
    for ny in range(0,N2):
        S300[nx*N2+ny,nx*N2+ny]=1
S300*=3*J3*M
#part 1initialize sparse matrix from numpy
S301=np.zeros((N1*N2,N1*N2),dtype=complex)
for nx in range(0,N1-1):
    for ny in range(0,N2):
        S301[nx*N2+ny,nx*N2+N2+ny]=1
        S301[nx*N2+N2+ny,nx*N2+ny]=1
S301*=3*J3/2

#part 2
S302=np.zeros((N1*N2,N1*N2),dtype=complex)
for nx in range(0,N1):
    for ny in range(0,N2):
        S302[nx*N2+ny,nx*N2+(ny+1)%N2]=1
        S302[nx*N2+(ny+1)%N2,nx*N2+ny]=1

S302*=3*J3/2
#assemble spatial part of H30
S30=S300+S301+S302
H30=np.kron(S30,sigma3)
# rowN,colN=H30.shape
# print(np.count_nonzero(H30)/(rowN*colN))
H30Sparse=csc_matrix(H30)
U3Sparse=expm(-1j*dt*H30Sparse)
# print(U3Sparse)
tInitEnd=datetime.now()

print("init time: ",tInitEnd-tInitStart)

inLinFile=f"./linearEigs/AllN2{N2}J2{J2Coef}.csv"

tableStr=np.loadtxt(inLinFile,dtype="str",delimiter=",")[nRowStart:nRowEndPastOne]

def str2Complex(vecStr):
    ret=[]
    for elem in vecStr:
        ret.append(complex(elem))
    return ret


def str2Num(tableStr):
    nRow, nCol=tableStr.shape
    tableNum=[]
    for n in range(nRow):
        oneRow=[]
        for j in range(0,3):
            oneRow.append(float(tableStr[n][j]))
        vecStr=tableStr[n][3:]

        vecComplex=str2Complex(vecStr)

        oneRow.extend(vecComplex)
        tableNum.append(oneRow)
    return tableNum

tableNum=str2Num(tableStr)

def oneStepNonlinear(vec):
    """

    :param vec:
    :return: 1/2dt nonlinear evolution
    """
    rst=[np.exp(-1j*1/2*g*dt*np.abs(elem)**2)*elem for elem in vec]
    return rst


def oneStep1(vec):
    """

    :param vec:
    :return: one step evolution for the 1st 1/3 time
    """
    #1/2dt nonlinear evolution
    tmp1=oneStepNonlinear(vec)
    #dt linear evolution
    tmp2=U1Sparse.dot(tmp1)
    #1/2dt nonlinear evolution
    tmp3=oneStepNonlinear(tmp2)
    return tmp3

def oneStep2(vec):
    """

    :param vec:
    :return: one step evolution for the 2nd 1/3 time
    """
    #1/2 dt nonlinear evolution
    tmp1=oneStepNonlinear(vec)
    #dt linear evolution
    tmp2=U2Sparse.dot(tmp1)
    #1/2 dt nonlinear evolution
    tmp3=oneStepNonlinear(tmp2)
    return tmp3

def oneStep3(vec):
    """

    :param vec:
    :return: one step evolution for the 3nd 1/3 time
    """

    #1/2 dt nonlinear evolution
    tmp1=oneStepNonlinear(vec)
    #dt linear evolution
    tmp2=U3Sparse.dot(tmp1)
    #1/2dt nonlinear evolution
    tmp3=oneStepNonlinear(tmp2)
    return tmp3
def generateWavefunctions(initVec):
    """

    :param initVec:
    :return: wavefunctions at each time step
    """

    ret=[initVec]
    for q in range(0,Q):
        psiCurr=ret[-1]
        psiNext=oneStep1(psiCurr)
        ret.append(psiNext)
    for q in range(0,Q):
        psiCurr=ret[-1]
        psiNext=oneStep2(psiCurr)
        ret.append(psiNext)
    for q in range(0,Q):
        psiCurr=ret[-1]
        psiNext=oneStep3(psiCurr)
        ret.append(psiNext)
    return ret


lTmp=len(tableNum[0][3:])*30#length of a vector in computation
UqTensor=torch.zeros((3*Q,lTmp,lTmp),dtype=torch.cfloat)
UqTensorCuda = UqTensor.cuda()
eps=1e-5
maxit=50

def iteration(b,psiLinVec):
    """
    :param b: index of k2
    :param psiLinVec: input linear eigenvector
    :return:
    """

    #construct initial wavefunction
    # declare psiInit
    psiInit = np.zeros((N1 * N2 * 2,), dtype=complex)
    k2=2*np.pi*b/N2

    for n1 in range(0,N1):
        for n2 in range(0,N2):
            vecTmp=np.zeros((N1*N2,),dtype=complex)
            vecTmp[n1*N2+n2]=1
            vecRightTmp=[psiLinVec[2*n1],psiLinVec[2*n1+1]]
            psiInit/=np.sqrt(N2)*np.exp(1j*k2*n2)*np.kron(vecTmp,vecRightTmp)
    psiInit/=np.linalg.norm(psiInit,2)
    psiNext = psiInit[:]
    psiAll = generateWavefunctions(psiNext)







