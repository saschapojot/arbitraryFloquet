import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.linalg import  expm
from pathlib import Path
from scipy.sparse import csc_matrix

import pandas as pd
############consts
J1=0.5*np.pi

J3=0.2*np.pi
J2Coef=0.5
J2=J2Coef*np.pi

M=1

sigma3 = np.array([[1, 0], [0, -1]],dtype=complex)
sigma2 = np.array([[0, -1j], [1j, 0]],dtype=complex)
sigma1 = np.array([[0, 1], [1, 0]],dtype=complex)

N1 = 30
N2 =  30
Q=60
dt=1/(3*Q)
g=1

tInitStart=datetime.now()
#spatial part of H10
S10=np.zeros((N1*N2,N1*N2),dtype=complex)
for nx in range(0,N1-1):
    for ny in range(0,N2):
        S10[nx*N2+ny,nx*N2+N2+ny]=1
        S10[nx*N2+N2+ny,nx*N2+ny]=-1

H10=3*J1/(2*1j)*np.kron(S10,sigma1)
U1=expm(-1j*dt*H10)

def H20Mat(J2):
    """

    :param J2:
    :return:
    """
    #spatial part of H20
    S20=np.zeros((N1*N2,N1*N2),dtype=complex)
    for nx in range(0,N1):
        for ny in range(0,N2):
            S20[nx*N2+ny,nx*N2+(ny+1)%N2]=1
            S20[nx*N2+(ny+1)%N2,nx*N2+ny]=-1
    return 3*J2/(2*1j)*np.kron(S20,sigma2)

def U2Mat(J2):
    return expm(-1j*dt*H20Mat(J2))



U2=U2Mat(J2)


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
U3=expm(-1j*dt*H30)
rowN,colN=U3.shape

U1Sparse=csc_matrix(U1)
U2Sparse=csc_matrix(U2)
U3Sparse=csc_matrix(U3)

tInitEnd=datetime.now()
print("init time: ",tInitEnd-tInitStart)

inFile="./nonlinEigPlots/center/eigwvfunctionCenter.csv"
inDtFrm=pd.read_csv(inFile,header=None)
psiInitStr=np.array(inDtFrm.iloc[0,:])
def str2complexVec(vecStr):
    ret=[]
    for elem in vecStr:
        ret.append(complex(elem))
    return ret
psiInit=str2complexVec(psiInitStr)

# mat=np.zeros((N1,N2))
# for j in range(0,int(len(psiInit)/2)):
#     n2=j%N2
#     n1=int((j-n2)/N2)
#     mat[n1,n2]=np.abs(psiInit[2*j])**2+np.abs(psiInit[2*j+1])**2

# plt.figure()
# im=plt.imshow(mat,cmap=plt.cm.RdBu,interpolation="bilinear")
# plt.colorbar(im)
# plt.savefig("tmp.png")
# plt.close()

def vec2Mat(vec):
    """

    :param vec: length=2*N1*N2
    :return: matrix representation of wavefunction
    """
    mat=np.zeros((N1,N2),dtype=float)
    for j in range(0,int(len(vec)/2)):
        n2=j%N2
        n1=int((j-n2)/N2)
        mat[n1,n2]=np.abs(vec[2*j])**2+np.abs(vec[2*j+1])**2
    return mat




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


def evolutionOnePeriod(vec):
    for q in range(0,Q):
        vec=oneStep1(vec)
    for q in range(0,Q):
        vec=oneStep2(vec)
    for q in range(0,Q):
        vec=oneStep3(vec)
    return vec
periodNum=100
tEvolveStart=datetime.now()
dataAll=[psiInit]


for prd in range(0,periodNum):
    psiInit=evolutionOnePeriod(psiInit)
    dataAll.append(psiInit)
tEvolveEnd=datetime.now()

print(str(periodNum)+" period evolution time: ",tEvolveEnd-tEvolveStart)

outFigsDir="./nonlinEigPlots/center/evoFigs/"
Path(outFigsDir).mkdir(parents=True,exist_ok=True)
tPltStart=datetime.now()
for j in range(0,len(dataAll)):
    vecTmp=dataAll[j]
    outMat=vec2Mat(vecTmp)
    plt.figure()
    im = plt.imshow(outMat, cmap=plt.cm.RdBu, interpolation="bilinear")
    plt.colorbar(im)
    plt.xlabel("$n_{2}$")
    plt.ylabel("$n_{1}$")
    plt.title(str(j)+"$T$")
    plt.savefig(outFigsDir + "j" + str(j) + ".png")
    plt.close()

tPltEnd=datetime.now()
print(f"plt time: {tPltEnd-tPltStart}")