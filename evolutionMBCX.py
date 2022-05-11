import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.linalg import  expm
from pathlib import Path
from scipy.sparse import csc_matrix
import mpmath
import pandas as pd
import shutil
import os
###this script computes evolution for MBCX
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
Q=30
dt=1/(3*Q)
g=-1

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
###############################################
###############read from linear state csv
# inLeftFile="./leftLinearJ2"+str(J2Coef)+"N2"+str(N2)+"/leftVecsJ2"+str(J2Coef)+".csv"
# inLeftDtFrm=pd.read_csv(inLeftFile,header=None)
#
# b=14
# selectedRow=inLeftDtFrm.loc[inLeftDtFrm.iloc[:,1]==b]
# k2=2*np.pi*b/N2
#
# vecLeftStr=np.array(selectedRow.iloc[0,3:])
#############################################
############################################
##read from nonlinear state csv
inLeftDir="./leftNonlinJ20.5N230/g-1nonLinEigPlots/b13/"
inLeftFile=inLeftDir+"eigwvfunction.csv"
inLeftDtFrm=pd.read_csv(inLeftFile,header=None)
vecLeftStr=np.array(inLeftDtFrm.iloc[0,:])
vecLeft=np.array([complex(elem) for elem in vecLeftStr])

psiInit=np.zeros((N1*N2*2,) ,dtype=complex)

mu=5

for n1 in range(0,N1):
    for n2 in range(0,N2):
        latticeNum=n1*N2+n2
        factor=1/np.cosh(mu*(n2-N2/2))
        psiInit[latticeNum*2]=vecLeft[latticeNum*2]*factor
        psiInit[latticeNum*2+1]=vecLeft[latticeNum*2+1]*factor

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
psiInit/=np.linalg.norm(psiInit,2)
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
periodNum=10
tEvolveStart=datetime.now()
dataAll=[psiInit]
for prd in range(0,periodNum):
    psiInit=evolutionOnePeriod(psiInit)
    dataAll.append(psiInit)

tEvolveEnd=datetime.now()
print(f"evolution {periodNum} periods time: {tEvolveEnd-tEvolveStart}")
outLeftDir=inLeftDir+"evo/"
if os.path.exists(outLeftDir) and os.path.isdir(outLeftDir):
    shutil.rmtree(outLeftDir)
Path(outLeftDir).mkdir(parents=True,exist_ok=True)
for j in range(0,len(dataAll)):
    psiTmp=dataAll[j]
    matTmp=vec2Mat(psiTmp)
    plt.figure()
    imTmp=plt.imshow(matTmp,cmap=plt.cm.RdBu,interpolation="bilinear")
    plt.colorbar(imTmp)
    plt.xlabel("$n_{2}$")
    plt.ylabel("$n_{1}$")
    plt.title("g="+str(g)+", $\mu=$"+str(mu)+", j="+str(j))
    plt.savefig(outLeftDir+str(j)+".png")