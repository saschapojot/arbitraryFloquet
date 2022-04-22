
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
# this script computes the nonlinear eigenvalue problem
############consts
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
Q=30
dt=1/(3*Q)
g=0.1
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

# U1Sparse=csc_matrix(U1)
# U2Sparse=csc_matrix(U2)
# U3Sparse=csc_matrix(U3)
print("init time: ",tInitEnd-tInitStart)

# inLeftFile="./leftLinearJ2"+str(J2Coef)+"N2"+str(N2)+"/leftVecsJ2"+str(J2Coef)+".csv"
# inLeftDtFrm=pd.read_csv(inLeftFile,header=None)
#
b=17
# selectedRow=inLeftDtFrm.loc[inLeftDtFrm.iloc[:,1]==b]
k2=2*np.pi*b/N2
#
# vecLeftStr=np.array(selectedRow.iloc[0,3:])
# vecLeft=np.array([complex(elem) for elem in vecLeftStr])

psiInit=np.zeros((N1*N2*2,) ,dtype=complex)

mu=0.5

for n1 in range(0,N1):
    for n2 in range(0,N2):
        vecTmp=np.zeros((N1*N2,),dtype=complex)
        vecTmp[n1*N2+n2]=1
        vecRightTmp=[1,1]
        psiInit+=1/np.sqrt(N2)*np.exp(1j*k2*n2)*np.kron(vecTmp,vecRightTmp)

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

# def generateUn(vecsAll):
#     lengthTmp=len(vecsAll[0])
#     retUn=identity(lengthTmp,dtype=complex)
#     for q in range(0,Q):
#         vecTmp=vecsAll[q]
#         U1q=expm(-1j*1/3*dt*(H10Sparse+g*diags(np.abs(vecTmp)**2)))
#         retUn=U1q@retUn
#
#     for q in range(Q,2*Q):
#         vecTmp=vecsAll[q]
#         U2q=expm(-1j*1/3*dt*(H20Sparse+g*diags(np.abs(vecTmp)**2)))
#         retUn=U2q@retUn
#     for q in range(2*Q,3*Q):
#         vecTmp=vecsAll[q]
#         U3q=expm(-1j*1/3*dt*(H30Sparse+g*diags(np.abs(vecTmp)**2)))
#         retUn=U3q@retUn
#     return retUn
psiAll=generateWavefunctions(psiInit)
lTmp=len(psiAll[0])
def oneExpToGetUq(q):
    if q>=0 and q<Q:
        vec1Tmp=psiAll[q]
        U1q=expm(-1j*1/3*dt*(H10Sparse+g*diags(np.abs(vec1Tmp)**2)))
        return [q,U1q]
    elif q>=Q and q<2*Q:
        vec2Tmp=psiAll[q]
        U2q=expm(-1j*1/3*dt*(H20Sparse+g*diags(np.abs(vec2Tmp)**2)))
        return [q,U2q]
    elif q>=2*Q and q<3*Q:
        vec3Tmp=psiAll[q]
        U3q=expm(-1j*1/3*dt*(H30Sparse+g*diags(np.abs(vec3Tmp)**2)))
        return [q,U3q]

tOneRoundStart=datetime.now()
###########################calculate expm using scipy
threadNum=48
tExpStart=datetime.now()
pool0=Pool(threadNum)
qList=range(0,3*Q)
ret0=pool0.map(oneExpToGetUq,qList)
tExpEnd=datetime.now()
print(f'expm time: {tExpEnd-tExpStart}')
######################################
#data serialization
# sortedRet0=sorted(ret0,key=lambda elem:elem[0])
####################################### device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

####################################### calculating H and U using pytorch
# def retHToBeExponentiated(q):
#     if q>=0 and q<Q:
#         vec1Tmp=psiAll[q]
#         retH1=-1j*1/3*dt*(H10Sparse+g*diags(np.abs(vec1Tmp)**2))
#         return [q,retH1]
#     elif q>=Q and q<2*Q:
#         vec2Tmp = psiAll[q]
#         retH2=-1j*1/3*dt*(H20Sparse+g*diags(np.abs(vec2Tmp)**2))
#         return [q,retH2]
#     elif q>=2*Q and q<3*Q:
#         vec3Tmp=psiAll[q]
#         retH3=-1j*1/3*dt*(H30Sparse+g*diags(np.abs(vec3Tmp)**2))
#         return numpy[q,retH3]
#

# threadNum=24
# pool1=Pool(threadNum)
# qList=range(0,3*Q)
# ret1=pool1.map(retHToBeExponentiated,qList)
#
# HTensor=torch.zeros((3*Q,lTmp,lTmp),dtype=torch.cfloat)
# for elem in ret1:
#     q=elem[0]
#     HTmp=elem[1]
#     HTensor[q,:,:]=torch.from_numpy(HTmp.toarray())

# HTensor.cuda()


# tInitHEnd=datetime.now()
# print(f"init H time: {tInitHEnd-tInitStart}")


######################################
############################matrix multiplication procedure that is too slow
#
# tProdStart=datetime.now()
# lTmp=len(psiAll[0])
# retUn=identity(lTmp,dtype=complex)
# for elem in sortedRet0:
#     UTmp=elem[1]
#     retUn=UTmp@retUn
#
# tProdEnd=datetime.now()
# print(f"prod time: {tProdEnd-tProdStart}")
############################################\

tInitTensorStart=datetime.now()


UqTensor=torch.zeros((3*Q,lTmp,lTmp),dtype=torch.cfloat)


for elem in  ret0:
    q=elem[0]
    U=elem[1].toarray()
    UqTensor[q,:,:]=torch.from_numpy(U)

tInitTensorEnd=datetime.now()

print(f"init U tensor time: {tInitTensorEnd-tInitTensorStart}")

retUn=torch.eye(lTmp,dtype=torch.cfloat).cuda()
tToCudaStart=datetime.now()
UqTensor=UqTensor.cuda()
tToCudaEnd=datetime.now()
print(f"to cuda time:  {tToCudaEnd-tToCudaStart}")
tProdStart=datetime.now()
for q in range(0,3*Q):
    retUn=UqTensor[q,:,:]@retUn
tProdEnd=datetime.now()
print(f"prod time: {tProdEnd-tProdStart}")

tEigStart=datetime.now()
UNP=retUn.cpu().detach().numpy()
eigVals,vecs=np.linalg.eig(UNP)
prodsAll=[np.abs(np.vdot(vec,psiInit)) for vec in vecs]
inds=np.argsort(prodsAll)
print(prodsAll[inds[0]])
tEigEnd=datetime.now()
print(f"eig time: {tEigEnd-tEigStart}")

tOneRoundEnd=datetime.now()
print(f"one round time: {tOneRoundEnd-tOneRoundStart}")