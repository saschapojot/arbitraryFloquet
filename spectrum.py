import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from multiprocessing import Pool
from datetime import datetime

J1=0.5*np.pi
J3=0.2*np.pi
M=1
sigma3=np.array([[1,0],[0,-1]])
sigma2=np.array([[0,-1j],[1j,0]])
sigma1=np.array([[0,1],[1,0]])

N=100
dk=2*np.pi/N
#k ind from 0 to N-1



J2Min=0.1*np.pi
J2Max=9.5*np.pi
dJ2=0.1*np.pi
maxIndJ2=int((J2Max-J2Min)/dJ2)+2

J2ValsAll=[J2Min+n*dJ2 for n in range(0,maxIndJ2)]


def F3(ab):
    a,b=ab
    ka=a*dk
    kb=b*dk
    return [a,b,expm(-1j*J3*(M+np.cos(ka)+np.cos(kb))*sigma3)]

def F2(bJ2Ind):

    b,J2Ind=bJ2Ind
    J2=J2ValsAll[J2Ind]
    kb=b*dk
    return [b,J2Ind,expm(-1j*J2*np.sin(kb)*sigma2)]

def F1(a):
    ka=a*dk
    return [a,expm(-1j*J1*np.sin(ka)*sigma1)]




F3Tensor=np.zeros((N,N,2,2),dtype=complex)

F2Tensor=np.zeros((N,maxIndJ2,2,2),dtype=complex)

F1Tensor=np.zeros((N,2,2),dtype=complex)

F3InData=[[a,b] for a in range(0,N) for b in range(0,N)]
F2InData=[[b,J2Ind] for b in range(0,N) for J2Ind in range(0,maxIndJ2)]
F1InData=[a for a in range(0,N)]

threaNum=24
tFStart=datetime.now()
pool0=Pool(threaNum)

ret3=pool0.map(F3,F3InData)


pool1=Pool(threaNum)
ret2=pool1.map(F2,F2InData)

pool2=Pool(threaNum)

ret1=pool2.map(F1,F1InData)


tFEnd=datetime.now()
print("F time : ",tFEnd-tFStart)

#populate F3Tensor
tProdStart=datetime.now()
lTmp=len(psiAll[0])
retUn=identity(lTmp,dtype=complex)
for elem in sortedRet0:
    UTmp=elem[1]
    retUn=UTmp@retUn

tProdEnd=datetime.now()
print(f"prod time: {tProdEnd-tProdStart}")
for itemTmp in ret3:
    a,b,UTmp=itemTmp
    F3Tensor[a,b,:,:]=UTmp

#populate F2Tensor
for itemTmp in ret2:
    b,J2Ind,UTmp=itemTmp
    F2Tensor[b,J2Ind,:,:]=UTmp

#populate F1Tensor
for itemTmp in ret1:
    a,UTmp=itemTmp
    F1Tensor[a,:,:]=UTmp



def UBarEigValAndVecs(abJ2Ind):
    a,b,J2Ind=abJ2Ind
    F3Tmp=F3Tensor[a,b,:,:]
    F2Tmp=F2Tensor[b,J2Ind,:,:]
    F1Tmp=F1Tensor[a,:,:]
    UTmp=F3Tmp@F2Tmp@F1Tmp
    eigs, vecs=np.linalg.eig(UTmp)
    phases=np.angle(eigs)
    inds=np.argsort(phases)
    phasesSorted=[phases[ind] for ind in inds]
    vecsSorted=[vecs[:,ind] for ind in inds]
    return [a,b,J2Ind,phasesSorted,vecsSorted]


abJ2IndAll=[[a,b,J2Ind] for a in range(0,N) for b in range(0,N) for J2Ind in range(0,maxIndJ2)]

tUBarStart=datetime.now()
pool3=Pool(threaNum)

retUBarEigAll=pool3.map(UBarEigValAndVecs,abJ2IndAll)
tUBarEnd=datetime.now()
print("UBar time: ",tUBarEnd-tUBarStart)

eigVecsTensor=np.zeros((maxIndJ2,N,N,2,2),dtype=complex)
tInitStart=datetime.now()
for itemTmp in retUBarEigAll:
    a,b,J2Ind,phases,vecs=itemTmp
    eigVecsTensor[J2Ind,a,b,:,0]=vecs[0]
    eigVecsTensor[J2Ind,a,b,:,1]=vecs[1]

tInitEnd=datetime.now()
print("init time: ",tInitEnd-tInitStart)

def cherNum(J2IndbandNum):
    J2Ind,bandNum=J2IndbandNum

    chTmp=0
    for a in range(0,N):
        for b in range(0,N):
            vecab=eigVecsTensor[J2Ind,a,b,:,bandNum]
            vecap1b=eigVecsTensor[J2Ind,(a+1)%N,b,:,bandNum]
            vecap1bp1=eigVecsTensor[J2Ind,(a+1)%N,(b+1)%N,:,bandNum]
            vecabp1=eigVecsTensor[J2Ind,a,(b+1)%N,:,bandNum]
            chTmp+=-np.angle(
                np.vdot(vecab,vecap1b)
                *np.vdot(vecap1b,vecap1bp1)
                *np.vdot(vecap1bp1,vecabp1)
                *np.vdot(vecabp1,vecab)
            )
    return [J2Ind,bandNum,chTmp/(2*np.pi)]


inChNumData=[[J2Ind,bandNum] for J2Ind in range(0,maxIndJ2) for bandNum in [0,1]]

pool4=Pool(threaNum)


t4Start=datetime.now()
ret4=pool4.map(cherNum,inChNumData)
t4End=datetime.now()

print("Chern number time: ",t4End-t4Start)

#data serialization

J2Band0=[]
J2Band1=[]
chNumBand0=[]
chNumBand1=[]
for itemTmp in ret4:
    J2Ind, bandNum,chTmp=itemTmp
    if bandNum==0:
        J2Band0.append(J2ValsAll[J2Ind]/np.pi)
        chNumBand0.append(chTmp)
    else:
        J2Band1.append(J2ValsAll[J2Ind]/np.pi)
        chNumBand1.append(chTmp)


plt.figure()
plt.plot(J2Band0,chNumBand0,color="blue",label="$C_{0}$",linestyle="-")

plt.plot(J2Band1,chNumBand1,color="red",label="$C_{1}$",linestyle="--")

xTickStart=int(J2ValsAll[0]/np.pi)
xTickEnd=int(J2ValsAll[-1]/np.pi)+1
plt.xticks(range(xTickStart,xTickEnd))
plt.xlabel("$J_{2}/\pi$")
plt.ylabel("Chern number")
plt.yticks([-1,0,1])


plt.legend()
plt.savefig("spectrum.png")
