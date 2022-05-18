import numpy as np
from datetime import datetime
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm
import matplotlib.pyplot as plt
from multiprocessing import Pool

tStart=datetime.now()
J1 = 0.5 * np.pi
J3 = 0.2 * np.pi
rangeJ2=range(0,12)
J2ValsAll=[0.5*j*np.pi for j in rangeJ2]

M = 1
sigma0=np.array([[1,0],[0,1]],dtype=complex)
sigma3 = np.array([[1, 0], [0, -1]],dtype=complex)
sigma2 = np.array([[0, -1j], [1j, 0]],dtype=complex)
sigma1 = np.array([[0, 1], [1, 0]],dtype=complex)

Nx = 70
Ny = 70

#spatial part of H1
S1=np.zeros((Nx*Ny,Nx*Ny),dtype=complex)
for ny in range(0,Ny):
    for nx in range(0,Nx-1):
        S1[nx*Ny+ny,(nx+1)*Ny+ny]=1
        S1[(nx+1)*Ny+ny,nx*Ny+ny]=-1

H1=np.kron(3*J1/(2*1j)*S1,sigma1)
H1Sparse=csc_matrix(H1)
U1=expm(-1j*1/3*H1Sparse)
U1Dense=U1.toarray()
#spatial part of H2
S2=np.zeros((Nx*Ny,Nx*Ny),dtype=complex)
for nx in range(0,Nx):
    for ny in range(0,Ny-1):
        S2[nx*Ny+ny,nx*Ny+ny+1]=1
        S2[nx*Ny+ny+1,nx*Ny+ny]=-1

def H2MatSparse(j):
    J2Tmp=J2ValsAll[j]
    return csc_matrix(np.kron(3*J2Tmp/(2*1j)*S2,sigma2))
def U2DenseMat(j):
    U2Tmp=expm(-1j*1/3*H2MatSparse(j))
    return U2Tmp.toarray()

#spatial part of H3

S3=np.zeros((Nx*Ny,Nx*Ny),dtype=complex)
#part 1
for nx in range(0,Nx):
    for ny in range(0,Ny):
        S3[nx*Ny+ny,nx*Ny+ny]=2*M

#part 2
for ny in range(0,Ny):
    for nx in range(0,Nx-1):
        S3[nx*Ny+ny,(nx+1)*Ny+ny]=1
        S3[(nx+1)*Ny+ny,nx*Ny+ny]=1
#part 3
for nx in range(0,Nx):
    for ny in range(0,Ny-1):
        S3[nx*Ny+ny,nx*Ny+ny+1]=1
        S3[nx*Ny+ny+1,nx*Ny+ny]=1

H3=np.kron(3*J3/2*S3,sigma3)
H3Sparse=csc_matrix(H3)
U3=expm(-1j*1/3*H3Sparse)
U3Dense=U3.toarray()

def UMat(j):
    UTmp=U3Dense@U2DenseMat(j)@U1Dense
    return UTmp

EValsAll=[0.065*np.pi,0.935*np.pi]

pSp=np.zeros((2*Ny,Nx*Ny),dtype=complex)
for n in range(0,Ny):
    pSp[n,n]=1
for n in range(1,Ny+1):
    pSp[-n,-n]=1

P=np.kron(pSp,sigma0)

PTP=P.T@P

def G(EIndj):
    EInd,j=EIndj
    E=EValsAll[EInd]
    U=UMat(j)
    nrow,ncol=U.shape
    id=np.eye(nrow,dtype=complex)
    S=P@np.linalg.inv(id-np.exp(1j*E)*U@(id-PTP))*np.exp(1j*E)@U@P.T
    tE=S[0:(2*Ny),(-2*Ny):]
    return [EInd,j,np.real(np.trace(tE.T.conj()@tE))]

threadNum=24

inDataAll=[[EInd, j] for EInd in [0,1] for j in rangeJ2]

pool0=Pool(threadNum)
retAll=pool0.map(G,inDataAll)

tEnd=datetime.now()
print("computation time: ",tEnd-tStart)

GE0=[]
GEpi=[]
for itemTmp in retAll:
    EInd,j,GTmp=itemTmp
    if EInd==0:
        GE0.append(GTmp)
    else:
        GEpi.append(GTmp)

plt.figure()
J2Plt=[elem/np.pi for elem in J2ValsAll]
plt.plot(J2Plt,GE0,color="black")
plt.plot(J2Plt,GEpi,color="red")
plt.savefig("tmp.png")
