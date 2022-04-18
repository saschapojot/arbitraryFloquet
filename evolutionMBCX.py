import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from scipy.linalg import  expm
from pathlib import Path
from scipy.sparse import csc_matrix
############consts
J1=0.5*np.pi

J3=0.2*np.pi

M=1

sigma3 = np.array([[1, 0], [0, -1]])
sigma2 = np.array([[0, -1j], [1j, 0]])
sigma1 = np.array([[0, 1], [1, 0]])

N1 = 20
N2 = 20
Q=30
dt=1/(3*Q)

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


J2=1.5*np.pi
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




