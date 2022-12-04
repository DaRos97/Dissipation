import numpy as np

def Liouvillian(N,Jx,Jy,G):
    H = np.zeros((2**N,2**N), dtype = complex)
    if N > 1:
        for i in range(N-1):
            X_1 = X(i,N)
            X_2 = X(i+1,N)
            Y_1 = Y(i,N)
            Y_2 = Y(i+1,N)
            H += np.matmul(X_1,X_2)*Jx        #element-wise multiplication
            H += np.matmul(Y_1,Y_2)*Jy
    #Hermitian part of Liouvillian
    L = -1j*np.kron(H,identity(N)) + 1j*np.kron(identity(N),H.T)

    #Limdblad part of Liouvillian
    D = np.zeros((2**(2*N),2**(2*N)), dtype = complex)
    for i in range(N):
        L_p = Sigma_minus(i,N)
        L_m = Sigma_plus(i,N)
        D += (G*np.kron(L_m,np.transpose(L_p))
              - G/2*np.kron(np.matmul(L_p,L_m),identity(N))
              - G/2*np.kron(identity(N),np.transpose(np.matmul(L_p,L_m)))
              )
    L += D
    return L


def identity(N):
    op_id = np.zeros((2**N,2**N), dtype = complex)
    for i in range(2**N):
        op_id[i,i] = 1
    return op_id

def Z(site,N):
    Z_site = np.array([[1,0],[0,-1]])
    temp = Z_site if site == 0 else identity(1)
    for i in range(1,N):
        if i == site:
            temp = np.kron(temp,Z_site)
        else:
            temp = np.kron(temp,identity(1))
    return temp

def X(site,N):
    X_site = np.array([[0,1],[1,0]])
    temp = X_site if site == 0 else identity(1)
    for i in range(1,N):
        if i == site:
            temp = np.kron(temp,X_site)
        else:
            temp = np.kron(temp,identity(1))
    return temp

def Y(site,N):
    Y_site = np.array([[0,-1j],[1j,0]])
    temp = Y_site if site == 0 else identity(1)
    for i in range(1,N):
        if i == site:
            temp = np.kron(temp,Y_site)
        else:
            temp = np.kron(temp,identity(1))
    return temp

def Sigma_plus(site,N):
    Sp_site = np.array([[0,2],[0,0]])
    temp = Sp_site if site == 0 else identity(1)
    for i in range(1,N):
        if i == site:
            temp = np.kron(temp,Sp_site)
        else:
            temp = np.kron(temp,identity(1))
    return temp

def Sigma_minus(site,N):
    Sm_site = np.array([[0,0],[2,0]])
    temp = Sm_site if site == 0 else identity(1)
    for i in range(1,N):
        if i == site:
            temp = np.kron(temp,Sm_site)
        else:
            temp = np.kron(temp,identity(1))
    return temp

############
############    Initial conditions
############

def ic_Zp(N):
    rho_0 = np.zeros(2**(2*N), dtype=complex)
    Z_i = np.array([1,0])#np.tensordot(np.array([[1,0],[0,-1]]),np.array([1,0]),1)
    if N == 1:
        return np.ravel(np.tensordot(Z_i,Z_i,0))
    rho_temp = np.kron(Z_i,Z_i)
    for i in range(1,N-1):
        rho_temp = np.kron(rho_temp,Z_i)
    rho_temp = np.tensordot(rho_temp,rho_temp,0)
    rho_0 = np.ravel(rho_temp)
    return rho_0
def ic_Zm(N):
    rho_0 = np.zeros(2**(2*N), dtype=complex)
    Z_i = np.tensordot(np.array([[1,0],[0,-1]]),np.array([0,1]),1)
    if N == 1:
        return np.ravel(np.tensordot(Z_i,Z_i,0))
    rho_temp = np.kron(Z_i,Z_i)
    for i in range(1,N-1):
        rho_temp = np.kron(rho_temp,Z_i)
    rho_temp = np.tensordot(rho_temp,rho_temp,0)
    rho_0 = np.ravel(rho_temp)
    return rho_0
#
def ic_Xp(N):
    rho_0 = np.zeros(2**(2*N), dtype=complex)
    X_i = 1/np.sqrt(2)*np.array([1,1])#np.tensordot(np.array([[0,1],[1,0]]),np.array([1,0]),1)
    if N == 1:
        return np.ravel(np.tensordot(X_i,X_i,0))
    rho_temp = np.kron(X_i,X_i)
    for i in range(1,N-1):
        rho_temp = np.kron(rho_temp,X_i)
    rho_temp = np.tensordot(rho_temp,rho_temp,0)
    rho_0 = np.ravel(rho_temp)
    return rho_0
def ic_Xm(N):
    rho_0 = np.zeros(2**(2*N), dtype=complex)
    X_i = 1/np.sqrt(2)*np.array([1,-1])#np.tensordot(np.array([[0,1],[1,0]]),np.array([1,0]),1)
    if N == 1:
        return np.ravel(np.tensordot(X_i,X_i,0))
    rho_temp = np.kron(X_i,X_i)
    for i in range(1,N-1):
        rho_temp = np.kron(rho_temp,X_i)
    rho_temp = np.tensordot(rho_temp,rho_temp,0)
    rho_0 = np.ravel(rho_temp)
    return rho_0
#
def ic_Yp(N):
    rho_0 = np.zeros(2**(2*N), dtype=complex)
    Y_i = 1/2*np.array([1-1j,1+1j])
    if N == 1:
        return np.ravel(np.tensordot(Y_i,np.conjugate(Y_i).T,0))
    rho_temp = np.kron(Y_i,Y_i)
    for i in range(1,N-1):
        rho_temp = np.kron(rho_temp,Y_i)
    rho_temp = np.tensordot(rho_temp,np.conjugate(rho_temp).T,0)
    rho_0 = np.ravel(rho_temp)
    return rho_0
def ic_Ym(N):
    rho_0 = np.zeros(2**(2*N), dtype=complex)
    Y_i = 1/2*np.array([1+1j,1-1j])
    if N == 1:
        return np.ravel(np.tensordot(Y_i,np.conjugate(Y_i).T,0))
    rho_temp = np.kron(Y_i,Y_i)
    for i in range(1,N-1):
        rho_temp = np.kron(rho_temp,Y_i)
    rho_temp = np.tensordot(rho_temp,np.conjugate(rho_temp).T,0)
    rho_0 = np.ravel(rho_temp)
    return rho_0
def ic_IT(N):
    rho_0 = np.zeros(2**(2*N), dtype=complex)
    for i in range(2**N):
        rho_0[i+2**N*i] = 1/2**N
    return rho_0


