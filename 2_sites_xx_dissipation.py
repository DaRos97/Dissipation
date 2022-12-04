import numpy as np
import matplotlib.pyplot as plt
import functions as fs
import sys
import getopt
import matplotlib.colors as mcolors
from tqdm import tqdm

argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv, "N:",["IC=","rate=","Jx=","Jy=","diag","Tf=","time_steps=","ac"])
    N = 2
    Jx = 1
    Jy = 1
    Tf = 10
    G = 0.1
    steps_t = 100
    diagonalize = False
    IC = "Zp"
    compute_ac = False
except:
    print("Error")
    exit()
for opt, arg in opts:
    if opt in ['-N']:
        N = int(arg)
    if opt == '--rate':
        G = float(arg)
    if opt == '--IC':
        IC = arg
    if opt == '--Jx':
        Jx = float(arg)
    if opt == '--Jy':
        Jy = float(arg)
    if opt == '--diag':
        diagonalize = True
    if opt == '--Tf':
        Tf = float(arg)
    if opt == '--time_steps':
        steps_t = int(arg)
    if opt == '--ac':
        compute_ac = True

print("Computing time evolution of Jx=",Jx,",Jy=",Jy,",decay rate=",G,", ",N," sites and initial condition=",IC)
#Initial Condition
IC_functions = {"IT":fs.ic_IT,"Zp":fs.ic_Zp,"Zm":fs.ic_Zm,"Xp":fs.ic_Xp,"Xm":fs.ic_Xm,"Yp":fs.ic_Yp,"Ym":fs.ic_Ym}
rho_0 = IC_functions[IC](N)
#
dirname = "Liouvillian_diagonalization_results/"

filename = dirname + "jx_jy_G_N=" + "{:2.2f}".format(Jx).replace('.','-') + "_" + "{:2.2f}".format(Jy).replace('.','-') + "_" + "{:2.2f}".format(G).replace('.','-') + "_" + str(N)
#Model: H = \sum_i(Jx*X_i*X_{i+1} + Jy*Y_i*Y_{i+1})     
#Dissipation:   decay --> jump operator = \sigma^-

if diagonalize:
    L = fs.Liouvillian(N,Jx,Jy,G)
    print("Liouvillian computed, now starting the diagonalization of ",2**(2*N)," x ",2**(2*N)," matrix")
    #Diagonalize liouvillian
    l,v = np.linalg.eig(L)
    print("Diagonalization completed")
    np.save(filename+'_eigenvalues.npy',l)
    np.save(filename+'_eigenvectors.npy',v)
else:
    l = np.load(filename+'_eigenvalues.npy')
    v = np.load(filename+'_eigenvectors.npy')

print("Computing constants and components of density matrix")
#Solve the differential equation:
#rho(t)[i] = \sum_n C_n*e^{l*t}*v_n[i]
#solve M*C = rho(0) --> matrix times vector = vector, we want to find C
C = np.linalg.solve(v,rho_0)
#compute the density matrix
T = np.linspace(0,Tf,steps_t)
dm_t = np.zeros((steps_t,2**(2*N)),dtype = complex)  #density matrix at all times -> [time,components]
for t_i,t in tqdm(enumerate(T)):
    for i in range(2**(2*N)):
        for n in range(2**(2*N)):
            dm_t[t_i,i] += C[n]*np.exp(l[n]*t)*v[i,n]       #the first index of "v" decides which component of rho I am time evolving
#####################
print("Now computing the observables")
op_z = fs.Z(N//2,N)
ev_opz = np.zeros(steps_t,dtype=complex)   #expectation value of sigma_z
purity = np.zeros(steps_t,dtype=complex)   #expectation value of sigma_x
for t in range(steps_t):
    rho_mat = np.reshape(dm_t[t],(2**N,2**N))
    ev_opz[t] = np.trace(np.matmul(rho_mat,op_z))
    purity[t] = np.trace(np.matmul(rho_mat,rho_mat))
if compute_ac:
    #autocorrelator
    rho_0p = np.tensordot(np.kron(op_z,fs.identity(N)),rho_0,axes=1)
    Cp = np.linalg.solve(v,rho_0p)
    dm_tp = np.zeros((steps_t,2**(2*N)),dtype = complex)  #density matrix at all times -> [time,components]
    for t_i,t in tqdm(enumerate(T)):
        for i in range(2**(2*N)):
            for n in range(2**(2*N)):
                dm_tp[t_i,i] += Cp[n]*np.exp(l[n]*t)*v[i,n]       #the first index of "v" decides which component of rho I am time evolving
    ac = np.zeros(steps_t,dtype=complex)   #expectation value of sigma_x
    for t in range(steps_t):
        rho_mat = np.reshape(dm_t[t],(2**N,2**N))
        rho_matp = np.reshape(dm_tp[t],(2**N,2**N))
        ac[t] = np.trace(np.matmul(np.matmul(rho_mat,op_z),rho_matp))

plt.figure(figsize = (12,6))
plt.suptitle("initial condition: "+IC)
plt.subplot(1,2,1)
plt.plot(T,np.real(ev_opz),label="<S^z>",color='r')
plt.plot(T,np.real(purity),label="purity",color='b')
if compute_ac:
    plt.plot(T,np.real(ac),label="autocorrelator",color='g')
plt.legend()

#Density matrix components
list_col = list(mcolors.TABLEAU_COLORS)
for i in range(5):
    list_col += list_col

plt.subplot(1,2,2)
#plt.figure(figsize = (12,12))
for c in range(2**(2*N)):
    plt.plot(T,np.real(dm_t[:,c]),color=list_col[c])#,label=str(c))
plt.plot(T,np.real(dm_t[:,-1]),color='r',label="|1><1|")
plt.plot(T,np.real(dm_t[:,0]),color='b',label="|0><0|")

plt.legend()
plt.show()





























