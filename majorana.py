import functions as fs
import numpy as np


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

#Build Hamiltonian

#Build M-matrix

#Build A-matrix

#Diagonalize A-matrix

#Compute Liouvillian eigenvalues

#Compute observables through (46) and (47)

#Plot

