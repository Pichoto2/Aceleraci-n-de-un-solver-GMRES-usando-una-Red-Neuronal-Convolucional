from src_dir import mk_laplace_2d, mk_laplace_2d_Tensor
import torch
import numpy as np
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as pp
from mpl_toolkits.mplot3d import Axes3D  
from src_dir import Gauss_pdf_2D
from src_dir import cnn_preconditionerOnline_timed_2D, CNNPredictorOnline_2D,\
                    timer, GMRES, PreconditionerTrainer, CNN_30, FluidNet2D10, FluidNet2D30
import sys
import os
import shutil


# np.random.seed(0)
# torch.manual_seed(0) 

if len(sys.argv) > 1:
    nombre_carpeta = sys.argv[1]
    archivo_actual = os.path.abspath(__file__)
    archivo_copia = os.path.join(nombre_carpeta, "demo.py")
    shutil.copy2(archivo_actual, archivo_copia)




# ---------------------------------------------------------------------------------------------



# Set dimension of the NxN grid used
# Note: For optimal performance, the neural network "cnn_collectionOnline2D.py"
# can be tweaked with appropriate kernel dilations, however the code should
# still work and yield resuluts for any dimension of input provided
dim = 64 # Dimensión de las matrices b y x0

# Default initial guess used for direct un-preconditioned GMRES is the zero
# solution
x0 = np.squeeze(np.zeros((dim,dim))) # Inicializa x^0 como una matriz dim x dim de 0s
print(np.shape(x0))
x0Type = 'Zero Solution 2D'

# Tolerancia relativa para el algoritmo GMRES
e = 1e-3

# Restarted GMRES parameters
nmax_iter = 25 # Dimensión subespacio de Krylov
restart   = 10000 # Número de veces que resetea el algoritmo GMRES utilizando la última iteración como elemento de partida

# Create domain [-1,1]x[-1,1]
# Define grid values at midpoints of cartesian grid
DomainL = -1.0
DomainR =  1.0
dx = (DomainR-DomainL)/(dim-1)
x1 = np.linspace(DomainL+dx,DomainR-dx,dim)
x2 = np.linspace(DomainL+dx,DomainR-dx,dim)
X, Y = np.meshgrid(x1, x2, sparse=False, indexing='ij') # Malla
Area = (dx*(dim-1))**2 # Área de la malla

# Create 2D laplace opertor as a stencil opertor for a N-cell 2D grid
# Can be found in linop.py in src_di
A = mk_laplace_2d(dim, dim, bc="dirichlet", xlo=0, xhi=0, ylo=0, yhi=0)
AType = '2D Laplacian'

# Total number of steps (problemas) in simulation
n_steps = 10000



# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------




# Note: # Model dimention inputs are not used for the current network in cnn_predictorOnline2D.py (but must be passed into wrapper)
InputDim  = dim # dimension input NN
OutputDim = dim # dimension output NN
# Number of samples to collect before using preduction from Neural Network:
Initial_set = 32

nn_precon = CNNPredictorOnline_2D(InputDim, OutputDim, Area, dx, FluidNet2D30) # intialize the NN
# TODO: using diagnostic_probe=22 => end of first inner loop for the current setup => generalize this
trainer   = PreconditionerTrainer(nn_precon, Initial_set=Initial_set) # initialize the NN trainer

@timer # the next function will return the runtime
@cnn_preconditionerOnline_timed_2D(trainer) # the next function will be trained with trainer
def MLGMRES(A, b, x0, e, nmax_iter, restart, debug):
    return GMRES(A, b, x0, e, nmax_iter, restart, debug)

@timer
def GMRES_timed(A, b, x0, e, nmax_iter, restart, debug):
    return GMRES(A, b, x0, e, nmax_iter, restart, debug)




# ---------------------------------------------------------------------------------------------




from src_dir import Gauss_pdf_2D, resid, StatusPrinter

np.random.seed(0)

# Initialize lists that hold time data (time-to-solutuin, trainining time,
# MLGMRES time, etc)
run_time_ML_list     = []
GmresRunTimeOriginal = []
SpeedUp              = []
trainTime_list       = []

# Set debug mode to GMRES, save data of the inner loop of Arnoldi Iteration
debug = False

NonML_Err_List      = []
NonML_Err_List_Full = []

# Index of Poisson problems solved
for ProbIdx in range(n_steps):

    # Set RHS of Poisson problem
    # Construct b with a dipole
    xloc      = np.random.uniform(x1[0], x1[-1])
    yloc      = np.random.uniform(x2[0], x2[-1])
    xlocShift = np.random.uniform(-0.25, 0.25)
    ylocShift = np.random.uniform(-0.25, 0.25)
    AmplitudeFactor  = np.random.uniform(0.01, 10)
    AmplitudeFactor2 = AmplitudeFactor*np.random.uniform(1, 2)
    sigma = 0.07*np.random.uniform(0.9, 1.1)
    b     = AmplitudeFactor*Gauss_pdf_2D(X, Y, xloc, yloc, sigma) \
          + AmplitudeFactor2*Gauss_pdf_2D(X, Y, xloc + xlocShift, yloc + ylocShift, sigma)
    Field = np.random.normal(loc=0.0, scale=1.0, size=(dim,dim))
    Field = AmplitudeFactor*np.random.normal(loc=0.0, scale=1.0, size=(dim, dim))
    b = b + Field



    # b=np.maximum(xloc*2*X*np.sin(ProbIdx),yloc*2*Y*np.cos(ProbIdx))  # Linear gradient example
    b = b*(dx**2.0) # Finite difference grid spacing

    # First GMRES call (solve up to e1 tolerance) with ML wrapper
    trainer.ProbCount = ProbIdx  # TODO: This should probably be automatically incremented
    Out, run_time1_ML = MLGMRES(A, b, x0, e, nmax_iter, restart, debug) # GMRES with the trainer Cambiado **

    # Collect ML assisted Run-times
    run_time_ML_list.append(run_time1_ML)

    if len(trainer.trainTime) > 0:
        trainTime_list.append(trainer.trainTime[-1]) # TODO: this second list is not needed
        trainer.emptyTrainTime() # Cambiado **
    
    # Direct GMRES call up to e1 tolerance
    NonML_Out1, run_time1 = GMRES_timed(A, b, x0, e, nmax_iter, restart, debug)  # Cambiado **
    if debug:
        NonML_Err = resid(A, NonML_Out1, b)
    else:
        NonML_Err = resid(A, np.asarray([NonML_Out1[-1]]), b)
    NonML_Err_List.append(NonML_Err / np.linalg.norm(b)) # Cambiado **

    ## Collect  direct GMRES time
    GmresRunTimeOriginal.append(run_time1)

    ## Ratio of run-times
    SpeedUp.append(run_time1/trainer.ML_GMRES_Time_list[-1])
    
    # Update user on status
    # StatusPrinter().update_simulation(SpeedUp[-1], ProbIdx)


StatusPrinter().finalize()


MLGMRES_GMRES_ONLY = sum(trainer.ML_GMRES_Time_list)
run_time           = sum(GmresRunTimeOriginal)
run_time_ML        = sum(run_time_ML_list)
trainTime_total    = sum(trainTime_list)

# Cambiado **
trainInfo = trainer.trainInfo 
trainAccelList = []
diffTrains = []
prevTime = 0
prevIt = 0

for i, par in enumerate(trainInfo):
    if i != 0:
        trainAccelList.append(prevTime / par[-1])
        diffTrains.append(par[0] - prevIt)
    prevTime = par[-1]
    prevIt = par[0]

trainAccelAvg = np.mean(trainAccelList)
# Cambiado *

print("Runtime of Non-decorated version is: ",     run_time)
print("Runtime of MLGMRES decorator is: ",         run_time_ML)
print("Runtime of MLGMRES (only GMRES time) is: ", MLGMRES_GMRES_ONLY)
print("Runtime of training (backprop) is: ",       trainTime_total)
print("Number of tains: ",                         len(trainInfo)) # Cambiado **
print("Train acceleration avarage: ",              trainAccelAvg) # Cambiado **




# ---------------------------------------------------------------------------------------------




import matplotlib.pyplot as pp
from src_dir import moving_average

# Compute moving average of GMRES and MLGMRES error
AVG   = np.zeros((n_steps, 1))
count = np.arange(0, n_steps)

Err_Array = np.asarray(NonML_Err_List)
count     = np.arange(0, n_steps)
for j in range(0, n_steps):
    AVG[j] = moving_average(np.asarray(Err_Array[:j]), j)

Err_Array_ML = np.asarray(trainer.Err_list)
# Err_Array_ML = [i[-1] for i in trainer.Err_list] # Cambiado **
AVGML        = np.zeros((n_steps, 1))
for j in range(0, n_steps):
    AVGML[j] = moving_average(np.asarray(Err_Array_ML[:j]), j)    

# Compute moving average of GMRES and MLGMRES run-times
GmresRunTimeOriginal_AVG = np.zeros((n_steps, 1))
ML_GMRES_Time_AVG        = np.zeros((n_steps, 1))
ML_Total_Time_AVG        = np.zeros((n_steps, 1))

for j in range(0, n_steps):
    GmresRunTimeOriginal_AVG[j] = moving_average(np.asarray(GmresRunTimeOriginal[:j]), j)

for j in range(0, n_steps):
    ML_GMRES_Time_AVG[j] = moving_average(np.asarray(trainer.ML_GMRES_Time_list[:j]), j)

for j in range(0, n_steps):
    ML_Total_Time_AVG[j] = moving_average(np.asarray(run_time_ML_list[:j]), j)




# ---------------------------------------------------------------------------------------------




pp.plot(count,Err_Array_ML,'.b',label='MLGMRES error', alpha=0.2)
pp.plot(count[10:-1],AVGML[10:-1],'k',label='Average MLGMRES error', alpha=0.8)
pp.plot(count,Err_Array,'.r',label='GMRES error', alpha=0.2)
pp.plot(count[10:-1],AVG[10:-1],'g',label='Average GMRES error', alpha=0.5)

pp.xlabel('$i$')
pp.ylabel('$||r_2||_2$')
pp.title('Error as a function of $i$-th iteration')
pp.legend(loc='best')
pp.yscale("log")
pp.savefig(os.path.join(nombre_carpeta, "Error_Graph.png"))
pp.clf()

np.savetxt(os.path.join(nombre_carpeta, "GMRES_Error.txt"),Err_Array)
np.savetxt(os.path.join(nombre_carpeta, "MLGMRES_Error.txt"),Err_Array_ML)



# ----------------




# Cambiado **
pp.plot(run_time_ML_list,'.b',label='MLGMRES', alpha=0.2)
pp.plot(GmresRunTimeOriginal,'.r', label='GMRES', alpha=0.2)
pp.plot(count[10:-1],GmresRunTimeOriginal_AVG[10:-1],'g', label='GMRES Average', alpha= 0.8)
pp.plot(count[10:-1],ML_Total_Time_AVG[10:-1],'k', label='MLGMRES Average', alpha=0.5)

max_AVG = np.max(GmresRunTimeOriginal_AVG)

pp.ylabel('Time (s)')
pp.ylim([0,max_AVG + max_AVG/2])
pp.xlabel('i')
pp.title('Total run time')
pp.legend(loc='best')
pp.savefig(os.path.join(nombre_carpeta, "Total_Runtime_Graph.png"))
pp.clf()


# -------------------



pp.plot(trainer.ML_GMRES_Time_list,'.b',label='MLGMRES', alpha=0.2)
pp.plot(GmresRunTimeOriginal,'.r', label='GMRES', alpha=0.2)
pp.plot(count[10:-1],GmresRunTimeOriginal_AVG[10:-1],'g', label='GMRES Average', alpha=0.7)
pp.plot(count[10:-1],ML_GMRES_Time_AVG[10:-1],'k', label='MLGMRES Average', alpha=0.7)

pp.ylabel('Time (s)')
pp.ylim([0,None]) # Cambiado **
pp.xlabel('i')
pp.title('GMRES run time')
# pp.legend(loc='best')
pp.savefig(os.path.join(nombre_carpeta, "GMRES_Runtime_Graph.png"))
pp.clf()

np.savetxt(os.path.join(nombre_carpeta, "MLGMRES_Time.txt"),trainer.ML_GMRES_Time_list)
np.savetxt(os.path.join(nombre_carpeta, "GMRES_Time.txt"),GmresRunTimeOriginal)



# ----------------------



if debug:
    RHSIndex=-1
    pp.semilogy(NonML_Err_List_Full[RHSIndex],'.r',label='GMRES')
    pp.semilogy(trainer.Err_list[RHSIndex],'.b',label='MLGMRES ')
    pp.legend(loc='best')
    pp.xlabel('GMRES iterations')
    pp.ylabel('$||r||_2$')
    pp.title('Convergence of Algorithim for Final Linear Problem')
    pp.savefig(os.path.join(nombre_carpeta, "Convergence_Graph.png"))

    pp.clf()
    np.savetxt(os.path.join(nombre_carpeta, "GMRES_IterError.txt"),NonML_Err_List_Full[RHSIndex])
    np.savetxt(os.path.join(nombre_carpeta, "MLGMRES_IterError.txt"),trainer.Err_list[RHSIndex])



# ----------------------



GMRESAVG=GmresRunTimeOriginal_AVG[10:-1]
# Cambiado **
# MLGMRESAVG=ML_GMRES_Time_AVG[10:-1]
MLGMRESAVG=ML_Total_Time_AVG[10:-1]
Ratio=np.divide(GMRESAVG,MLGMRESAVG)

pp.plot(Ratio,'.b', alpha=0.5)
pp.xlabel('i')
pp.ylabel('GMRES/MLGMRES')
pp.title("NN Total Speed Up ")
pp.savefig(os.path.join(nombre_carpeta, "Total_Accel_Points.png"))
pp.clf()


# ----------------------


RedRatio = []

for i in range(len(Ratio)):
    ventana = Ratio[max(0, i-49):i+1]
    RedRatio.append(sum(ventana) / len(ventana))

pp.plot(np.arange(0,len(RedRatio)), RedRatio)
pp.xlabel('i = 1-'+str(n_steps)+'   |   cada 25')
pp.xticks([])
pp.ylabel('GMRES/MLGMRES')
pp.title("Total Speed Up ")
pp.savefig(os.path.join(nombre_carpeta, "Total_Accel_Lines.png"))
pp.clf()



# ----------------------


GMRESAVG=GmresRunTimeOriginal_AVG[10:-1]
# Cambiado **
MLGMRESAVG=ML_GMRES_Time_AVG[10:-1]
Ratio=np.divide(GMRESAVG,MLGMRESAVG)

pp.plot(Ratio,'.b', alpha=0.5)
pp.xlabel('i')
pp.ylabel('GMRES/MLGMRES')
pp.title("NN GMRES Speed Up ")
pp.savefig(os.path.join(nombre_carpeta, "GMRES_Accel.png"))
pp.clf()

np.savetxt(os.path.join(nombre_carpeta, "SpeedUp.txt"),Ratio)


# Cosas nuevas

pp.plot(diffTrains)
pp.xlabel('')
pp.ylabel('Difference')
pp.title("Evolution of iterations between trainers")
pp.savefig(os.path.join(nombre_carpeta, "diff_Train.png"))
