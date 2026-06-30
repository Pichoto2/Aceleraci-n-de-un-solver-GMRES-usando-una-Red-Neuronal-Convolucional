import sys
import numpy as np
import matplotlib.pyplot as pp
import torch
import os

dims = [64, 256, 1024, 4096, 16384]


GMRESTimeIt = [94, 595, 4497, 57694, 1187652]
MLGMRESTotalTimeIt = [306, 1039, 2002, 19596, 556861]
TrainTimeIt = [179, 423, 629, 1054, 2059]
TrainRatioIt = [1.16, 3.68, 3.43, 4.35, 10.78]

x_pos = range(len(dims))


pp.figure()
pp.plot(x_pos, GMRESTimeIt, marker='o', label="GMRES")
#pp.plot(x_pos, GMRESTimeRestart, marker='o', label="GMRES-Restart")
pp.plot(x_pos, MLGMRESTotalTimeIt, marker='o', label="MLGMRES")
#pp.plot(x_pos, MLGMRESTotalTimeRestart, marker='o', label="MLGMRES-Restart")

pp.xticks(x_pos, dims)
pp.xlabel("$n$")
pp.ylabel("Tiempo de ejecución (s)")

pp.legend()
pp.show()

pp.figure()
pp.plot(x_pos, TrainRatioIt, marker='o')
#pp.plot(x_pos, TrainRatioRestart, marker='o', label="TrainRatio-Restart")
pp.xticks(x_pos, dims)

pp.xlabel("$n$")
pp.ylabel("Media Tiempo/Entrenamiento (s)")

pp.legend()
pp.show()

ultimos = -2000

f_8I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_8I.txt")
f_16I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_16I.txt")
f_32I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_32I.txt")
f_64I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_64I.txt")
f_128I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_128I.txt")

f_8I_red = f_8I[:ultimos]
f_16I_red = f_16I[:ultimos]
f_32I_red = f_32I[:ultimos]
f_64I_red = f_64I[:ultimos]
f_128I_red = f_128I[:ultimos]

def maxPromedio(Ratio):
    RedRatio = []
    n = 1000

    for i in range(len(Ratio)):
        ventana = Ratio[max(0, i-(n-1)):i+1]   # últimos 10
        RedRatio.append(sum(ventana) / len(ventana))

    return RedRatio

x_pos = range(len(dims))
x = np.arange(0,len(f_8I))
speedUpIt = [np.mean(maxPromedio(f_8I_red)), np.mean(maxPromedio(f_16I_red)), np.mean(maxPromedio(f_32I_red)), np.mean(maxPromedio(f_64I_red)), np.mean(maxPromedio(f_128I_red))]
print(speedUpIt)


pp.figure()
pp.plot(x_pos, speedUpIt, marker='o')
# pp.plot(x_pos, speedUpRestart, marker='o', label="GMRES-Restart")

pp.xticks(x_pos, dims)
pp.xlabel("$n$")
pp.ylabel("Aceleración GMRES (promedio)")

pp.legend()
pp.show()

pp.figure()
pp.plot(x, maxPromedio(f_8I), label="n=64")
pp.plot(x, maxPromedio(f_16I), label="n=256")
pp.plot(x, maxPromedio(f_32I), label="n=1024")
pp.plot(x, maxPromedio(f_64I), label="n=4096")
pp.plot(x, maxPromedio(f_128I), label="n=16384")

pp.xlabel("$t$")
pp.ylabel("Aceleración GMRES (promedio)")

pp.legend()
pp.show()