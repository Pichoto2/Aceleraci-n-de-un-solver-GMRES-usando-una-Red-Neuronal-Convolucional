import sys
import numpy as np
import matplotlib.pyplot as pp
import torch
import os


dims = [64, 256, 784, 1024, 1600, 4096]


GMRESTimeIt = [447, 2530, 14038, 21508, 49003, 305675]
GMRESTimeRestart = [378, 2884, 15678, 21282, 49734, 309397]

MLGMRESTotalTimeIt = [2231, 3701, 7852, 10318, 18319, 103363]
MLGMRESTotalTimeRestart = [776, 2538, 8520, 10098, 20482, 115270]

TrainTimeIt = [1992, 2019, 1534, 1992, 1809, 2980]
TrainTimeRestart = [218, 136, 1548,2007, 2403, 1134]

TrainRatioIt = [3.5, 3.76, 3.36, 3.5, 3.47, 4.35]
TrainRatioRestart = [0.82, 3.67, 3.6, 3.66, 3.56, 4.37]

nTrainsIt = np.round(np.array(TrainTimeIt) / np.array(TrainRatioIt))
nTrainsRestart = np.round(np.array(TrainTimeRestart) / np.array(TrainRatioRestart))

x_pos = range(len(dims))


pp.figure()
pp.plot(x_pos, GMRESTimeIt, marker='o', label="$\epsilon_r=10^{-3}$")
#pp.plot(x_pos, GMRESTimeRestart, marker='o', label="GMRES-Reinicio")
pp.plot(x_pos, MLGMRESTotalTimeIt, marker='o', label="$\epsilon_r=10^{-2}$")
#pp.plot(x_pos, MLGMRESTotalTimeRestart, marker='o', label="MLGMRES-Reinicio")

pp.xticks(x_pos, dims)
pp.xlabel("$n$")
pp.ylabel("Tiempo de ejecución (s)")

pp.legend()
pp.show()

pp.figure()
pp.plot(x_pos, TrainRatioIt, marker='o')
#pp.plot(x_pos, TrainRatioRestart, marker='o', label="Cada Reinicio")
pp.xticks(x_pos, dims)

pp.xlabel("$n$")
pp.ylim(0, 4.45)
pp.ylabel("Media Tiempo/Entrenamiento (s)")

pp.legend()
pp.show()

pp.figure()
pp.plot(x_pos, nTrainsIt, marker='o', label="Cada It")
pp.plot(x_pos, nTrainsRestart, marker='o', label="Cada Reinicio")
pp.xticks(x_pos, dims)

pp.ylim(0)
pp.xlabel("$n$")
pp.ylabel("Número de Entrenamientos")

pp.legend()
pp.show()


ultimos = -10000
f_8I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_8I.txt")
f_8R = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_8R.txt")

f_16I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_16I.txt")
f_16R = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_16R.txt")

f_28I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_28I.txt")
f_28R = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_28R.txt")

f_32I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_32I.txt")
f_32R = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_32R.txt")

f_40I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_40I.txt")
f_40R = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_40R.txt")

f_64I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_64I.txt")
f_64R = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_64R.txt")

f_8I_red = f_8I[:ultimos]
f_8R_red = f_8R[:ultimos]

f_16I_red = f_16I[:ultimos]
f_16R_red = f_16R[:ultimos]

f_28I_red = f_28I[:ultimos]
f_28R_red = f_28R[:ultimos]

f_32I_red = f_32I[:ultimos]
f_32R_red = f_32R[:ultimos]

f_40I_red = f_40I[:ultimos]
f_40R_red = f_40R[:ultimos]

f_64I_red = f_64I[:ultimos]
f_64R_red = f_64R[:ultimos]

def maxPromedio(Ratio):
    RedRatio = []
    n = 1000

    for i in range(len(Ratio)):
        ventana = Ratio[max(0, i-(n-1)):i+1]   # últimos 10
        RedRatio.append(sum(ventana) / len(ventana))

    return RedRatio

x_pos = range(len(dims))
x = np.arange(0,len(f_8I))
speedUpIt = [np.mean(maxPromedio(f_8I_red)), np.mean(maxPromedio(f_16I_red)), np.mean(maxPromedio(f_28I_red)), np.mean(maxPromedio(f_32I_red)), np.mean(maxPromedio(f_40I_red)), np.mean(maxPromedio(f_64I_red))]
speedUpRestart = [np.mean(maxPromedio(f_8R_red)), np.mean(maxPromedio(f_16R_red)), np.mean(maxPromedio(f_28R_red)), np.mean(maxPromedio(f_32R_red)), np.mean(maxPromedio(f_40R_red)), np.mean(maxPromedio(f_64R_red))]

print(speedUpIt)
print(speedUpRestart)

pp.figure()
pp.plot(x_pos, speedUpIt, marker='o')
# pp.plot(x_pos, speedUpRestart, marker='o', label="Cada Reinicio")


pp.xticks(x_pos, dims)
pp.xlabel("$n$")
pp.ylabel("Aceleración GMRES (promedio)")

pp.legend()
pp.show()

pp.figure()
pp.plot(x, maxPromedio(f_8I), label="$n=64$")
pp.plot(x, maxPromedio(f_16I), label="$n=256$")
pp.plot(x, maxPromedio(f_28I), label="$n=784$")
pp.plot(x, maxPromedio(f_32I), label="$n=1024$")
pp.plot(x, maxPromedio(f_40I), label="$n=1600$")
pp.plot(x, maxPromedio(f_64I), label="$n=4096$")

pp.xlabel("$t$")
pp.ylabel("Aceleración GMRES (promedio)")

pp.legend()
pp.show()




dims = [64, 256, 1024, 4096, 16384]


GMRESTimeIt = [94, 595, 4426, 57694, 1187652]

MLGMRESTotalTimeIt = [306, 1039, 2519, 19596, 556861]

TrainTimeIt = [179, 423, 481, 1054, 2059]

TrainRatioIt = [1.16, 3.67, 3.53, 4.36, 10.78]

x_pos = range(len(dims))


pp.figure()
pp.plot(x_pos, GMRESTimeIt, marker='o', label="GMRES")
pp.plot(x_pos, MLGMRESTotalTimeIt, marker='o', label="MLGMRES")

pp.xticks(x_pos, dims)
pp.xlabel("$n$")
pp.ylabel("Tiempo de ejecución (s)")

pp.legend()
pp.show()

pp.figure()
pp.plot(x_pos, TrainRatioIt, marker='o')
pp.xticks(x_pos, dims)

pp.xlabel("$n$")
pp.ylim(0)
pp.ylabel("Media Tiempo/Entrenamiento (s)")

pp.legend()
pp.show()


ultimos = -2000
f_8I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_10kP_8I.txt")[:ultimos]
f_16I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_10kP_16I.txt")[:ultimos]
f_32I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_10kP_32I.txt")[:ultimos]
f_64I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_10kP_64I.txt")[:ultimos]
f_128I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_10kP_128I.txt")[:ultimos]

x_pos = range(len(dims))
speedUpIt = [maxPromedio(f_8I), maxPromedio(f_16I), maxPromedio(f_32I), maxPromedio(f_64I), maxPromedio(f_128I)]

pp.figure()
pp.plot(x_pos, speedUpIt, marker='o')

pp.xticks(x_pos, dims)
pp.xlabel("$n$")
pp.ylabel("Tiempo de ejecución (s)")

pp.legend()
pp.show()
"""

"""
dims = [64, 256, 784, 1024, 1600, 4096]


GMRESTimeIt_30kP = [365, 1888, 9828, 14074, 29330, 167859]
GMRESTimeRestart_30kP = [376, 1848, 9388, 13153, 33986, 170307]

MLGMRESTotalTimeIt_30kP = [1837, 2565, 4497, 7931, 12423, 42923]
MLGMRESTotalTimeRestart_30kP = [881, 2504, 5971, 7351, 9628, 48986]

TrainTimeIt_30kP = [1452, 1664, 1795, 1607, 1441, 2381]
TrainTimeRestart_30kP = [311, 1629, 1388, 1504, 1872, 2380]

TrainRatioIt_30kP = [3.38, 3.54, 3.5, 3.46, 3.5, 4.4]
TrainRatioRestart_30kP = [0.6, 3.5, 3.31, 3.44, 3.55, 4.35]

nTrainsIt_30kP = np.round(np.array(TrainTimeIt_30kP) / np.array(TrainRatioIt_30kP))
nTrainsRestart_30kP = np.round(np.array(TrainTimeRestart_30kP) / np.array(TrainRatioRestart_30kP))

x_pos = range(len(dims))


pp.figure()
pp.plot(x_pos, GMRESTimeIt_30kP, marker='o', label="GMRES-It")
pp.plot(x_pos, GMRESTimeRestart_30kP, marker='o', label="GMRES-Reinicio")
pp.plot(x_pos, MLGMRESTotalTimeIt_30kP, marker='o', label="MLGMRES-It")
pp.plot(x_pos, MLGMRESTotalTimeRestart_30kP, marker='o', label="MLGMRES-Reinicio")

pp.xticks(x_pos, dims)
pp.xlabel("$n$")
pp.ylabel("Tiempo de ejecución (s)")

pp.legend()
pp.show()

pp.figure()
pp.plot(x_pos, TrainRatioIt_30kP, marker='o', label="Cada It")
pp.plot(x_pos, TrainRatioRestart_30kP, marker='o', label="Cada Reinicio")
pp.xticks(x_pos, dims)

pp.xlabel("$n$")
pp.ylim(0)
pp.ylabel("Media Tiempo/Entrenamiento (s)")

pp.legend()
pp.show()

ultimos = -5000

f_30kP_8I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_30kP_8I.txt")
f_30kP_8R = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_30kP_8R.txt")

f_30kP_16I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_30kP_16I.txt")
f_30kP_16R = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_30kP_16R.txt")

f_30kP_28I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_30kP_28I.txt")
f_30kP_28R = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_30kP_28R.txt")

f_30kP_32I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_30kP_32I.txt")
f_30kP_32R = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_30kP_32R.txt")

f_30kP_40I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_30kP_40I.txt")
f_30kP_40R = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_30kP_40R.txt")

f_30kP_64I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_30kP_64I.txt")
f_30kP_64R = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_T001_30kP_64R.txt")

f_30kP_8I_red = f_30kP_8I[:ultimos]
f_30kP_8R_red = f_30kP_8R[:ultimos]

f_30kP_16I_red = f_30kP_16I[:ultimos]
f_30kP_16R_red = f_30kP_16R[:ultimos]

f_30kP_28I_red = f_30kP_28I[:ultimos]
f_30kP_28R_red = f_30kP_28R[:ultimos]

f_30kP_32I_red = f_30kP_32I[:ultimos]
f_30kP_32R_red = f_30kP_32R[:ultimos]

f_30kP_40I_red = f_30kP_40I[:ultimos]
f_30kP_40R_red = f_30kP_40R[:ultimos]

f_30kP_64I_red = f_30kP_64I[:ultimos]
f_30kP_64R_red = f_30kP_64R[:ultimos]

x_pos = range(len(dims))
x = np.arange(0,len(f_30kP_8I))
speedUpIt_30kP = [np.mean(maxPromedio(f_30kP_8I_red)), np.mean(maxPromedio(f_30kP_16I_red)), np.mean(maxPromedio(f_30kP_28I_red)), np.mean(maxPromedio(f_30kP_32I_red)), np.mean(maxPromedio(f_30kP_40I_red)), np.mean(maxPromedio(f_30kP_64I_red))]
speedUpRestart_30kP = [np.mean(maxPromedio(f_30kP_8R_red)), np.mean(maxPromedio(f_30kP_16R_red)), np.mean(maxPromedio(f_30kP_28R_red)), np.mean(maxPromedio(f_30kP_32R_red)), np.mean(maxPromedio(f_30kP_40R_red)), np.mean(maxPromedio(f_30kP_64R_red))]
print(speedUpIt_30kP)
print(speedUpRestart_30kP)



pp.figure()
pp.plot(x_pos, speedUpIt_30kP, marker='o', label="Cada It")
pp.plot(x_pos, speedUpRestart_30kP, marker='o', label="Cada Reinicio")

pp.xticks(x_pos, dims)
pp.xlabel("$n$")
pp.ylabel("Aceleración GMRES (promedio)")

pp.legend()
pp.show()

pp.figure()
pp.plot(x_pos, nTrainsIt_30kP, marker='o', label="Cada It")
pp.plot(x_pos, nTrainsRestart_30kP, marker='o', label="Cada Reinicio")
pp.xticks(x_pos, dims)

pp.xlabel("$n$")
pp.ylim(0)
pp.ylabel("Número de Entrenamientos")

pp.legend()
pp.show()

pp.figure()
pp.plot(x_pos, TrainRatioIt_30kP, marker='o', label="Cada It")
pp.plot(x_pos, TrainRatioRestart_30kP, marker='o', label="Cada Reinicio")
pp.xticks(x_pos, dims)

pp.xlabel("$n$")
pp.ylabel("Media Tiempo/Entrenamiento (s)")

pp.legend()
pp.show()

pp.figure()
pp.plot(x, maxPromedio(f_30kP_8I), label="$n=64$")
pp.plot(x, maxPromedio(f_30kP_16I), label="$n=256$")
pp.plot(x, maxPromedio(f_30kP_28I), label="$n=784$")
pp.plot(x, maxPromedio(f_30kP_32I), label="$n=1024$")
pp.plot(x, maxPromedio(f_30kP_40I), label="$n=1600$")
pp.plot(x, maxPromedio(f_30kP_64I), label="$n=4096$")

pp.xlabel("$t$")
pp.ylabel("Aceleración GMRES (promedio)")

pp.legend()
pp.show()
