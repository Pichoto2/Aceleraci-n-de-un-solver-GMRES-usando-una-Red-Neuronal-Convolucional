import sys
import numpy as np
import matplotlib.pyplot as pp
import torch
import os


dims = ["FluidNet2D30", "CNN_30", "FluidNet2D10", "SingleDenseLayer"]

x_pos = range(len(dims))


ultimos = -2000
f_FN30 = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_FN30.txt")
f_CNN30 = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_CNN30.txt")
f_FN10 = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_FN10.txt")
f_SDL = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_SDL.txt")

f_FN30_red = f_FN30[:ultimos]
f_CNN30_red = f_CNN30[:ultimos]
f_FN10_red = f_FN10[:ultimos]
f_SDL_red = f_SDL[:ultimos]

x=np.arange(0, len(f_FN30)) 


def maxPromedio(Ratio):
    RedRatio = []
    n = 1000

    for i in range(len(Ratio)):
        ventana = Ratio[max(0, i-(n-1)):i+1]   # últimos 10
        RedRatio.append(sum(ventana) / len(ventana))

    return RedRatio

x_pos = range(len(dims))
speedUpIt = [np.mean(maxPromedio(f_FN30_red)), np.mean(maxPromedio(f_CNN30_red)), 
             np.mean(maxPromedio(f_FN10_red)), np.mean(maxPromedio(f_SDL_red))]
print(speedUpIt)

pp.figure()
pp.plot(x, maxPromedio(f_FN30), label="FluidNet2D30")
pp.plot(x, maxPromedio(f_CNN30), label="CNN_30")
pp.plot(x, maxPromedio(f_FN10), label="FluidNet2D10")
pp.plot(x, maxPromedio(f_SDL), label="SingleDenseLayer")


pp.xlabel("$t$")
pp.ylabel("Aceleración GMRES (promedio)")

pp.legend()
pp.show()


