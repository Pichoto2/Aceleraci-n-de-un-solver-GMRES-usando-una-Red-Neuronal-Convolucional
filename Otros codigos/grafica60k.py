import sys
import numpy as np
import matplotlib.pyplot as pp
import torch
import os


ultimos = -20000

f_64I = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_60kP_64I.txt")
f_64R = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_60kP_64R.txt")

f_64I_red = f_64I[:ultimos]
f_64R_red = f_64R[:ultimos]

def maxPromedio(Ratio):
    RedRatio = []
    n = 1000

    for i in range(len(Ratio)):
        ventana = Ratio[max(0, i-(n-1)):i+1]   # últimos 10
        RedRatio.append(sum(ventana) / len(ventana))

    return RedRatio

x = np.arange(0,len(f_64I))
speedUpIt = np.mean(maxPromedio(f_64I_red))
speedUpRestart = np.mean(maxPromedio(f_64R_red))

print(speedUpIt)
print(speedUpRestart)

pp.figure()
pp.plot(x, maxPromedio(f_64I), label="GMRES-It")
pp.plot(x, maxPromedio(f_64R), label="GMRES-Reinicio")

pp.xlabel("$t$")
pp.ylabel("Aceleración GMRES (promedio)")

pp.legend()
pp.show()