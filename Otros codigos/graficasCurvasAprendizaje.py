import sys
import numpy as np
import matplotlib.pyplot as pp
import torch
import os

path_CNN30 = "E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/curvasAprendizaje_CNN30"
path_FN10 = "E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/curvasAprendizaje_FN10"
path_FN30 = "E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/curvasAprendizaje_FN30"
path_SDL = "E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/curvasAprendizaje_SDL"
path_64_FN30 = "E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/curvasAprendizaje_64_FN30"
path_64_CNN30 = "E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/curvasAprendizaje_64_CNN30"

dims = ["FluidNet2D30", "CNN_30", "FluidNet2D10", "SingleDenseLayer"]

x_pos = range(len(dims))

def getGraphic(path):
    res = []
    txt_list = os.listdir(path)

    for i, txt in enumerate(txt_list):
        f = np.loadtxt(os.path.join(path, txt))
        res.append(maxPromedio(f))
    return res


def maxPromedio(Ratio):
    RedRatio = []
    n = 200

    for i in range(len(Ratio)):
        ventana = Ratio[max(0, i-(n-1)):i+1]   # últimos 10
        RedRatio.append(sum(ventana) / len(ventana))

    return RedRatio

"""
graph_CNN30 = getGraphic(path_CNN30)
graph_FN30 = getGraphic(path_FN30)
graph_FN10 = getGraphic(path_FN10)
graph_SDL = getGraphic(path_SDL)

x = np.arange(0, len(graph_CNN30[0])) 
x_final = np.arange(0, len(graph_CNN30[-1]))


pp.figure()

pp.semilogy(x, graph_FN30[0], label=("FluidNet2D30"))
pp.semilogy(x, graph_CNN30[0], label=("CNN_30"))
pp.semilogy(x, graph_FN10[0], label=("FluidNet2D10"))
pp.semilogy(x, graph_SDL[0], label=("SingleDenseLayer"))

pp.xlabel("Épocas del primer entrenamiento")
pp.ylabel("Función de pérdida")
pp.legend()
pp.show()

pp.figure()

pp.semilogy(x_final, graph_FN30[-1], label=("FluidNet2D30"))
pp.semilogy(x_final, graph_CNN30[-1], label=("CNN_30"))
pp.semilogy(x_final, graph_FN10[-1], label=("FluidNet2D10"))
pp.semilogy(x_final, graph_SDL[-1], label=("SingleDenseLayer"))

pp.xlabel("Épocas del último entrenamiento")
pp.ylabel("Función de pérdida")

pp.legend()
pp.show()


pp.figure()

pp.semilogy(np.arange(0, len(graph_FN30[5])), graph_FN30[5], label=("FluidNet2D30"))
pp.semilogy(x, graph_CNN30[5], label=("CNN_30"))
pp.semilogy(np.arange(0, len(graph_FN10[5])), graph_FN10[5], label=("FluidNet2D10"))
pp.semilogy(x, graph_SDL[5], label=("SingleDenseLayer"))

pp.xlabel("Épocas del entrenamiento número 100")
pp.ylabel("Función de pérdida")

pp.legend()
pp.show()


f_128 = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/curvaAprendizaje_128I.txt")

graph_128I = maxPromedio(f_128)

x_128 = np.arange(0, len(graph_128I))

pp.figure()

pp.semilogy(x_128, graph_128I)

pp.xlabel("Épocas del primer entrenamiento")
pp.ylabel("Función de pérdida")

pp.legend()
pp.show()
"""



graph_64_CNN30 = getGraphic(path_64_CNN30)
graph_64_FN30 = getGraphic(path_64_FN30)

x = np.arange(0, len(graph_64_CNN30[0])) 

speedUp_FN30 = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_FN30_64I.txt")
speedUp_CNN30 = np.loadtxt("E:/Jorge/uni-master/Practicas CIEMAT/GMRES-Learning-main/datosGPUyAceleraciones/SpeedUp_CNN30_64I.txt")

problems = np.arange(0,len(speedUp_CNN30))
print(np.mean(maxPromedio(speedUp_CNN30)),np.mean(maxPromedio(speedUp_FN30)))

pp.figure()

pp.plot(problems, maxPromedio(speedUp_FN30), label="FluidNet2D30")
pp.plot(problems, maxPromedio(speedUp_CNN30), label="CNN_30")

pp.xlabel("Problema")
pp.ylabel("Aceleración GMRES (promedios)")

pp.legend()
pp.show()

pp.figure()

pp.semilogy(x, graph_64_FN30[0], label=("CA-FluidNet2D30"))
pp.semilogy(x, graph_64_CNN30[0], label=("CA-CNN_30"))

pp.xlabel("Épocas del primer entrenamiento")
pp.ylabel("Función de pérdida")
pp.legend()
pp.show()

pp.figure()

pp.semilogy(np.arange(0, len(graph_64_FN30[-1])), graph_64_FN30[-1], label=("CA-FluidNet2D30"))
pp.semilogy(np.arange(0, len(graph_64_CNN30[-1])), graph_64_CNN30[-1], label=("CA-0-CNN_30"))

pp.xlabel("Épocas del último entrenamiento")
pp.ylabel("Función de pérdida")

pp.legend()
pp.show()


pp.figure()

pp.semilogy(np.arange(0, len(graph_64_FN30[1])), graph_64_FN30[1], label=("CA-FluidNet2D30"))
pp.semilogy(np.arange(0, len(graph_64_CNN30[1])), graph_64_CNN30[1], label=("CA-CNN_30"))

pp.xlabel("Épocas del entrenamiento número 200")
pp.ylabel("Función de pérdida")

pp.legend()
pp.show()
