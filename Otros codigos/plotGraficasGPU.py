import matplotlib.pyplot as pp
import numpy as np
import pandas as pd


string = "12 7 8 7 5"
numeros = [int(x) for x in string.split()]
number_trains = np.array([12, 7, 8, 7, 5])
train_average = numeros / number_trains
media = np.mean(numeros)
print(media)


df_32R = pd.read_csv('.graficasGPU/gpu_log_D32R.csv')
df_32I = pd.read_csv('graficasGPU/gpu_log_D32I.csv')
df_64I = pd.read_csv('graficasGPU/gpu_log_D64I.csv')
df_64R = pd.read_csv('graficasGPU/gpu_log_D64R.csv')
df_32FI = pd.read_csv('graficasGPU/gpu_log_D32FI.csv')
df_32FR = pd.read_csv('graficasGPU/gpu_log_D32FR.csv')
df_64FI = pd.read_csv('graficasGPU/gpu_log_D64FI.csv')
df_64FR = pd.read_csv('graficasGPU/gpu_log_D64FR.csv')

media_no_nula1 = np.mean(df_32R.where(df_32R["gpu_util_pct"]!=0).dropna()["gpu_util_pct"])
media_no_nula2 = np.mean(df_32I.where(df_32I["gpu_util_pct"]!=0).dropna()["gpu_util_pct"])
media_no_nula_64I = np.mean(df_64I.where(df_64I["gpu_util_pct"]!=0).dropna()["gpu_util_pct"])
media_no_nula_64R = np.mean(df_64R.where(df_64R["gpu_util_pct"]!=0).dropna()["gpu_util_pct"])
media_no_nula_32FI = np.mean(df_32FI.where(df_32FI["gpu_util_pct"]!=0).dropna()["gpu_util_pct"])
media_no_nula_32FR = np.mean(df_32FR.where(df_32FR["gpu_util_pct"]!=0).dropna()["gpu_util_pct"])
media_no_nula_64FI = np.mean(df_64FI.where(df_64FI["gpu_util_pct"]!=0).dropna()["gpu_util_pct"])
media_no_nula_64FR = np.mean(df_64FR.where(df_64FR["gpu_util_pct"]!=0).dropna()["gpu_util_pct"])

print(np.mean(df_64FI["gpu_util_pct"]), np.mean(df_64FR["gpu_util_pct"]))

gpu_32R = df_32R.where(df_32R["gpu_util_pct"]!=0).dropna()["gpu_util_pct"]
gpu_32I = df_32I.where(df_32I["gpu_util_pct"]!=0).dropna()["gpu_util_pct"]
gpu_64R = df_64R.where(df_64R["gpu_util_pct"]!=0).dropna()["gpu_util_pct"]
gpu_64I = df_64I.where(df_64I["gpu_util_pct"]!=0).dropna()["gpu_util_pct"]
gpu_32FI = df_32FI.where(df_32FI["gpu_util_pct"]!=0).dropna()["gpu_util_pct"]
gpu_32FR = df_32FR.where(df_32FR["gpu_util_pct"]!=0).dropna()["gpu_util_pct"]
gpu_64FR = df_64FR.where(df_64FR["gpu_util_pct"]!=0).dropna()["gpu_util_pct"]
gpu_64FI = df_64FI.where(df_64FI["gpu_util_pct"]!=0).dropna()["gpu_util_pct"]

#gpu_32R = df_32R["gpu_util_pct"]
#gpu_32I = df_32I["gpu_util_pct"]
#gpu_32FR = df_32FR["gpu_util_pct"]
#gpu_32FI = df_32FI["gpu_util_pct"]

t1 = np.arange(0, len(gpu_32I))
t2 = np.arange(0, len(gpu_32R))
t3 = np.arange(0, len(gpu_64I))
t4 = np.arange(0, len(gpu_64R))
t5 = np.arange(0, len(gpu_32FI))
t6 = np.arange(0, len(gpu_32FR))
t7 = np.arange(0, len(gpu_64FI))
t8 = np.arange(0, len(gpu_64FR))

plot32I,=pp.plot(t1, gpu_32I, 'b', alpha=0.5, label="Dim-32/It CNN")
plot32R,=pp.plot(t2, gpu_32R, 'g', alpha=0.5, label="Dim-32/Restart CNN")
pp.legend(handles=[plot32I, plot32R])
pp.xlabel("Instantes de uso de GPU con \u0394t=1")
pp.ylabel("% GPU")
pp.show()

plot64I,=pp.plot(t3, gpu_64I, 'b', alpha=0.5, label="Dim-64/It")
plot64R,=pp.plot(t4, gpu_64R, 'g', alpha=0.5, label="Dim-64/Restart")
pp.legend(handles=[plot64I, plot64R])
pp.xlabel("Instantes de uso de GPU con \u0394t=1")
pp.ylabel("% GPU")
pp.show()

plot32FI,=pp.plot(t5, gpu_32FI, 'b', alpha=0.5, label="Dim-32/It FluidNet")
plot32FR,=pp.plot(t6, gpu_32FR, 'g', alpha=0.5, label="Dim-32/Restart FluidNet")
pp.legend(handles=[plot32FI, plot32FR])
pp.xlabel("Instantes de uso de GPU con \u0394t=1")
pp.ylabel("% GPU")
pp.show()

plot64FI,=pp.plot(t7, gpu_64FI, 'b', alpha=0.5, label="Dim-64/It")
plot64FR,=pp.plot(t8, gpu_64FR, 'g', alpha=0.5, label="Dim-64/Restart")
pp.legend(handles=[plot64FI, plot64FR])
pp.xlabel("Instantes de uso de GPU con \u0394t=1")
pp.ylabel("% GPU")
pp.show()