import sys
import numpy as np
import os

if len(sys.argv) > 1:
    nombre_carpeta = sys.argv[1]

Err_Array = np.zeros(100)

np.savetxt(nombre_carpeta+"\\GMRES_Error.txt",Err_Array)