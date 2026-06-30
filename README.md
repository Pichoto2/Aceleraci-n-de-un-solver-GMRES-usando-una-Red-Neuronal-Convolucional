# Aceleraci-n-de-un-solver-GMRES-usando-una-Red-Neuronal-Convolucional
Repositorio del TFM de Jorge Torrejón López con las modificaciones del código original: https://github.com/ML4FnP/GMRES-Learning. También se encuentran los datos de las diferentes pruebas realizadas.

----------------------------------------------------------------------

En este repositorio se encuentran los datos recopilados de las pruebas del TFM titulado "Aceleración de un solver precondicionado GMRES usando IA".

En la sección principal del drive se encuentran los diferentes documentos:
 - DiagramaGMRES.drawio: diagrama de flujo del funcionamiento del código + arquitectura de la red neuronal.
 - Accelerating.GMRES.with.NNs_2021.pdf: artículo del que se basa en su mayoría el TFM.

Por otro lado, se tienen diferentes carpetas donde se han ido almacenando las pruebas y los códigos:
 1. Otros códigos: post-procesado de aceleraciones o datos relacionados con la GPU (para la ejecución, cambiar la ruta de lectura de los ficheros (están en datosGPUyAceleraciones).
	- graficasPorTamano.py: se encarga de mostrar la relación de tiempo/tamaño y aceleración/tamaño con tolerancia 10^-3.
	- graficasPorTamanoTolerancia001.py: se encarga de mostrar la relación de tiempo/tamaño y aceleración/tamaño con tolerancia 10^-2.
	- plotGraficasGPU.py: se encarga de mostrar la gráfica de consumo de GPU y en la salida muestra métricas como la media no nula.
  	- graficasCurvasAprendizaje.py: se encarga de mostrar las curvas de aprendizaje de las diferentes redes neuronales.
  	- grafica60k.py: se encarga de mostrar la información gráfica de los casos con una población de 60k problemas.
  	- graficasNN.puy: se encargad de mostrar la información gráficas de los casos dependiendo de la red neuronal.

 2. GMRES-Learning-main: código principal del artículo con modificaciones.
	- run.sh: es el archivo para mandar el job, dentro hay una variable llamada carpeta para cambiar el nombre de la carpeta donde se va a guardar las salidas.
	- demo.py: script main del código, en este fichero se encuentran los parámetros importantes:
		* dim: dimensión de la malla del problema (dim^2 es la dimensión de la matriz).
		* e: tolerancia relativa del GMRES.
		* nmax_iter: dimensión del susbespacio de Krylov
		* restart: número máximo de restart si no alcanza antes la convergencia.
		* n_steps: número de problemas a resolver
	- sleep.py y run_sleep.sh: scripts auxiliares para las pruebas de CPU.
	- AdvectionDiffusion_Demo.ipynb: script notebook con el problema de difusión.
	- src_dir: carpeta con el resto de scripts:
		* gmres.py: contiene el algoritmo GMRES, para cambiar si una ejecución se quiere hacer cada restart, se debe comentar la línea 142 (x.append(x_sol + g))
		* cnn_predictorOnline2D.py: contiene las clases de entrenamiento de la red neuronal.
		* cnn_collectionOnline2D.py: contiene las clases con las estructuras de las diferentes redes neuronales.
		* util.py: funciones auxiliares comunes.
		* linop.py: funciones acerca del problema a resolver.
		* prueba.py: script auxiliar para pruebas.

Por último, el resto de carpetas contienen los resultados de las pruebas, dentro de "Pruebas en Turgalium" se tiene:
 - PruebasDim128: resultados con matrices de tamaño 16.384.
 - PruebasLearningRate: resultados cambiando el learning rate.
 - Pruebas30kProblemas: resultados de 30.000 problemas con tamaños diferentes con tolerancia 0.001 o 10^-3.
 - PruebasTol001: resultados de 30.000 problemas con tamaños diferentes con tolerancia 0.01 o 10^-2.
 - PruebasCores: resultados de las diferentes ejecuciones utilizando diferentes cores.
 - PruebasRedesNeuronales: resultados de las diferentes ejecuciones utilizando diferentes redes neuronales.
