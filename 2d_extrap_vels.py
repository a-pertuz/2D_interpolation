"""
Este script realiza la interpolación 2D de datos de análisis de velocidad NMO desde un archivo dado y visualiza los resultados.
El script sigue estos pasos:
1. Solicita al usuario el nombre del archivo de entrada y los valores máximos para Trace y TWT.
2. Extrae el nombre de la línea del nombre del archivo de entrada.
3. Define los nombres de los archivos de entrada y salida.
4. Configura los parámetros de interpolación y los rangos para Trace y TWT.
5. Lee los datos del archivo de entrada.
6. Crea una malla 2D para los valores de Trace y TWT.
7. Realiza la interpolación utilizando un interpolador de Función de Base Radial (RBF).
8. Imprime los valores máximos y mínimos de las velocidades interpoladas.
9. Prepara y guarda los datos interpolados en un archivo de texto y un archivo binario.
10. Grafica las velocidades interpoladas y los puntos de datos originales, incluyendo anotaciones y una barra de color.
El script requiere las siguientes bibliotecas:
- numpy
- matplotlib
- scipy
- os
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import os

# --------------------- Solicitar entradas del usuario ---------------------
# Pedir al usuario el nombre del archivo y los valores máximos para Trace y TWT
archivo_velocidades = str(input("Introduce el nombre del archivo: "))
maximo_trace = int(input("Introduce el valor máximo para Trace: "))
maximo_twt = int(input("Introduce el valor máximo para TWT: "))

# Obtener el nombre de la línea del archivo (asumiendo que está en el nombre del archivo)
nombre_linea = archivo_velocidades.split('_')[0]

# -------------------- Definir archivo de entrada y salida --------------------
# Extraer el nombre base del archivo y crear el nombre para el archivo de salida
nombre_base_velocidades = os.path.splitext(os.path.basename(archivo_velocidades))[0]
archivo_interpolado = f'{nombre_base_velocidades}_interp_2D.dat'

# -------------------- Definir parámetros de la interpolación --------------------
# Crear los rangos para Trace y TWT
rango_trace = np.linspace(1, maximo_trace, maximo_trace)
rango_twt = np.arange(0, maximo_twt + 1, 4)  # Intervalo de 4 unidades para TWT

# --------------------- Leer los datos desde el archivo ---------------------
# Leer el archivo de datos, asumiendo que la primera fila es un encabezado
trazas, tiempos_twt, velocidades_nmo = np.loadtxt(archivo_velocidades, skiprows=1, unpack=True)

# ---------------------- Crear malla para interpolación ----------------------
# Crear malla 2D para los valores de Trace y TWT
malla_trazas, malla_twt = np.meshgrid(rango_trace, rango_twt)

# ------------------------ Realizar la interpolación ------------------------
# Crear interpolador RBF (Radial Basis Function) para extrapolación e interpolación
interpolador = RBFInterpolator(np.column_stack((trazas, tiempos_twt)), velocidades_nmo, kernel='linear', smoothing=10)

# Realizar la interpolación para la malla de Trace y TWT
velocidades_interpoladas = interpolador(np.column_stack((malla_trazas.ravel(), malla_twt.ravel()))).reshape(malla_trazas.shape)

# --------------------- Imprimir los valores máximos y mínimos ---------------------
# Obtener y mostrar los valores máximo y mínimo de las velocidades interpoladas
velocidad_maxima = np.max(velocidades_interpoladas)
velocidad_minima = np.min(velocidades_interpoladas)
print(f"Valor máximo de VNMO: {velocidad_maxima}")
print(f"Valor mínimo de VNMO: {velocidad_minima}")

# -------------------------- Preparar datos para guardar --------------------------
# Preparar los datos de salida para guardar en el archivo
datos_salida = np.column_stack((malla_trazas.ravel().astype(int), malla_twt.ravel().astype(int), velocidades_interpoladas.ravel()))
encabezado = 'Trace TWT VNMO'
formato = '%d %d %.2f'

# Guardar los datos interpolados en un archivo de texto
np.savetxt(archivo_interpolado, datos_salida, header=encabezado, fmt=formato, comments='')

# Guardar los datos interpolados en un archivo binario
archivo_binario = f'{nombre_base_velocidades}_interp_2D.bin'
velocidades_interpoladas.astype(np.float32).tofile(archivo_binario)

# ------------------------ Graficar los resultados -------------------------
# Crear una figura para la visualización
plt.figure(figsize=(14, 8))

# Graficar los contornos de las velocidades interpoladas
contorno = plt.contourf(malla_trazas, malla_twt, velocidades_interpoladas, levels=100, cmap='rainbow')

# Graficar los puntos originales como dispersión (scatter)
puntos_originales = plt.scatter(trazas, tiempos_twt, c=velocidades_nmo, cmap='rainbow', s=1)

# Añadir etiquetas con los valores de velocidad VNMO en cada punto original
for i, valor_vnmo in enumerate(velocidades_nmo):
    plt.annotate(f'{valor_vnmo:.0f}', (trazas[i], tiempos_twt[i]), va='center', ha='center', fontsize=8)

# Añadir barra de color (colorbar) para representar las velocidades VNMO
barra_color = plt.colorbar(contorno, orientation='vertical', extend='both', label='VNMO (m/s)')
barra_color.ax.invert_yaxis()  # Invertir la dirección de los ticks de la barra de color

# Ajustar el rango y los ticks del eje X
plt.xlim(-50, maximo_trace + 50)
plt.xticks(range(0, maximo_trace, 50))
plt.xticks(list(plt.xticks()[0]) + [maximo_trace])

# Añadir etiquetas y título al gráfico
plt.xlabel('Trace')
plt.ylabel('TWT (ms)')
plt.title(f'Interpolación del análisis de velocidad NMO de la línea {nombre_linea}')

# Invertir el eje Y para que el valor de TWT aumente hacia abajo
plt.gca().invert_yaxis()

# Mostrar el gráfico
plt.show()
