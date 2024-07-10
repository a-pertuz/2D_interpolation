import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import os

# Solicitar al usuario introducir max_trace y max_twt
vels_file = str(input("Introduce el nombre del archivo: "))
max_trace = int(input("Introduce el valor máximo para Trace: "))
max_twt = int(input("Introduce el valor máximo para TWT: "))
nombre_linea = vels_file.split('_')[0]

# Archivo de entrada y salida
archivo_vels_base = os.path.splitext(os.path.basename(vels_file))[0]
file_interpolated = f'{archivo_vels_base}_interp_2D.dat'

# Parámetros de la interpolación
trace_range = np.linspace(1, max_trace, max_trace)
TWT_range = np.arange(0, max_twt + 1, 4)  # intervalo de 4 unidades

# Leer los datos desde el archivo
trace, TWT, VNMO = np.loadtxt(vels_file, skiprows=1, unpack=True)

# Crear una malla para la interpolación
trace_grid, TWT_grid = np.meshgrid(trace_range, TWT_range)

# Realizar la interpolación y extrapolación usando RBFInterpolator
interpolator = RBFInterpolator(np.column_stack((trace, TWT)), VNMO, kernel='linear', smoothing=10)
VNMO_grid = interpolator(np.column_stack((trace_grid.ravel(), TWT_grid.ravel()))).reshape(trace_grid.shape)

# Imprimir los valores máximos y mínimos del resultado
VNMO_max, VNMO_min = np.max(VNMO_grid), np.min(VNMO_grid)
print(f"Valor máximo de VNMO: {VNMO_max}")
print(f"Valor mínimo de VNMO: {VNMO_min}")

# Preparar datos para guardar en archivo
output_data = np.column_stack((trace_grid.ravel().astype(int), TWT_grid.ravel().astype(int), VNMO_grid.ravel()))
header = 'Trace TWT VNMO'
fmt = '%d %d %.2f'

# Guardar el resultado en un archivo de texto con columnas separadas por tabuladores
np.savetxt(file_interpolated, output_data, header=header, fmt=fmt, comments='')

# Graficar los resultados
plt.figure(figsize=(14, 8))
contour = plt.contourf(trace_grid, TWT_grid, VNMO_grid, levels=100, cmap='rainbow')
scatter_og = plt.scatter(trace, TWT, c=VNMO, cmap='rainbow',s=1)

# Agregar etiquetas con los valores Z
for i, label_vnmo in enumerate(VNMO):
    plt.annotate(f'{label_vnmo:.0f}', (trace[i], TWT[i]), va='center', ha='center', fontsize='8')


cbar = plt.colorbar(contour, orientation='vertical', extend='both', label='VNMO (m/s)')  # Extendemos la barra con extend='both'

# Ajustar los ticks de la barra de color
cbar.ax.invert_yaxis()  # Invierte la dirección de los ticks

plt.xlim(-50, max_trace+50)
plt.xticks(range(0, max_trace, 50))
plt.xticks(list(plt.xticks()[0]) + [max_trace])

plt.xlabel('Trace')
plt.ylabel('TWT (ms)')
plt.title('Interpolación del análisis de velocidad NMO de la línea ' + nombre_linea)
plt.gca().invert_yaxis()
plt.show()
