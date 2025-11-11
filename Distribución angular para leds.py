import numpy as np
import matplotlib.pyplot as plt

# Parámetros del LED
theta_total = 115  # grados, ángulo total
theta_half = theta_total / 2  # 57.5°
n_points = 500  # resolución angular

# Ángulo desde -57.5° a +57.5°
theta_deg = np.linspace(-theta_half, theta_half, n_points)
theta_rad = np.radians(theta_deg)

# Distribución de intensidad típica (asumimos forma coseno^n)
# Se usa n=30 para un haz moderadamente estrecho (simula 115° FWHM)
n = 30
I = np.cos(theta_rad) ** n
I = I / np.max(I)  # normalizar a 1

# Graficar distribución
plt.figure(figsize=(10, 5))
plt.plot(theta_deg, I, color='orange', lw=2)
plt.title('Distribución Angular de Intensidad (Haz de 115°)', fontsize=14)
plt.xlabel('Ángulo desde el eje central (°)')
plt.ylabel('Intensidad relativa (normalizada)')
plt.grid(True)
plt.axvline(-theta_half, color='gray', linestyle='--', label='-57.5°')
plt.axvline(theta_half, color='gray', linestyle='--', label='+57.5°')
plt.legend()
plt.tight_layout()
plt.show()
