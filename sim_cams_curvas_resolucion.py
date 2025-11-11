# =============================================
# Simulación de posición y resolución de cámaras IMX708 e IMX219
# respecto a un domo, con visualización y cálculos geométricos
# =============================================

import numpy as np
import matplotlib.pyplot as plt
from math import radians, tan

# =============================
# 1. PARÁMETROS DE LAS CÁMARAS (NO TOCAR YA VIENE DE LAS ESPECIFICACIONES) 
# =============================

# Datos de la cámara IMX708 (Arducam)
imx708 = {
    "nombre": "IMX708 (Arducam)",
    "res_horizontal": 4608,  # número de píxeles horizontales
    "fov_deg": 66            # ángulo de campo de visión horizontal en grados
}

# Datos de la cámara IMX219
imx219 = {
    "nombre": "IMX219",
    "res_horizontal": 3280,  # número de píxeles horizontales
    "fov_deg": 62            # ángulo de campo de visión horizontal en grados
}

# Ancho de escena deseado en metros (15 cm)
W = 0.15                            # ESTE VALOR SE PUEDE CAMBIAR

# =============================
# 2. FUNCIONES DE CÁLCULO
# =============================

def calcular_parametros(cam, ancho_escena):
    # Convierte el FOV a radianes para poder usar funciones trigonométricas
    fov_rad = radians(cam["fov_deg"])
    # Aplica la fórmula de altura en función del ancho de escena y el FOV:
    # H = (W / 2) / tan(FOV / 2)
    H = (ancho_escena / 2) / tan(fov_rad / 2)
    # Calcula la resolución espacial (mm por píxel):
    # mm/px = (1000 * W) / resolucion_horizontal
    mm_por_px = (1000 * ancho_escena) / cam["res_horizontal"]
    return H, mm_por_px

# Calcular para ambas cámaras
h_imx708, mmpp_708 = calcular_parametros(imx708, W)
h_imx219, mmpp_219 = calcular_parametros(imx219, W)

# =============================
# 3. MOSTRAR RESULTADOS EN CONSOLA
# =============================

print("Resumen de posicionamiento y resolución:")
print("----------------------------------------")
print(f"{imx708['nombre']}:")
print(f"  Altura necesaria     : {h_imx708*100:.2f} cm")
print(f"  Resolución espacial  : {mmpp_708:.4f} mm/píxel")
print()
print(f"{imx219['nombre']}:")
print(f"  Altura necesaria     : {h_imx219*100:.2f} cm")
print(f"  Resolución espacial  : {mmpp_219:.4f} mm/píxel")

# =============================
# 4. VISUALIZACIÓN DEL DOMO Y LOS RAYOS
# =============================

def dibujar_campo_visual(fov_deg, h_camara, color, etiqueta):
    fov_rad = radians(fov_deg)
    half_fov = fov_rad / 2
    # Calcula los extremos del haz usando trigonometría:
    # x = +- tan(FOV/2) * H
    x_rayo_izq = [0, -np.tan(half_fov) * h_camara]
    x_rayo_der = [0,  np.tan(half_fov) * h_camara]
    y_rayo = [h_camara, 0]
    plt.plot(x_rayo_izq, y_rayo, color=color, linestyle='--', label=f'{etiqueta} - FOV')
    plt.plot(x_rayo_der, y_rayo, color=color, linestyle='--')

radio_domo = 0.075  # x cm              (ESTE RADIO SE PUEDE CAMBIAR)
theta = np.linspace(0, np.pi, 300)
x_domo = radio_domo * np.cos(theta)
y_domo = radio_domo * np.sin(theta)

plt.figure(figsize=(8, 6))
plt.plot(x_domo, y_domo, 'r', label='Perfil del domo')
plt.scatter(0, h_imx708, color='blue', label=f'Cámara IMX708 ({h_imx708*100:.1f} cm)')
plt.scatter(0, h_imx219, color='purple', label=f'Cámara IMX219 ({h_imx219*100:.1f} cm)')
dibujar_campo_visual(imx708["fov_deg"], h_imx708, 'blue', 'IMX708')
dibujar_campo_visual(imx219["fov_deg"], h_imx219, 'purple', 'IMX219')

plt.axhline(0, color='gray', linestyle='--')
plt.title('Ubicación de las cámaras respecto al domo (escena de 15 cm)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================
# 5. CURVAS DE RESOLUCIÓN VS ALTURA
# =============================

# Generar valores de altura entre 6 y 18 cm
h_vals = np.linspace(0.06, 0.15, 200)

# Función para calcular resolución espacial (mm/píxel) para cada cámara
def curva_mm_por_pixel(cam):
    fov_rad = radians(cam["fov_deg"])
    W_vals = 2 * h_vals * np.tan(fov_rad / 2)
    mmpp = (1000 * W_vals) / cam["res_horizontal"]
    return mmpp

mmpp_708_curve = curva_mm_por_pixel(imx708)
mmpp_219_curve = curva_mm_por_pixel(imx219)

# Graficar ambas curvas
plt.figure(figsize=(8, 5))
plt.plot(h_vals * 100, mmpp_708_curve, label='IMX708', color='blue')
plt.plot(h_vals * 100, mmpp_219_curve, label='IMX219', color='purple')
plt.axvline(h_imx708 * 100, linestyle='--', color='blue', alpha=0.5)
plt.axvline(h_imx219 * 100, linestyle='--', color='purple', alpha=0.5)
plt.title('Resolución espacial vs altura de cámara')
plt.xlabel('Altura de cámara [cm]')
plt.ylabel('mm por píxel')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
