from picamera2 import Picamera2
import cv2
import numpy as np

picam2 = Picamera2()
# Configurar modo de captura (ej: vista previa a baja resolución para rapidez)
picam2.configure(picam2.create_still_configuration(main={"size": (640, 480)}))
picam2.start()
# Esperar un poco para auto exposición (si está en auto)
import time; time.sleep(2)

# Desactivar AWB poniendo ganancias neutrales
picam2.set_controls({"ColourGains": (1.0, 1.0)})

# Capturar una imagen con la carta gris en escena
frame = picam2.capture_array()  # Devuelve un array NumPy con imagen BGR
# Definir ROI (ejemplo: cuadrante central donde colocamos la carta)
h, w, _ = frame.shape
roi = frame[h//4:3*h//4, w//4:3*w//4]  # (Esto debe ajustarse según la posición real de la carta)

# Calcular promedio BGR en la ROI
B_avg, G_avg, R_avg = cv2.mean(roi)[:3]  # mean devuelve (B,G,R,alpha)
# Calcular ganancias necesarias (tomando verde como referencia)
if R_avg > 0 and B_avg > 0:
    r_gain = G_avg / R_avg
    b_gain = G_avg / B_avg
else:
    r_gain = 1.0
    b_gain = 1.0  # Evitar división por cero en caso extremo
print(f"Ganancias calculadas - Roja: {r_gain:.2f}, Azul: {b_gain:.2f}")

# Aplicar las nuevas ganancias de balance de blancos
picam2.set_controls({"ColourGains": (r_gain, b_gain)})

# (Opcional) Tomar otra imagen para comprobar resultado
frame_corrected = picam2.capture_array()
roi_corr = frame_corrected[h//4:3*h//4, w//4:3*w//4]
B2, G2, R2 = cv2.mean(roi_corr)[:3]
print(f"Después de corrección - Promedio ROI -> R: {R2:.1f}, G: {G2:.1f}, B: {B2:.1f}")
