# opcion_A_preset_compatible.py
# -----------------------------------------------------------
# Barrido de tiempos de exposición "flicker-safe" con Picamera2
# y parámetros de captura manuales (ExposureTime, AnalogueGain,
# ColourGains). Guarda un JPG por tiempo y un TXT con metadatos.
#
# ¿Por qué 120 Hz?
# - En países con red eléctrica de 60 Hz (p. ej., Colombia), la mayoría
#   de luminarias alimentadas desde la red presentan modulación de luz
#   al doble: ~120 Hz (semiperiodos de 8.333... ms). Si el shutter NO
#   integra un número entero de semiperiodos, aparecen bandas (banding).
# - “Flicker-safe” => elegir tiempos de exposición múltiplos de 8.333 ms.
#
# "Flicker" => componente que detecta las fluctuaciones o cambios en la intensidad de la luz ambiental,
# especialmente las causadas por fuentes de iluminación artificial como los LEDs
#
# Notas importantes del pipeline:
# - Algunos controles (AeEnable/AwbEnable) pueden no “entrar” antes de
#   cam.start() dependiendo de la versión; por eso los fijamos después.
# - El sensor/ISP necesitan 1–2 frames para aplicar cambios. Aquí
#   “descartamos” 3 frames leyendo metadatos (equivale a esperar frames).
# - set_controls(ExposureTime) exige microsegundos enteros (int).
# - FrameDurationLimits se fija al mismo valor del ExposureTime para
#   asegurar que el shutter queped (evita que el driver recorte).
#
# ISO vs Gain:
# - En el EXIF del JPG verás un ISO aproximado (a veces redondeado).
#   Lo que realmente defines es AnalogueGain (≈ ISO/100).
#   Para reproducibilidad, confía en los metadatos de Picamera2:
#   ExposureTime_us_real y AnalogueGain_real.
# -----------------------------------------------------------

from picamera2 import Picamera2
import time, json, os

# === Parámetros de “flicker-safe” ===
# Si estás bajo iluminación de red de 60 Hz -> flicker ~120 Hz.
# Si estuvieras en 50 Hz -> usa 100.0 (semiperiodo = 10 ms).
FLICKER_HZ = 120.0
t_ms = 1000.0 / FLICKER_HZ   # semiperiodo en ms (8.333... ms para 120 Hz)

# Tiempos de exposición: múltiplos del semiperiodo (mitigan banding).
# Puedes añadir más (p. ej., 6*t_ms, 7*t_ms...) según tu rango de luz.
expos_ms = [1*t_ms, 2*t_ms, 3*t_ms, 4*t_ms, 5*t_ms]

# Ganancia analógica fija (factor, no “ISO”):
ganancia = 1.5  # ≈ ISO 150 teórico; el EXIF puede redondear a 200.

# ColourGains fijos (gains R,B). Puedes usar los medidos con tu ColorChecker
# o los que observaste con AWB auto para esta escena:
colour_gains_fijos = (1.32338, 1.75793)

# Carpeta de salida (se crea si no existe)
os.makedirs("serie_A", exist_ok=True)

with Picamera2() as cam:
    # Configuración de captura. Fijamos la resolución del flujo principal.
    cam.configure(cam.create_still_configuration(main={"size": (4056, 3040)}))

    resumen = []

    # Recorremos los tiempos de exposición deseados
    for ms in expos_ms:
        # Picamera2 usa microsegundos enteros (int) para ExposureTime
        us = int(round(ms * 1000))

        # 1) Prefijar lo que acepta antes de start():
        #    - AnalogueGain, ColourGains, ExposureTime y FrameDurationLimits
        #      suelen aplicarse bien antes de iniciar el stream.
        #    - Igual reforzaremos Ae/Awb después de start() (ver paso 2).
        cam.set_controls({
            "AnalogueGain": ganancia,
            "ColourGains": colour_gains_fijos,
            "ExposureTime": us,
            # Forzamos que la duración del frame (periodo) sea >= ExposureTime.
            # Aquí lo igualamos para evitar que el driver “recorte” el shutter.
            "FrameDurationLimits": (us, us)
        })

        # Iniciar el flujo de la cámara
        cam.start()

        # 2) Por compatibilidad de versiones: Ae/Awb suelen requerir estar
        #    “vivos” (streaming) para aceptarse. Los fijamos aquí en OFF.
        cam.set_controls({"AeEnable": False, "AwbEnable": False})

        # 3) Latencia de aplicación de controles / estado del ISP:
        #    tras un cambio de controles, los primeros 1–2 frames pueden
        #    salir con estado previo. Para asegurarnos, “dejamos pasar”
        #    3 frames leyendo metadatos (cada read espera al frame siguiente).
        cam.capture_metadata(); cam.capture_metadata(); cam.capture_metadata()

        # 4) Captura y nombre de archivo (evitamos el punto en el tiempo)
        tag = f"{ms:.3f}".replace(".", "p")
        jpg = f"serie_A/foto_{tag}ms_g{ganancia:.1f}.jpg"
        cam.capture_file(jpg)

        # 5) Leemos los metadatos REALES aplicados en esa toma
        md = cam.capture_metadata()
        fila = {
            "file": jpg,
            "ExposureTime_us_set": us,                   # lo que pedimos
            "ExposureTime_us_real": md.get("ExposureTime"),  # lo que se aplicó
            "AnalogueGain_real": md.get("AnalogueGain"),
            "ColourGains_real": md.get("ColourGains")
        }
        resumen.append(fila)

        # Guardamos un TXT con los metadatos junto al JPG (auditoría)
        with open(jpg.replace(".jpg", ".txt"), "w") as f:
            json.dump(fila, f, indent=2)

        print(fila)  # log en consola

        # 6) Cerramos el stream para la siguiente iteración (opción A)
        #    Ventaja: el primer frame de cada sesión ya nace con tus valores.
        cam.stop()

    # 7) Resumen global en JSON
    with open("serie_A/resumen.json", "w") as f:
        json.dump(resumen, f, indent=2)
