
# Sistema-de-fenotipado-de-semillas-mediante-procesamiento-de-imágenes

Este proyecto implementa un sistema automatizado de fenotipado de semillas mediante procesamiento digital de imágenes. Permite extraer características morfológicas, de color y textura de semillas, facilitando el análisis cuantitativo y la comparación entre muestras. Está orientado a aplicaciones en agronomía, bancos de germoplasma e investigación científica.


## 📌 Objetivo

Desarrollar una herramienta robusta, precisa y reproducible para la extracción automática de características fenotípicas de semillas a partir de imágenes digitales, minimizando la intervención manual y controlando la variabilidad de medición.


## ⚙️ Características principales

- Procesamiento de imágenes individuales y por lotes.
- Soporte para imágenes RAW y formatos estándar (JPG, PNG, TIFF).
- Segmentación avanzada: Otsu, adaptativa, reglas HSV y combinaciones.
- Extracción de características:
  - Área, perímetro, ejes principales, circularidad, elongación, convexidad, simetría, defectos superficiales
  - Color (intensidad, saturación, colorfulness, homogeneidad)
  - Textura (GLCM)
- Cálculo de envolvente convexa (convex hull), mínimo rectángulo contenedor y bounding rect.
- Conversión de medidas de píxeles a milímetros mediante calibración con objetos de referencia.
- Evaluación de segmentación: métricas IoU, Dice, RMSE, MAE, R², etc.
- Exportación estructurada de resultados en formato CSV.
- Visualización de máscaras intermedias y overlays para trazabilidad.


## 🖼️ Ejemplo de flujo de procesamiento

1. Imagen original de semillas.
2. Aplicación de umbral adaptativo y máscara HSV.
3. Segmentación individual y etiquetado.
4. Extracción de métricas por semilla.
5. Conversión de unidades y exportación de resultados.


## 🛠️ Tecnologías utilizadas

- Python 3.7+
- OpenCV
- scikit-image
- numpy
- pandas
- imageio
- rawpy (opcional para RAW)


## 📸 Especificaciones del sistema de captura

- **Cámara:** Arducam 16MP Autofocus (sensor IMX519)
- **Resolución:** 4608 × 2592 px
- **Tamaño de píxel:** 1.4μm × 1.4μm
- **Distancia objeto-cámara:** 12 cm
- **Calibración:** basada en objetos de referencia conocidos (monedas colombianas medidas con calibrador digital)


## 🧪 Validación

Se utilizó una estrategia de validación cruzada con objetos de dimensiones conocidas para verificar precisión métrica del sistema. El algoritmo fue ajustado para minimizar error relativo en área, forma y color. Se emplean métricas como IoU, Dice, RMSE, MAE y R² para comparar la segmentación automática contra referencias geométricas.


## 📁 Estructura del repositorio

- `Semillas_fenotipado_v1.py`: Script principal de procesamiento y extracción de características.
- `App_Adquisicion.py`: Adquisición de imágenes.
- `Calibración_cam.py`: Calibración geométrica y óptica.
- `ColourGains_test.py`, `Viñeteo_Flatfield.py`: Pruebas de color y corrección de viñeteo.
- `Simulacion_domo_completaV1.py`, `sim_cams_curvas_resolucion.py`: Simulación y análisis óptico.

## 📦 Requisitos e instalación

Instala los paquetes necesarios con:
```bash
pip install opencv-python scikit-image numpy pandas imageio rawpy
```

## 🚀 Ejemplo de uso

Procesar una imagen:
```bash
python Semillas_fenotipado_v1.py --image <ruta_imagen>
```

Procesar un dataset:
```bash
python Semillas_fenotipado_v1.py --dataset_dir <ruta_carpeta>
```

## 📤 Salida del sistema

- CSV con características extraídas por semilla.
- Imágenes de depuración (máscaras, overlays).
- Recortes individuales de semillas.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request para sugerencias o mejoras.

