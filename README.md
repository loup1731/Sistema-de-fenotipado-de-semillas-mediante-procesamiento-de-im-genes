# Sistema de fenotipado de semillas mediante procesamiento de imágenes

Este proyecto implementa un sistema automatizado de fenotipado de semillas, basado en visión por computador, que permite identificar y cuantificar características morfológicas y cromáticas clave a partir de imágenes. Está orientado al análisis de accesiones conservadas en bancos de germoplasma para apoyar procesos de caracterización, clasificación y conservación.

## 📌 Objetivo

Desarrollar una herramienta robusta, precisa y reproducible para la extracción automática de características fenotípicas de semillas a partir de imágenes digitales, minimizando la intervención manual y controlando la variabilidad de medición.

## ⚙️ Funcionalidades principales

- Detección automática de semillas individuales en imágenes RGB.
- Corrección de condiciones de iluminación mediante normalización y filtros adaptativos.
- Umbralización avanzada con reglas en espacio de color HSV y métodos adaptativos (e.g., Gaussian).
- Extracción de características como:
  - Área (mm²)
  - Circularidad
  - Color (intensidad, saturación)
  - Número y distribución de manchas
- Exportación estructurada de resultados en formato `.csv` para análisis posteriores.
- Visualización de máscaras intermedias (`mask_hsv`, `mask_adapt`, `mask_final`) para trazabilidad del algoritmo.

## 🖼️ Ejemplo de flujo de procesamiento

1. Imagen original de semillas.
2. Aplicación de umbral adaptativo y máscara HSV.
3. Segmentación individual.
4. Extracción de métricas por semilla.

## 🛠️ Tecnologías utilizadas

- Python 3.10+
- OpenCV
- NumPy / SciPy
- Pandas
- Matplotlib
- Scikit-Image

## 📸 Especificaciones del sistema de captura

- **Cámara:** Arducam 16MP Autofocus (sensor IMX519)
- **Resolución:** 4608 × 2592 px
- **Tamaño de píxel:** 1.4μm × 1.4μm
- **Distancia objeto-cámara:** 12 cm
- **Calibración:** basada en objetos de referencia conocidos (monedas colombianas medidas con calibrador digital)

## 🧪 Validación

Se utilizó una estrategia de validación cruzada con objetos de dimensiones conocidas para verificar precisión métrica del sistema. El algoritmo fue ajustado para minimizar error relativo en área, forma y color.

## 📁 Estructura del repositorio

