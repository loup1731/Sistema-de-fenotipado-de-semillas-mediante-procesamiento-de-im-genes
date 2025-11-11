# Copilot Instructions for AI Agents

## Project Overview
- This project performs seed phenotyping using image processing (Python).
- Main script: `Semillas_fenotipado_v1.py` handles both single-image and batch dataset processing.
- Input: Seed images (supports RAW and standard formats).
- Output: CSVs with extracted features, overlay images, debug masks, and cropped seed images.

## Key Components
- **Image Reading**: `_read_image_any` supports RAW via `rawpy` and standard formats via OpenCV.
- **Segmentation**: Multiple thresholding modes (Otsu, adaptive, HSV rules, or combined) with ROI masking.
- **Morphological Processing**: Cleans masks, fills holes, y aplica análisis de contornos.
	- Para cada contorno de semilla:
		- Se calcula el convex hull (envolvente convexa) usando `skimage.morphology.convex_hull_image` (por defecto) o `cv2.convexHull` si se requiere OpenCV.
		- Se obtiene el mínimo rectángulo contenedor (rotated min area rect) usando `cv2.minAreaRect`.
		- Se obtiene el bounding rect (rectángulo alineado a ejes) usando `cv2.boundingRect`.
- **Feature Extraction**: Calcula área, perímetro, ejes, convexidad, color, textura (GLCM), simetría y defectos superficiales para cada semilla, usando las envolventes y rectángulos calculados.
- **Batch Processing**: `process_dataset` procesa recursivamente subcarpetas y agrega resultados.

## Developer Workflows
- Run single-image or batch processing from CLI or interactively (e.g., Spyder/Jupyter).
- CLI usage: `python Semillas_fenotipado_v1.py --image <path>` or `--dataset_dir <dir>` with various options.
- Outputs are saved in the same folder as the input image or dataset.

## Project Conventions
- All measurements are in pixels.
- Masks and overlays are saved as PNGs; features as CSV.
- Debug and crop outputs are optional and controlled by CLI flags.
- Uses `skimage`, `opencv-python`, `numpy`, `pandas`, and optionally `rawpy`.
- Convex hulls for contours are computed using `skimage.morphology.convex_hull_image` (not OpenCV by default).

## Extending/Modifying
- To add new features, update the `process_image` function and ensure new columns are added to the output DataFrame.
- For new segmentation methods, extend `make_seed_mask`.
- For new output formats, modify the batch helpers and CLI output logic.

## Examples
- Ver `process_one_file` y `process_dataset` para patrones de procesamiento por lote e individual.
- Para convex hull: ver uso de `morphology.convex_hull_image` o `cv2.convexHull` en `process_image`.
- Para mínimo rectángulo contenedor: ver uso de `cv2.minAreaRect`.
- Para bounding rect: ver uso de `cv2.boundingRect`.

## External Integration
- No external APIs or services; all processing is local.
- RAW image support is optional and requires `rawpy`.

---
If you add new scripts or workflows, update this file to document new conventions or patterns.
