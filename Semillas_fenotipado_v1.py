#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fenotipado de semillas (1–5 semillas, fondo blanco)
- Unidades: PIXELES.
- SINGLE y DATASET.
- Umbralización: Otsu, Adaptativa (mean/gaussian), Regla HSV, o HSV+Adaptativa.
- Debug: guarda mask_hsv, mask_adapt y mask_final por imagen.

Ajustes clave:
- Adaptive en canal V (HSV) con GaussianBlur 5x5 (menos “sal y pimienta”).
- Regla HSV con V_min y V_max (fondo claro controlado).
- Fusión OR/AND configurable.
- ROI aplicado a cada máscara antes de fusionar.
- Morfología reforzada + fill holes por flood fill.
- min_obj / min_hole relativos si se pasan <= 0.
"""

import argparse, os, glob
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, morphology, color

# ---------------- RAW support ----------------
try:
    import rawpy
except Exception:
    rawpy = None

RAW_EXTS = {".dng",".nef",".cr2",".cr3",".arw",".rw2",".orf",".raf",".srw",".pef"}

def _read_image_any(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ruta no existe: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTS:
        if rawpy is None:
            raise RuntimeError("RAW detectado. Instala:  pip install rawpy imageio")
        with rawpy.imread(path) as raw:
            rgb16 = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16)
        rgb8 = (rgb16 >> 8).astype(np.uint8)
        return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return bgr

def _ensure_dir(p):
    if p:
        os.makedirs(p, exist_ok=True)
    return p

# --------- Compatibilidad GLCM ----------
try:
    from skimage.feature import graycomatrix, graycoprops
except Exception:
    try:
        from skimage.feature.texture import graycomatrix, graycoprops
    except Exception:
        from skimage.feature import greycomatrix as graycomatrix
        from skimage.feature import greycoprops as graycoprops

# --------- Perímetro Crofton -----------
try:
    from skimage.measure import perimeter_crofton as _perimeter_crofton
    HAS_CROFTON = True
except Exception:
    HAS_CROFTON = False

# ---------------- Utilidades ----------------
def safe_div(a, b, default=np.nan):
    return a / b if (b not in (0, None) and np.isfinite(b) and b != 0) else default

def hasler_susstrunk_colorfulness(img_bgr_masked):
    B, G, R = cv2.split(img_bgr_masked.astype(np.float32))
    rg = np.abs(R - G); yb = np.abs(0.5*(R + G) - B)
    return np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3*np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)

def dice_coeff(a, b):
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    return safe_div(2.0 * inter, a.sum() + b.sum())

def reflect_mask_over_major_axis(mask, region):
    rr, cc = np.nonzero(mask)
    coords = np.vstack((cc, rr, np.ones_like(rr))).astype(np.float32)
    cy, cx = region.centroid
    theta = region.orientation
    T1 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], np.float32)
    R  = np.array([[ np.cos(-theta),-np.sin(-theta),0],
                   [ np.sin(-theta),  np.cos(-theta),0],
                   [0,0,1]], np.float32)
    M_reflect = np.array([[-1,0,0],[0,1,0],[0,0,1]], np.float32)
    T2 = np.array([[1,0,cx],[0,1,cy],[0,0,1]], np.float32)
    M = T2 @ (R @ (M_reflect @ (np.linalg.inv(R) @ T1)))
    coords_ref = M @ coords
    x_ref = np.round(coords_ref[0]).astype(int)
    y_ref = np.round(coords_ref[1]).astype(int)
    h, w = mask.shape
    valid = (x_ref>=0)&(x_ref<w)&(y_ref>=0)&(y_ref<h)
    ref_mask = np.zeros_like(mask, bool)
    ref_mask[y_ref[valid], x_ref[valid]] = True
    return ref_mask

def estimate_volume_ellipsoid(a_px, b_px, c_px):
    return (4.0/3.0) * np.pi * (a_px/2.0) * (b_px/2.0) * (c_px/2.0)

def detect_surface_defects(L_norm, seed_mask, sigma=2.0, thresh=0.06):
    L_blur = cv2.GaussianBlur(L_norm, (0,0), sigmaX=sigma, sigmaY=sigma)
    diff = np.abs(L_norm - L_blur)
    local = cv2.GaussianBlur(diff, (0,0), sigmaX=sigma*2, sigmaY=sigma*2) + 1e-6
    score = diff / local; score[~seed_mask] = 0
    defect_mask = score > thresh
    defect_mask = morphology.remove_small_objects(defect_mask, min_size=8)
    defect_mask = morphology.remove_small_holes(defect_mask, area_threshold=16)
    signed = (L_blur - L_norm)
    dark_mask  = (signed >  thresh) & seed_mask
    light_mask = (signed < -thresh) & seed_mask
    _, n_def   = ndi.label(defect_mask)
    _, n_dark  = ndi.label(dark_mask)
    _, n_light = ndi.label(light_mask)
    return {
        "defect_area_ratio": defect_mask.sum() / max(seed_mask.sum(), 1),
        "defect_count": int(n_def),
        "dark_spot_count": int(n_dark),
        "light_spot_count": int(n_light),
    }

def glcm_features(gray_u8_masked):
    g = gray_u8_masked.astype(np.uint8, copy=True)
    mask = g > 0
    if mask.any():
        g[~mask] = int(np.median(g[mask]))
    else:
        return {k: np.nan for k in
                ["glcm_contrast","glcm_homogeneity","glcm_energy",
                 "glcm_correlation","glcm_dissimilarity","glcm_ASM"]}
    q = (g // 8).astype(np.uint8)
    P = graycomatrix(q, distances=[1],
                     angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                     levels=32, symmetric=True, normed=True)
    feats = {}
    for name in ["contrast","homogeneity","energy","correlation","dissimilarity","ASM"]:
        feats[f"glcm_{name}"] = float(graycoprops(P, name).mean())
    return feats

# ---------------- Umbralización (4 modos) ----------------
def make_seed_mask(
    bgr, roi_mask,
    th_mode="hsv_adaptive",
    # adaptativa
    blockSize=41, C=-2, adaptive_method="gaussian",
    # hsv
    hsv_S_min=35, hsv_V_min=80, hsv_V_max=235,
    # fusión
    fusion_mode="or",
    # limpieza
    min_obj=-1, min_hole=-1, closing_disk=0,  # closing_disk queda ignorado por el nuevo pipeline
    # debug
    return_debug=False
):
    """
    th_mode: 'otsu' | 'adaptive' | 'hsv_rule' | 'hsv_adaptive'
    Devuelve: mask_final (bool) y, si return_debug=True, dict con
              'mask_hsv', 'mask_adapt', 'mask_final' (uint8 0/255).
    """
    h, w = bgr.shape[:2]

    # --- Canal para adaptive: V de HSV + blur (más estable) ---
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    V_blur = cv2.GaussianBlur(V, (5,5), 0)  # anti-grano

    # --- ROI aplicado al canal base (evitar borde del domo) ---
    if roi_mask is not None:
        V_blur = V_blur.copy()
        V_blur[~roi_mask] = 255  # simular fondo claro fuera del ROI

    mask_hsv = None
    mask_adapt = None

    if th_mode == "otsu":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if roi_mask is not None:
            gray = gray.copy(); gray[~roi_mask] = 255
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = th > 0

    elif th_mode == "adaptive":
        method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if adaptive_method.lower() == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
        th_ad = cv2.adaptiveThreshold(V_blur, 255, method, cv2.THRESH_BINARY,  # no invertido
                                      blockSize=max(3, blockSize | 1), C=C)
        th_ad = cv2.bitwise_not(th_ad)  # fondo claro -> semillas blancas
        mask_adapt = th_ad
        mask = th_ad > 0

    elif th_mode == "hsv_rule":
        H, S, Vc = cv2.split(hsv)
        mask_hsv = ((S >= int(hsv_S_min)) &
                    (Vc >= int(hsv_V_min)) &
                    (Vc <= int(hsv_V_max))).astype(np.uint8) * 255
        # ROI a HSV
        if roi_mask is not None:
            mask_hsv = cv2.bitwise_and(mask_hsv, (roi_mask.astype(np.uint8)*255))
        mask = mask_hsv > 0

    else:  # hsv_adaptive
        # HSV rule
        H, S, Vc = cv2.split(hsv)
        mask_hsv = ((S >= int(hsv_S_min)) &
                    (Vc >= int(hsv_V_min)) &
                    (Vc <= int(hsv_V_max))).astype(np.uint8) * 255
        # Adaptive en V_blur
        method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if adaptive_method.lower() == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
        th_ad = cv2.adaptiveThreshold(V_blur, 255, method, cv2.THRESH_BINARY,
                                      blockSize=max(3, blockSize | 1), C=C)
        th_ad = cv2.bitwise_not(th_ad)
        mask_adapt = th_ad

        # ROI por máscara
        if roi_mask is not None:
            roi_u8 = (roi_mask.astype(np.uint8)*255)
            mask_hsv   = cv2.bitwise_and(mask_hsv, roi_u8)
            mask_adapt = cv2.bitwise_and(mask_adapt, roi_u8)

        if fusion_mode.lower() == "and":
            mask = (mask_adapt > 0) & (mask_hsv > 0)
        else:  # "or"
            mask = ((mask_adapt > 0) | (mask_hsv > 0))

    # ------------ Limpieza y pulido ------------
    # Umbrales relativos si vienen <= 0
    if (min_obj is None) or (min_obj <= 0) or (min_hole is None) or (min_hole <= 0):
        img_area = h * w
        if (min_obj is None) or (min_obj <= 0):
            min_obj = int(0.004 * img_area)     # ~0.4% del frame
        if (min_hole is None) or (min_hole <= 0):
            min_hole = int(0.0015 * img_area)   # ~0.15% del frame

    mask = mask.astype(bool)
    mask = morphology.remove_small_objects(mask, min_size=int(min_obj))
    mask = morphology.remove_small_holes(mask, area_threshold=int(min_hole))
    mask = morphology.binary_closing(mask, morphology.disk(5))
    mask = morphology.binary_opening(mask, morphology.disk(3))

    # Fill holes extra vía flood fill (OpenCV)
    m = (mask.astype(np.uint8) * 255)
    hh, ww = m.shape
    ff = m.copy()
    mask_ff = np.zeros((hh+2, ww+2), np.uint8)
    cv2.floodFill(ff, mask_ff, (0,0), 255)
    holes = cv2.bitwise_not(ff)
    m = cv2.bitwise_or(m, holes)
    mask = (m > 0)

    if roi_mask is not None:
        mask &= roi_mask

    if return_debug:
        dbg = {
            "mask_hsv":   mask_hsv if mask_hsv is not None else np.zeros((h,w), np.uint8),
            "mask_adapt": mask_adapt if mask_adapt is not None else np.zeros((h,w), np.uint8),
            "mask_final": (mask.astype(np.uint8) * 255)
        }
        return mask, dbg
    return mask

# -------------- Pipeline (en píxeles) --------------
def process_image(
        # --- Comparación automática con máscara convex hull (IoU y Dice) ---
        # Generar máscara convex hull global
        convex_ref = np.zeros_like(mask, dtype=np.uint8)
        lab_ref, n_ref = ndi.label(mask)
        for i in range(1, n_ref+1):
            seed_mask = (lab_ref == i)
            cnts, _ = cv2.findContours(seed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                hull = cv2.convexHull(cnts[0])
                cv2.drawContours(convex_ref, [hull], -1, 1, -1)
        # Calcular métricas
        mask_bin = mask.astype(bool)
        convex_bin = convex_ref.astype(bool)
        intersection = np.logical_and(mask_bin, convex_bin).sum()
        union = np.logical_or(mask_bin, convex_bin).sum()
        iou = intersection / union if union > 0 else float('nan')
        dice = (2 * intersection) / (mask_bin.sum() + convex_bin.sum()) if (mask_bin.sum() + convex_bin.sum()) > 0 else float('nan')
        print(f"[Métricas vs. Convex Hull] IoU: {iou:.4f} | Dice: {dice:.4f}")
    path,
    thickness_px=None, thickness_ratio=None,
    spot_sigma=2.0, spot_thresh=0.06,
    save_overlay=None, save_crops=False, crop_dir=None,
    bounding_rect_margin=20,
    # ROI
    roi_frac=1.00, roi_band_px=20,
    # threshold opts
    th_mode="hsv_adaptive",
    blockSize=41, C=-2, adaptive_method="gaussian",
    hsv_S_min=35, hsv_V_min=80, hsv_V_max=235,
    fusion_mode="or",
    min_obj=-1, min_hole=-1, closing_disk=0,
    # debug masks
    save_masks=False, masks_dir=None
):
    bgr = _read_image_any(path)
    h, w = bgr.shape[:2]

    # ROI circular + banda muerta
    cx, cy = w // 2, h // 2
    r = int(min(cx, cy) * float(roi_frac))
    r_core = max(0, r - int(roi_band_px))
    Y, X = np.ogrid[:h, :w]
    dist2 = (X - cx)**2 + (Y - cy)**2
    roi_mask  = dist2 <= (r * r)
    core_mask = dist2 <= (r_core * r_core)

    mask, dbg = make_seed_mask(
        bgr, roi_mask=roi_mask,
        th_mode=th_mode,
        blockSize=blockSize, C=C, adaptive_method=adaptive_method,
        hsv_S_min=hsv_S_min, hsv_V_min=hsv_V_min, hsv_V_max=hsv_V_max,
        fusion_mode=fusion_mode,
        min_obj=min_obj, min_hole=min_hole, closing_disk=closing_disk,
        return_debug=True
    )
    mask &= core_mask


    # guardar máscaras de debug y referencia convex hull
    if save_masks:
        folder = os.path.dirname(path)
        base   = os.path.splitext(os.path.basename(path))[0]
        outdir = masks_dir if masks_dir else os.path.join(folder, f"masks_{base}")
        _ensure_dir(outdir)
        cv2.imwrite(os.path.join(outdir, "01_mask_hsv.png"),   dbg["mask_hsv"])
        cv2.imwrite(os.path.join(outdir, "02_mask_adapt.png"), dbg["mask_adapt"])
        cv2.imwrite(os.path.join(outdir, "03_mask_final.png"), dbg["mask_final"])
        # Máscara de referencia convex hull (por semilla)
        convex_ref = np.zeros_like(mask, dtype=np.uint8)
        lab, n = ndi.label(mask)
        for i in range(1, n+1):
            seed_mask = (lab == i)
            cnts, _ = cv2.findContours(seed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                hull = cv2.convexHull(cnts[0])
                cv2.drawContours(convex_ref, [hull], -1, 255, -1)
        cv2.imwrite(os.path.join(outdir, "mask_convex_reference.png"), convex_ref)

    # --- Comparación automática con máscara convex hull (IoU y Dice) ---
    # Generar máscara convex hull global (binaria)
    convex_ref_bin = np.zeros_like(mask, dtype=np.uint8)
    lab_ref, n_ref = ndi.label(mask)
    for i in range(1, n_ref+1):
        seed_mask = (lab_ref == i)
        cnts, _ = cv2.findContours(seed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            hull = cv2.convexHull(cnts[0])
            cv2.drawContours(convex_ref_bin, [hull], -1, 1, -1)
    # Calcular métricas
    mask_bin = mask.astype(bool)
    convex_bin = convex_ref_bin.astype(bool)
    intersection = np.logical_and(mask_bin, convex_bin).sum()
    union = np.logical_or(mask_bin, convex_bin).sum()
    iou = intersection / union if union > 0 else float('nan')
    dice = (2 * intersection) / (mask_bin.sum() + convex_bin.sum()) if (mask_bin.sum() + convex_bin.sum()) > 0 else float('nan')
    print(f"[Métricas vs. Convex Hull] IoU: {iou:.4f} | Dice: {dice:.4f}")

    # etiquetado
    lab, n = ndi.label(mask)
    if n == 0:
        raise RuntimeError("No se detectaron semillas.")
    if n > 5:
        lab = morphology.label(mask, connectivity=2); n = lab.max()

    # color/espacios
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_lab = color.rgb2lab(img_rgb)
    L = img_lab[...,0] / 100.0
    a = img_lab[...,1]
    b = img_lab[...,2]
    gray_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_u8[~mask] = 0

    props = measure.regionprops(lab)
    rows = []
    overlay = (img_rgb * 255.0).astype(np.uint8).copy()

    if save_crops and crop_dir:
        _ensure_dir(crop_dir)

    for i, region in enumerate(props, start=1):
        seed_mask = (lab == region.label)
        area_px = region.area
        perimeter_px = _perimeter_crofton(seed_mask, directions=4) if HAS_CROFTON else region.perimeter
        maj_px = region.major_axis_length
        min_px = region.minor_axis_length

        elongation   = safe_div(maj_px, min_px)
        circularity  = safe_div(4*np.pi*area_px, (perimeter_px**2))
        solidity     = region.solidity
        convex_area  = region.convex_area
        convex_img   = morphology.convex_hull_image(seed_mask)
        convex_perim = measure.perimeter(convex_img)
        convexity    = safe_div(convex_perim, perimeter_px)
        roundness    = safe_div(4*area_px, (np.pi*(maj_px**2)))
        eccentricity = region.eccentricity
        compactness  = safe_div(area_px, convex_area)

        L_vals = L[seed_mask]; a_vals = a[seed_mask]; b_vals = b[seed_mask]
        rgb_seed = (img_rgb.copy() * 255.0).astype(np.uint8); rgb_seed[~seed_mask] = 0
        colorfulness = hasler_susstrunk_colorfulness(cv2.cvtColor(rgb_seed, cv2.COLOR_RGB2BGR))
        color_feats = {
            "L_mean": float(np.mean(L_vals)*100.0), "L_std": float(np.std(L_vals)*100.0),
            "a_mean": float(np.mean(a_vals)),        "a_std": float(np.std(a_vals)),
            "b_mean": float(np.mean(b_vals)),        "b_std": float(np.std(b_vals)),
            "colorfulness": float(colorfulness),
            "reflectance_Lstar": float(np.mean(L_vals)*100.0),
            "color_homogeneity_inv": float(1.0 / (1e-6 + np.std(a_vals) + np.std(b_vals)))
        }

        gray_seed = gray_u8.copy(); gray_seed[~seed_mask] = 0
        tex_feats = glcm_features(gray_seed)
        defect_feats = detect_surface_defects(L, seed_mask, sigma=spot_sigma, thresh=spot_thresh)

        ref_mask = reflect_mask_over_major_axis(seed_mask, region)
        symmetry_dice = dice_coeff(seed_mask, ref_mask)

        if thickness_px is None and thickness_ratio is None:
            c_px = np.nan
        else:
            c_px = (thickness_px if thickness_px is not None else thickness_ratio * min_px)
        volume_px3 = estimate_volume_ellipsoid(maj_px, min_px, c_px) if np.isfinite(c_px) else np.nan



        # Encontrar contornos antes de usar cnts
        cnts, _ = cv2.findContours(seed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        convex_area_cv = np.nan
        convex_perim_cv = np.nan
        maj_px_hull = np.nan
        min_px_hull = np.nan
        elongation_hull = np.nan
        circularity_hull = np.nan
        roundness_hull = np.nan
        eccentricity_hull = np.nan
        compactness_hull = np.nan
        solidity_hull = np.nan
        convexity_hull = np.nan
        if cnts:
            hull = cv2.convexHull(cnts[0])
            convex_area_cv = cv2.contourArea(hull)
            convex_perim_cv = cv2.arcLength(hull, True)
            # Crear máscara del hull para análisis de ejes y otras métricas
            hull_mask = np.zeros_like(seed_mask, dtype=np.uint8)
            cv2.drawContours(hull_mask, [hull], -1, 1, -1)
            props_hull = measure.regionprops(hull_mask)
            if props_hull:
                region_hull = props_hull[0]
                maj_px_hull = region_hull.major_axis_length
                min_px_hull = region_hull.minor_axis_length
                elongation_hull = safe_div(maj_px_hull, min_px_hull)
                circularity_hull = safe_div(4*np.pi*convex_area_cv, (convex_perim_cv**2))
                roundness_hull = safe_div(4*convex_area_cv, (np.pi*(maj_px_hull**2)))
                eccentricity_hull = region_hull.eccentricity
                compactness_hull = safe_div(convex_area_cv, convex_area_cv)  # siempre 1
                solidity_hull = 1.0  # por definición
                convexity_hull = 1.0  # perímetro hull / perímetro hull

        rows.append({
            "seed_id": i,
            "area_px": float(area_px), "perimeter_px": float(perimeter_px),
            "major_axis_px": float(maj_px), "minor_axis_px": float(min_px),
            "elongation_px": float(elongation), "circularity": float(circularity),
            "solidity": float(solidity), "convexity": float(convexity),
            "roundness": float(roundness), "eccentricity": float(eccentricity),
            "compactness_A_over_Aconvex": float(compactness),
            "symmetry_dice": float(symmetry_dice),
            "volume_px3": float(volume_px3),
            # Métricas convex hull
            "area_hull": float(convex_area_cv),
            "perimeter_hull": float(convex_perim_cv),
            "major_axis_hull": float(maj_px_hull),
            "minor_axis_hull": float(min_px_hull),
            "elongation_hull": float(elongation_hull),
            "circularity_hull": float(circularity_hull),
            "roundness_hull": float(roundness_hull),
            "eccentricity_hull": float(eccentricity_hull),
            "compactness_hull": float(compactness_hull),
            "solidity_hull": float(solidity_hull),
            "convexity_hull": float(convexity_hull),
            **color_feats, **tex_feats, **defect_feats
        })


        # overlay y crop
        cnts, _ = cv2.findContours(seed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0,255,0), 2)
        # Convex hull sobre el contorno principal
        if cnts:
            hull = cv2.convexHull(cnts[0])
            cv2.drawContours(overlay, [hull], -1, (255,0,255), 2)
            # Mínimo rectángulo contenedor (rotado)
            rect = cv2.minAreaRect(cnts[0])
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(overlay, [box], 0, (0,255,255), 2)
            # Bounding rect (rectángulo alineado a ejes) expandido
            x_b, y_b, w_b, h_b = cv2.boundingRect(cnts[0])
            x0 = max(x_b - bounding_rect_margin, 0)
            y0 = max(y_b - bounding_rect_margin, 0)
            x1 = min(x_b + w_b + bounding_rect_margin, overlay.shape[1])
            y1 = min(y_b + h_b + bounding_rect_margin, overlay.shape[0])
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (255,128,0), 2)
        y, x = map(int, region.centroid)
        cv2.putText(overlay, f"#{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)

        if save_crops and crop_dir and cnts:
            # Usar el bounding rect expandido igual que en el overlay
            x_b, y_b, w_b, h_b = cv2.boundingRect(cnts[0])
            x0 = max(x_b - bounding_rect_margin, 0)
            y0 = max(y_b - bounding_rect_margin, 0)
            x1 = min(x_b + w_b + bounding_rect_margin, bgr.shape[1])
            y1 = min(y_b + h_b + bounding_rect_margin, bgr.shape[0])
            crop = bgr[y0:y1, x0:x1]
            # Guardar como PNG (8 bits)
            cv2.imwrite(os.path.join(crop_dir, f"seed_{i:02d}.png"), crop)
            # Guardar como TIFF 16-bit sin compresión (o LZW si se puede)
            import imageio
            crop_16 = np.left_shift(crop.astype(np.uint16), 8)  # Escalar 8->16 bits
            tiff_path = os.path.join(crop_dir, f"seed_{i:02d}.tiff")
            try:
                imageio.imwrite(tiff_path, crop_16, format='TIFF', compression='none')
            except TypeError:
                # Si 'compression' no es soportado, guardar sin ese argumento
                imageio.imwrite(tiff_path, crop_16, format='TIFF')

    df = pd.DataFrame(rows)
    if save_overlay:
        cv2.imwrite(save_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return df

# -------------- Batch helpers --------------
def process_one_file(img_path, **kw):
    folder = os.path.dirname(img_path)
    base   = os.path.splitext(os.path.basename(img_path))[0]
    out_csv     = os.path.join(folder, f"features_{base}.csv")
    out_overlay = os.path.join(folder, f"overlay_{base}.png")
    crop_dir    = os.path.join(folder, f"crops_{base}") if kw.get("save_crops", False) else None
    if kw.get("save_masks", False) and not kw.get("masks_dir", None):
        kw["masks_dir"] = os.path.join(folder, f"masks_{base}")

    df = process_image(path=img_path, save_overlay=out_overlay, crop_dir=crop_dir, **kw)
    df.insert(0, "image_name", os.path.basename(img_path))
    df.insert(0, "folder_id", os.path.basename(folder))
    df.to_csv(out_csv, index=False)
    return df, out_csv, out_overlay

def process_dataset(dataset_dir, pattern="toma*.dng", **kw):
    subdirs = sorted([d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)])
    all_rows, processed = [], 0
    for d in subdirs:
        candidates = sorted(glob.glob(os.path.join(d, pattern)))
        candidates = [p for p in candidates if os.path.basename(p).lower().startswith("toma")]
        if not candidates:
            continue
        for p in candidates:
            try:
                df, csvp, ovp = process_one_file(p, **kw)
                all_rows.append(df.assign(_csv_path=csvp, _overlay_path=ovp))
                processed += 1
                print(f"[OK] {p}")
            except Exception as e:
                print(f"[ERROR] {p} -> {e}")
    if processed == 0:
        raise RuntimeError("No se encontraron imágenes para procesar.")
    master = pd.concat(all_rows, ignore_index=True)
    master_path = os.path.join(dataset_dir, "master_features.csv")
    master.to_csv(master_path, index=False)
    print(f"\nResumen: {processed} imágenes procesadas.")
    print(f"Master CSV: {master_path}")
    return master_path

# -------------- CLI --------------
def parse_args():
    p.add_argument("--bounding_rect_margin", type=int, default=20, help="Margen extra en píxeles para el bounding rect")
    p = argparse.ArgumentParser(description="Fenotipado de semillas (single/dataset). Unidades: píxeles.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=str)
    mode.add_argument("--dataset_dir", type=str)
    p.add_argument("--pattern", type=str, default="toma*.dng")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--thickness_px", type=float)
    g.add_argument("--thickness_ratio", type=float)

    p.add_argument("--spot_sigma", type=float, default=2.0)
    p.add_argument("--spot_thresh", type=float, default=0.06)
    p.add_argument("--roi_frac", type=float, default=1.00)
    p.add_argument("--roi_band_px", type=int, default=20)
    p.add_argument("--save_crops", action="store_true")
    p.add_argument("--save_overlay_single", type=str, default=None)

    # modos de umbral
    p.add_argument("--th_mode", type=str, default="hsv_adaptive",
                   choices=["otsu","adaptive","hsv_rule","hsv_adaptive"])
    p.add_argument("--blockSize", type=int, default=41)
    p.add_argument("--C", type=int, default=-2)
    p.add_argument("--adaptive_method", type=str, default="gaussian", choices=["gaussian","mean"])
    p.add_argument("--hsv_S_min", type=int, default=35)
    p.add_argument("--hsv_V_min", type=int, default=80)
    p.add_argument("--hsv_V_max", type=int, default=235)
    p.add_argument("--fusion_mode", type=str, default="or", choices=["or","and"])
    p.add_argument("--min_obj", type=int, default=-1)
    p.add_argument("--min_hole", type=int, default=-1)
    p.add_argument("--closing_disk", type=int, default=0)

    # debug masks
    p.add_argument("--save_masks", action="store_true")
    p.add_argument("--masks_dir", type=str, default=None)
    return p.parse_args()

def _in_ipython():
    try:
        get_ipython  # noqa: F821
        return True
    except NameError:
        return False

# -------------- Main --------------
if __name__ == "__main__":
    if _in_ipython():
        # ===== MODO SPYDER =====
        RUN_MODE = "image"     # "image" o "dataset"

        common_kw = dict(
            thickness_px=60, thickness_ratio=None,
            spot_sigma=2.0, spot_thresh=0.06,
            roi_frac=1.10, roi_band_px=20,
            th_mode="hsv_adaptive",
            blockSize=71, C=-2, adaptive_method="gaussian",
            hsv_S_min=30, hsv_V_min=90, hsv_V_max=245,
            fusion_mode="and",
            min_obj=4000, min_hole=1500, closing_disk=5,
            save_crops=True,
            save_masks=True
        )

        if RUN_MODE == "image":
            img_path = r"E:\Tesis\Dataset_Domo_Semicircular_S_Agrosavia\10060017\toma1.dng"
            df = process_image(
                path=img_path,
                save_overlay=os.path.splitext(img_path)[0] + "_overlay.png",
                crop_dir=os.path.join(os.path.dirname(img_path), "crops_toma1"),
                **common_kw
            )
            out_csv = os.path.splitext(img_path)[0] + "_features.csv"
            df.to_csv(out_csv, index=False)
            print(f"OK. Semillas detectadas: {len(df)}")
            print(f"CSV: {out_csv}")

        else:  # RUN_MODE == "dataset"
            dataset_dir = r"E:\Tesis\Dataset_Domo_Semicircular_S_Agrosavia"
            process_dataset(dataset_dir=dataset_dir, pattern="toma*.dng", **common_kw)

    else:
        # ===== MODO TERMINAL =====
        args = parse_args()
        kw = dict(
            thickness_px=args.thickness_px, thickness_ratio=args.thickness_ratio,
            spot_sigma=args.spot_sigma, spot_thresh=args.spot_thresh,
            roi_frac=args.roi_frac, roi_band_px=args.roi_band_px,
            th_mode=args.th_mode, blockSize=args.blockSize, C=args.C,
            adaptive_method=args.adaptive_method,
            hsv_S_min=args.hsv_S_min, hsv_V_min=args.hsv_V_min, hsv_V_max=args.hsv_V_max,
            fusion_mode=args.fusion_mode,
            min_obj=args.min_obj, min_hole=args.min_hole, closing_disk=args.closing_disk,
            save_crops=args.save_crops,
            save_masks=args.save_masks, masks_dir=args.masks_dir,
            bounding_rect_margin=args.bounding_rect_margin
        )
        if args.dataset_dir:
            process_dataset(dataset_dir=args.dataset_dir, pattern=args.pattern, **kw)
        else:
            df = process_image(path=args.image, save_overlay=args.save_overlay_single,
                                 crop_dir=os.path.join(os.path.dirname(args.image), "crops_single"),
                                 **kw)
            out_csv = os.path.splitext(args.image)[0] + "_features.csv"
            df.to_csv(out_csv, index=False)
            print(f"OK. Semillas detectadas: {len(df)}")
            print(f"CSV: {out_csv}")
