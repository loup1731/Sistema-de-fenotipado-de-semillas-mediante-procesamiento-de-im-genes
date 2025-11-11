"""
Fenotipado de semillas (1–5 semillas, fondo blanco, semillas oscuras)
+ MODO BATCH por dataset:
  Recorre subcarpetas (100600xx), ignora 'info.*', procesa 'toma*.dng'
  y genera CSV/overlay por imagen + un master CSV en la raíz.
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
            raise RuntimeError("Archivo RAW detectado. Instala:  python -m pip install rawpy imageio")
        with rawpy.imread(path) as raw:
            rgb16 = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16)
        rgb8 = (rgb16 >> 8).astype(np.uint8)
        return cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return bgr

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def _normalize_output_path(path, default_ext):
    if not path: return None
    root, ext = os.path.splitext(path)
    if ext == "": path = path + default_ext
    _ensure_dir(os.path.dirname(path))
    return path

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
                   [ np.sin(-theta), np.cos(-theta),0],
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

# -------------- Pipeline (en píxeles) --------------
def process_image(path, thickness_px=None, thickness_ratio=None,
                  spot_sigma=2.0, spot_thresh=0.06, save_overlay=None,
                  save_crops=False, crop_dir=None,
                  roi_frac=0.78, roi_band_px=30):
    bgr = _read_image_any(path)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ROI circular + banda muerta
    cx, cy = w // 2, h // 2
    r = int(min(cx, cy) * roi_frac)
    r_core = max(0, r - int(roi_band_px))
    Y, X = np.ogrid[:h, :w]
    dist2 = (X - cx)**2 + (Y - cy)**2
    roi_mask  = dist2 <= (r * r)
    core_mask = dist2 <= (r_core * r_core)

    gray_blur = cv2.GaussianBlur(gray, (0,0), 1.0)
    gray_blur[~roi_mask] = 255
    _, th = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = (th > 0) & roi_mask

    # Limpieza
    mask = morphology.remove_small_objects(mask, min_size=350)
    mask = morphology.remove_small_holes(mask, area_threshold=350)
    mask = morphology.binary_closing(mask, morphology.disk(2))
    mask = mask & core_mask

    lab, n = ndi.label(mask)
    if n == 0:
        raise RuntimeError("No se detectaron semillas.")
    if n > 5:
        lab = morphology.label(mask, connectivity=2); n = lab.max()

    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_lab = color.rgb2lab(img_rgb)
    L = img_lab[...,0] / 100.0
    a = img_lab[...,1]
    b = img_lab[...,2]
    gray_u8 = gray.copy(); gray_u8[~mask] = 0

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
            **color_feats, **tex_feats, **defect_feats
        })

        # overlay y crop
        cnts, _ = cv2.findContours(seed_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (0,255,0), 2)
        y, x = map(int, region.centroid)
        cv2.putText(overlay, f"#{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)

        if save_crops and crop_dir:
            ys, xs = np.where(seed_mask)
            y0, y1 = max(ys.min()-5,0), min(ys.max()+6,h)
            x0, x1 = max(xs.min()-5,0), min(xs.max()+6,w)
            crop = bgr[y0:y1, x0:x1]
            cv2.imwrite(os.path.join(crop_dir, f"seed_{i:02d}.png"), crop)

    df = pd.DataFrame(rows)
    if save_overlay:
        cv2.imwrite(save_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return df

# -------------- Batch helpers --------------
def process_one_file(img_path, args):
    folder = os.path.dirname(img_path)
    base   = os.path.splitext(os.path.basename(img_path))[0]  # p.ej. toma1
    # rutas de salida dentro de la misma carpeta
    out_csv     = os.path.join(folder, f"features_{base}.csv")
    out_overlay = os.path.join(folder, f"overlay_{base}.png")
    crop_dir    = os.path.join(folder, f"crops_{base}")

    df = process_image(
        path=img_path,
        thickness_px=args.thickness_px,
        thickness_ratio=args.thickness_ratio,
        spot_sigma=args.spot_sigma,
        spot_thresh=args.spot_thresh,
        save_overlay=out_overlay,
        save_crops=args.save_crops,
        crop_dir=(crop_dir if args.save_crops else None),
        roi_frac=args.roi_frac,
        roi_band_px=args.roi_band_px
    )
    # info para master
    df.insert(0, "image_name", os.path.basename(img_path))
    df.insert(0, "folder_id", os.path.basename(folder))
    df.to_csv(out_csv, index=False)
    return df, out_csv, out_overlay

def process_dataset(root_dir, pattern):
    subdirs = sorted([d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)])
    all_rows = []
    processed = 0
    for d in subdirs:
        # ignorar carpetas que no contengan tomas
        candidates = sorted(glob.glob(os.path.join(d, pattern)))
        # descartar 'info.*'
        candidates = [p for p in candidates if os.path.basename(p).lower().startswith("toma")]
        if not candidates:
            continue
        for p in candidates:
            try:
                df, csvp, ovp = process_one_file(p, args)  # usa args del main
                all_rows.append(df.assign(_csv_path=csvp, _overlay_path=ovp))
                processed += 1
                print(f"[OK] {p}")
            except Exception as e:
                print(f"[ERROR] {p} -> {e}")
    if processed == 0:
        raise RuntimeError("No se encontraron imágenes para procesar.")
    master = pd.concat(all_rows, ignore_index=True)
    master_path = os.path.join(root_dir, "master_features.csv")
    master.to_csv(master_path, index=False)
    print(f"\nResumen: {processed} imágenes procesadas.")
    print(f"Master CSV: {master_path}")
    return master_path

# -------------- CLI --------------
def parse_args():
    p = argparse.ArgumentParser(description="Fenotipado de semillas (modo single o dataset). Unidades: píxeles.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=str, help="Ruta a una imagen (DNG/JPG/PNG)")
    mode.add_argument("--dataset_dir", type=str, help="Carpeta raíz con subcarpetas 100600xx")

    p.add_argument("--pattern", type=str, default="*.dng", help="Patrón de archivos a procesar dentro de cada carpeta (ej. 'toma*.dng')")
    # Espesor en píxeles (opcional) — seguimos en px
    g = p.add_mutually_exclusive_group()
    g.add_argument("--thickness_px", type=float, help="Espesor estimado en píxeles")
    g.add_argument("--thickness_ratio", type=float, help="Espesor como fracción del eje menor (ej. 0.35)")
    p.add_argument("--spot_sigma", type=float, default=2.0)
    p.add_argument("--spot_thresh", type=float, default=0.06)
    p.add_argument("--roi_frac", type=float, default=0.78)
    p.add_argument("--roi_band_px", type=int, default=30)
    p.add_argument("--save_crops", action="store_true", help="Guardar crops por semilla en cada imagen")
    p.add_argument("--save_overlay_single", type=str, default=None, help="Solo en modo --image: ruta del overlay de salida")
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
        # ---- MODO SPYDER: elige "image" o "dataset" ----
        RUN_MODE = "image"   # "image" o "dataset"

        if RUN_MODE == "image":
            img_path = r"E:\Tesis\Dataset_Domo_Semicircular_S_Agrosavia\10060021\toma2.dng"

            df = process_image(
                path=img_path,
                thickness_px=60,          
                spot_sigma=2.0,
                spot_thresh=0.06,
                roi_frac=1.00,
                roi_band_px=20,
                save_overlay=os.path.splitext(img_path)[0] + "_overlay.png",
                save_crops=True,
                crop_dir=os.path.join(os.path.dirname(img_path), "crops_toma2"),
            )

            out_csv = os.path.splitext(img_path)[0] + "_features.csv"
            df.to_csv(out_csv, index=False)
            print(f"OK. Semillas detectadas: {len(df)}")
            print(f"CSV: {out_csv}")

        elif RUN_MODE == "dataset":
            dataset_dir = r"E:\Tesis\Dataset_Domo_Semicircular_S_Agrosavia"
            pattern     = "toma*.dng"

            # Ajustes globales del batch
            thickness_px = 60
            spot_sigma   = 2.0
            spot_thresh  = 0.06
            roi_frac     = 1.00
            roi_band_px  = 20
            save_crops   = True

            # Recorre carpetas y procesa
            process_dataset(
                dataset_dir=dataset_dir,
                pattern=pattern,
                thickness_px=thickness_px,
                thickness_ratio=None,
                spot_sigma=spot_sigma,
                spot_thresh=spot_thresh,
                roi_frac=roi_frac,
                roi_band_px=roi_band_px,
                save_crops=save_crops,
            )
    else:
        # ---- MODO TERMINAL: usa argparse normalmente ----
        args = parse_args()
        if getattr(args, "dataset_dir", None):
            process_dataset(
                dataset_dir=args.dataset_dir,
                pattern=args.pattern,
                thickness_px=args.thickness_px,
                thickness_ratio=args.thickness_ratio,
                spot_sigma=args.spot_sigma,
                spot_thresh=args.spot_thresh,
                roi_frac=args.roi_frac,
                roi_band_px=args.roi_band_px,
                save_crops=args.save_crops,
                save_overlay_single=args.save_overlay_single,
            )
        else:
            df = process_image(
                path=args.image,
                thickness_px=args.thickness_px,
                thickness_ratio=args.thickness_ratio,
                spot_sigma=args.spot_sigma,
                spot_thresh=args.spot_thresh,
                roi_frac=args.roi_frac,
                roi_band_px=args.roi_band_px,
                save_overlay=args.save_overlay_single,
                save_crops=args.save_crops,
                crop_dir=os.path.join(os.path.dirname(args.image), "crops_single"),
            )
            out_csv = os.path.splitext(args.image)[0] + "_features.csv"
            df.to_csv(out_csv, index=False)
            print(f"OK. Semillas detectadas: {len(df)}")
            print(f"CSV: {out_csv}")