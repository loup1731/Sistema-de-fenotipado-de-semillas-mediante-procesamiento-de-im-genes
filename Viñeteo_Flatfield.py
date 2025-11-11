#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vineteo_flatfield.py
Analiza viñeteo desde un DNG lineal (RAW), genera:
 - Visualización en dB del viñeteo
 - Perfil radial (CSV + PNG)
 - Mapa de corrección (TIFF + NPY)
 - Vista previa corregida (PNG)

Uso:
  python vineteo_flatfield.py --dng tu_foto.DNG [--bins 60] [--blur-frac 0.10] [--gamma 1.0]
"""

import argparse, os, csv
import numpy as np
import rawpy
import imageio.v3 as iio
import cv2
import matplotlib.pyplot as plt

# ---------------------- Utils ----------------------
def robust_percentile(x, p):
    x = x[np.isfinite(x)]
    return np.percentile(x, p) if x.size else 0.0

def stretch_percentiles(img, p_lo=1.0, p_hi=99.0):
    mask = np.isfinite(img)
    lo, hi = np.percentile(img[mask], [p_lo, p_hi]) if np.any(mask) else (0,1)
    out = (img - lo) / (hi - lo + 1e-12)
    return np.clip(out, 0, 1)

def ensure_odd(n):
    n = int(n)
    return n if (n % 2 == 1) else (n+1)

# ---------------------- RAW load (Green plane) ----------------------

def load_raw_green(dng_path):
    """
    Devuelve:
      G_full: plano verde continuo (float32, misma resolución que raw visible)
      meta  : dict con wl, bl
    """
    with rawpy.imread(dng_path) as raw:
        # niveles de negro y blanco robustos a variaciones
        bl_pc = getattr(raw, "black_level_per_channel", None)
        if bl_pc is not None and np.size(bl_pc) > 0:
            bl = float(np.median(bl_pc))
        else:
            bl = float(getattr(raw, "black_level", 0.0))

        wl = getattr(raw, "white_level", None)
        if wl is None:
            # fallback (no debería ser necesario con DNG correcto)
            wl = float(np.percentile(raw.raw_image_visible, 99.9))

        Iraw = raw.raw_image_visible.astype(np.float32)
        Ilin = np.clip(Iraw - bl, 0.0, wl - bl)

        H, W = Ilin.shape

        # Extraer dos verdes del mosaico (G1/G2) sin depender del patrón exacto.
        # Usamos posiciones [0,1] y [1,0] como aproximación típica.
        G1 = Ilin[0:H:2, 1:W:2]
        G2 = Ilin[1:H:2, 0:W:2]
        h = min(G1.shape[0], G2.shape[0])
        w = min(G1.shape[1], G2.shape[1])
        G = 0.5 * (G1[:h, :w] + G2[:h, :w])

        # Subir G al tamaño completo para mapear por píxel (replicamos en los dos sitios verdes y suavizamos)
        G_full = np.zeros_like(Ilin, dtype=np.float32)
        G_full[0:h*2:2, 1:w*2:2] = G
        G_full[1:h*2:2, 0:w*2:2] = G
        # Suavizado suave para rellenar huecos
        G_full = cv2.GaussianBlur(G_full, (0, 0), 0.6)

        meta = {"wl": float(wl), "bl": float(bl)}
        return G_full, meta

# ---------------------- Illumination estimation ----------------------
def estimate_illumination(G, blur_frac=0.10, sat_clip=0.98):
    H, W = G.shape
    max_lin = np.nanmax(G) if np.isfinite(G).any() else 1.0
    sat_thr = sat_clip * max_lin
    valid = (G < sat_thr) & np.isfinite(G)

    k = ensure_odd(max(31, int(min(H, W) * blur_frac)))
    L = cv2.GaussianBlur(G, (k, k), 0)
    L = np.where(L <= 1e-6, 1e-6, L)  # evitar división por cero
    return L, valid

# ---------------------- Relative gain & radial profile ----------------------
def relative_gain(G, L, valid):
    R = np.full_like(G, np.nan, dtype=np.float32)
    R[valid] = G[valid] / L[valid]
    med = np.median(R[valid]) if np.any(valid) else 1.0
    if med <= 0: med = 1.0
    R[valid] = R[valid] / med
    return R

def radial_profile(R, valid, bins=60):
    H, W = R.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.indices(R.shape, dtype=np.float32)
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    r_max = min(cx, cy, W-1-cx, H-1-cy)
    r_norm = r / (r_max + 1e-12)

    edges = np.linspace(0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    gain = np.zeros(bins, dtype=np.float32)
    gain_db = np.zeros(bins, dtype=np.float32)

    for i in range(bins):
        m = (r_norm >= edges[i]) & (r_norm < edges[i+1]) & valid & np.isfinite(R)
        if np.any(m):
            g = np.median(R[m])
            gain[i] = g
            gain_db[i] = 20 * np.log10(max(g, 1e-6))
        else:
            gain[i] = np.nan
            gain_db[i] = np.nan

    return centers, gain, gain_db

# ---------------------- Save helpers ----------------------
def save_profile_csv(path_csv, r, gain, gain_db):
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["radius_norm", "relative_gain", "relative_gain_dB"])
        for ri, gi, di in zip(r, gain, gain_db):
            w.writerow([
                f"{ri:.6f}",
                f"{gi:.6f}" if np.isfinite(gi) else "",
                f"{di:.6f}" if np.isfinite(di) else ""
            ])

def plot_profile_png(path_png, r, gain_db):
    plt.figure(figsize=(6, 4))
    plt.plot(r, gain_db, linewidth=2)
    plt.xlabel("Radio normalizado (0 = centro, 1 = borde)")
    plt.ylabel("Ganancia relativa [dB]")
    plt.title("Perfil radial de viñeteo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()

def build_correction_map(R, valid, smooth_frac=0.06, clip_range=(0.5, 2.0)):
    C = np.ones_like(R, dtype=np.float32)
    C[valid] = 1.0 / np.clip(R[valid], 1e-3, 1e3)
    H, W = C.shape
    k = ensure_odd(max(15, int(min(H, W) * smooth_frac)))
    C = cv2.GaussianBlur(C, (k, k), 0)
    lo, hi = clip_range
    C = np.clip(C, lo, hi)
    return C

def preview_corrected(G, C, gamma=1.0):
    Gc = G * C
    vis = stretch_percentiles(Gc, 0.5, 99.5)
    if gamma != 1.0:
        vis = np.power(vis, 1.0 / gamma)
    return (vis * 255).astype(np.uint8)

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dng", required=False, help="Ruta al archivo .DNG")
    ap.add_argument("--bins", type=int, default=60, help="Número de bins del perfil radial")
    ap.add_argument("--blur-frac", type=float, default=0.10, help="Fracción del tamaño para blur de iluminación")
    ap.add_argument("--gamma", type=float, default=1.0, help="Gamma vista previa corregida (>=1 oscurece)")
    args = ap.parse_args()

    dng_path = args.dng or r"E:/Domos_viñetados/Domo_Semicircular/viñetado/viñetado3.dng"
    dng_path = os.path.abspath(dng_path)
    outbase = os.path.splitext(dng_path)[0]

    print("[1/6] Cargando RAW y extrayendo plano verde...")
    G, meta = load_raw_green(dng_path)

    print("[2/6] Estimando iluminación (baja frecuencia)...")
    L, valid = estimate_illumination(G, blur_frac=args.blur_frac)

    print("[3/6] Ganancia relativa R=G/L (normalizada a mediana)...")
    R = relative_gain(G, L, valid)

    print("[4/6] Visualización del viñeteo en dB...")
    R_db = np.full_like(R, np.nan, dtype=np.float32)
    m = valid & np.isfinite(R)
    R_db[m] = 20*np.log10(np.clip(R[m], 1e-6, None))
    
    # Estirar usando SOLO válidos
    if np.any(m):
        lo, hi = np.percentile(R_db[m], [1, 99])
        denom = (hi - lo) if (hi > lo) else 1.0
        vis = (R_db - lo) / (denom + 1e-12)
    else:
        vis = np.zeros_like(R_db, dtype=np.float32)
    
    # Saneamiento para escritura
    vis = np.clip(vis, 0, 1)
    vis[~np.isfinite(vis)] = 0.0     # <- elimina NaN/Inf fuera de máscara
    iio.imwrite(f"{outbase}_vineteo_vis_db.png", (vis * 255).astype(np.uint8))


    print("[5/6] Perfil radial...")
    r, gain, gain_db = radial_profile(R, valid, bins=args.bins)
    save_profile_csv(f"{outbase}_profile_radial.csv", r, gain, gain_db)
    plot_profile_png(f"{outbase}_profile_radial.png", r, gain_db)

    H, W = R.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.indices(R.shape, dtype=np.float32)
    rn = np.sqrt((yy - cy)**2 + (xx - cx)**2) / (min(cx, cy, W-1-cx, H-1-cy) + 1e-12)

    center_m = (rn <= 0.15) & m
    edge_m   = (rn >= 0.85) & (rn <= 1.00) & m
    center_db = np.median(20*np.log10(np.clip(R[center_m], 1e-6, None))) if np.any(center_m) else np.nan
    edge_db   = np.median(20*np.log10(np.clip(R[edge_m], 1e-6, None)))   if np.any(edge_m)   else np.nan
    falloff_db = (edge_db - center_db) if (np.isfinite(center_db) and np.isfinite(edge_db)) else np.nan

    print(f"   Centro (r<=0.15) [dB]: {center_db:.3f}")
    print(f"   Borde  (0.85<=r<=1.00) [dB]: {edge_db:.3f}")
    print(f"   Caída centro→borde [dB]: {falloff_db:.3f}")

    print("[6/6] Mapa de corrección y vista previa...")
    C = build_correction_map(R, valid)
    np.save(f"{outbase}_correction_map.npy", C.astype(np.float32))
    iio.imwrite(f"{outbase}_correction_map.tiff", C.astype(np.float32))

    prev = preview_corrected(G, C, gamma=args.gamma)
    iio.imwrite(f"{outbase}_preview_corrected.png", prev)

    print("\nListo. Archivos generados en la misma carpeta del DNG:")
    print(f" - {outbase}_vineteo_vis_db.png")
    print(f" - {outbase}_profile_radial.csv")
    print(f" - {outbase}_profile_radial.png")
    print(f" - {outbase}_correction_map.npy")
    print(f" - {outbase}_correction_map.tiff")
    print(f" - {outbase}_preview_corrected.png")


if __name__ == "__main__":
    main()
