# ------------------------------------------------------------
# IMX708 GUI + Histograma (snapshot) en Qt (con post-callback)
# - Preview RGB fluido con QGlPicamera2
# - Control Manual: ExposureTime, AnalogueGain, LensPosition, WB
# - Captura RAW (DNG) en hilo (no bloquea UI)
# - Histograma "snapshot" (luminancia Y) en QLabel
#   * Botón "Actualizar histograma"
#   * Frame desde copia local (post-callback del preview)
#   * Sin HighGUI / sin contención de buffers
# ------------------------------------------------------------

import os
import re
import time
import threading
import numpy as np

# (opcional) para guardar TIFF RAW en otro flujo si lo deseas
try:
    import tifffile as tiff
    _HAS_TIFF = True
except Exception:
    _HAS_TIFF = False

import cv2

from PyQt5.QtCore import Qt, QUrl, QThread, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QVBoxLayout, QLabel, QWidget, QScrollArea,
    QToolBox, QFrame, QFormLayout, QComboBox, QPushButton,
    QDoubleSpinBox, QSpinBox, QLineEdit, QFileDialog, QSizePolicy
)
from PyQt5.QtGui import QDesktopServices, QImage, QPixmap

from picamera2 import Picamera2
from picamera2.previews.qt import QGlPicamera2


# ========= Worker en hilo para captura RAW (no bloquear UI) =========
class RawCaptureWorker(QObject):
    finished = pyqtSignal(str)   # ruta guardada
    failed   = pyqtSignal(str)   # mensaje de error

    def __init__(self, picam2, still_config, preview_config, filepath):
        super().__init__()
        self.picam2 = picam2
        self.still_config = still_config
        self.preview_config = preview_config
        self.filepath = filepath

    def run(self):
        try:
            self.picam2.switch_mode_and_capture_file(
                self.still_config, self.filepath, name="raw"
            )
            self.picam2.switch_mode(self.preview_config)
            self.finished.emit(self.filepath)
        except Exception as e:
            try:
                self.picam2.switch_mode(self.preview_config)
            except Exception:
                pass
            self.failed.emit(str(e))


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMX708 - Manual + Captura RAW (DNG) + Histograma (snapshot)")
        self.resize(1500, 900)

        # ===== Cámara y PREVIEW (RGB) =====
        self.picam2 = Picamera2()
        self.preview_config = self.picam2.create_preview_configuration(
            main={"size": (1536, 864)},
            buffer_count=6
        )
        self.picam2.configure(self.preview_config)

        # STILL config (RAW nativo 4608x2592, SRGGB10)
        self.still_config = self.picam2.create_still_configuration(
            raw={"size": (4608, 2592), "format": "SRGGB10"},
            buffer_count=1
        )

        self.qpicamera2 = QGlPicamera2(self.picam2, width=1280, height=720, keep_ar=True)

        # --- buffer local del último frame (copiado desde el preview vía callback) ---
        self._latest_rgb = None
        self._frame_lock = threading.Lock()

        # ===== Panel lateral =====
        self.side_panel = QFrame()
        self.side_panel.setFrameShape(QFrame.StyledPanel)
        self.side_panel.setFixedWidth(460)
        panel_layout = QVBoxLayout(); panel_layout.setSpacing(8)

        title = QLabel("Panel de Control")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight:600; font-size:15pt;")
        panel_layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.toolbox = QToolBox()
        self.toolbox.setStyleSheet("""
            QToolBox::tab {
                background:#e8e8e8;border:1px solid #c3c3c3;border-radius:4px;
                padding:6px;font-weight:bold;
            }
            QToolBox::tab:selected { background:#d0e0ff; }
        """)

        # ================== Pestaña: Manual ==================
        self.page_manual = QWidget()
        fm = QFormLayout(self.page_manual)

        self.spin_exp_us = QSpinBox(); self.spin_exp_us.setRange(10, 10_000_000); self.spin_exp_us.setSingleStep(100); self.spin_exp_us.setValue(10000)
        fm.addRow("ExposureTime (µs)", self.spin_exp_us)

        self.spin_gain = QDoubleSpinBox(); self.spin_gain.setDecimals(3); self.spin_gain.setSingleStep(0.05); self.spin_gain.setRange(1.12, 16.0); self.spin_gain.setValue(1.12)
        fm.addRow("AnalogueGain", self.spin_gain)

        self.spin_lens = QDoubleSpinBox(); self.spin_lens.setDecimals(2); self.spin_lens.setSingleStep(0.1); self.spin_lens.setRange(1.0, 16.0); self.spin_lens.setValue(1.0)
        fm.addRow("LensPosition", self.spin_lens)

        self.cmb_wb_mode = QComboBox()
        self.cmb_wb_mode.addItems(["Auto", "Kelvin (CT)", "Preset de escena", "Manual gains"])
        fm.addRow("WB Mode", self.cmb_wb_mode)

        self.spin_ct = QSpinBox(); self.spin_ct.setRange(100, 10000); self.spin_ct.setSingleStep(50); self.spin_ct.setValue(5500)
        fm.addRow("ColourTemperature (K)", self.spin_ct)

        self.cmb_awb_preset = QComboBox()
        self.cmb_awb_preset.addItems(["Incandescent", "Tungsten", "Fluorescent", "Indoor", "Daylight", "Cloudy"])
        fm.addRow("AWB Preset", self.cmb_awb_preset)

        self.spin_awb_r = QDoubleSpinBox(); self.spin_awb_r.setDecimals(3); self.spin_awb_r.setRange(0.50, 8.00); self.spin_awb_r.setSingleStep(0.05); self.spin_awb_r.setValue(1.73)
        self.spin_awb_b = QDoubleSpinBox(); self.spin_awb_b.setDecimals(3); self.spin_awb_b.setRange(0.50, 8.00); self.spin_awb_b.setSingleStep(0.05); self.spin_awb_b.setValue(2.14)
        fm.addRow("Red Gain", self.spin_awb_r); fm.addRow("Blue Gain", self.spin_awb_b)

        self.btn_apply_manual = QPushButton("Aplicar Manual")
        fm.addRow(self.btn_apply_manual)

        self._awb_mode_map = {"Auto":0,"Incandescent":1,"Tungsten":2,"Fluorescent":3,"Indoor":4,"Daylight":5,"Cloudy":6}

        self.btn_apply_manual.clicked.connect(self._apply_all_manual)
        self.cmb_wb_mode.currentTextChanged.connect(self._wb_ui_update)

        # ================== Pestaña: Histograma (snapshot) ==================
        self.page_hist = QWidget()
        fh = QFormLayout(self.page_hist)

        self.hist_label = QLabel("(Aún no se ha generado el histograma)")
        self.hist_label.setAlignment(Qt.AlignCenter)
        self.hist_label.setStyleSheet("background:#fafafa; border:1px solid #ddd;")
        # *** tamaño fijo para evitar reescalados borrosos ***
        self.hist_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._hist_w, self._hist_h = 360, 260
        self.hist_label.setFixedSize(self._hist_w, self._hist_h)

        self.btn_hist_update = QPushButton("Actualizar histograma")
        self.lbl_hist_stats = QLabel("Estadísticos: —")
        self.lbl_hist_stats.setStyleSheet("color:#555;")

        fh.addRow(self.hist_label)
        fh.addRow(self.btn_hist_update)
        fh.addRow(self.lbl_hist_stats)

        self.btn_hist_update.clicked.connect(self._update_histogram_snapshot)

        # ================== Pestaña: Captura ==================
        self.page_capture = QWidget()
        fc = QFormLayout(self.page_capture)

        self.le_base = QLineEdit(); self.le_base.setPlaceholderText("Selecciona carpeta base…"); self.le_base.setReadOnly(True)
        self.btn_base = QPushButton("Seleccionar…")
        hb_base = QHBoxLayout(); hb_base.addWidget(self.le_base, 1); hb_base.addWidget(self.btn_base, 0)
        fc.addRow("Carpeta base", hb_base)

        self.le_sub = QLineEdit(); self.le_sub.setPlaceholderText("Nombre de subcarpeta (opcional)")
        self.btn_sub_create = QPushButton("Crear")
        hb_sub = QHBoxLayout(); hb_sub.addWidget(self.le_sub, 1); hb_sub.addWidget(self.btn_sub_create, 0)
        fc.addRow("Subcarpeta nueva", hb_sub)

        self.le_effective = QLineEdit(); self.le_effective.setReadOnly(True)
        self.btn_open = QPushButton("Abrir en explorador")
        hb_eff = QHBoxLayout(); hb_eff.addWidget(self.le_effective, 1); hb_eff.addWidget(self.btn_open, 0)
        fc.addRow("Destino efectivo", hb_eff)

        self.le_filename = QLineEdit(); self.le_filename.setPlaceholderText("Nombre de archivo (sin o con .dng)")
        fc.addRow("Nombre de archivo", self.le_filename)

        self.lbl_info_res = QLabel("Resolución RAW: 4608 × 2592 (SRGGB10)")
        self.lbl_info_res.setStyleSheet("color:#555;")
        fc.addRow(self.lbl_info_res)

        self.btn_capture = QPushButton("Tomar RAW (DNG)")
        fc.addRow(self.btn_capture)

        self.btn_base.clicked.connect(self._select_base_folder)
        self.btn_sub_create.clicked.connect(self._create_subfolder)
        self.btn_open.clicked.connect(self._open_effective_path)
        self.le_sub.textChanged.connect(self._update_effective_path)
        self.btn_capture.clicked.connect(self._capture_raw_dng)

        # Añadir páginas + barra de estado
        self.toolbox.addItem(self.page_manual,  "Control parámetros (Manual)")
        self.toolbox.addItem(self.page_hist,    "Histograma (snapshot)")
        self.toolbox.addItem(self.page_capture, "Captura (RAW DNG)")

        scroll.setWidget(self.toolbox)
        panel_layout.addWidget(scroll)

        self.status = QLabel("")
        self.status.setStyleSheet("color:#333; padding:4px; border-top:1px solid #ddd;")
        panel_layout.addWidget(self.status)

        self.side_panel.setLayout(panel_layout)

        # Layout principal
        main = QHBoxLayout()
        main.addWidget(self.qpicamera2, 4)
        main.addWidget(self.side_panel, 1)
        self.setLayout(main)

        # Estado inicial
        self._started = False
        self._wb_ui_update(self.cmb_wb_mode.currentText())
        self._soft_sync_control_limits()
        self._update_effective_path()

        # Punteros de hilo de captura (sólo para DNG)
        self._cap_thread = None
        self._cap_worker = None

    # ---------- Utilidades ----------
    def _soft_sync_control_limits(self):
        try:
            cc = self.picam2.camera_controls
            if "ExposureTime" in cc:
                mn, mx, df = cc["ExposureTime"]
                self.spin_exp_us.setRange(max(10, int(mn)), max(int(mx), 1_000_000))
                self.spin_exp_us.setValue(max(10, int(df)))
            if "AnalogueGain" in cc:
                mn, mx, df = cc["AnalogueGain"]
                self.spin_gain.setRange(max(1.12, float(mn)), min(16.0, float(mx)))
                self.spin_gain.setValue(max(1.12, min(16.0, float(df))))
            if "LensPosition" in cc:
                mn, mx, df = cc["LensPosition"]
                self.spin_lens.setRange(max(1.0, float(mn)), min(16.0, float(mx)))
                self.spin_lens.setValue(max(1.0, min(16.0, float(df))))
            if "ColourTemperature" in cc:
                mn, mx, df = cc["ColourTemperature"]
                self.spin_ct.setRange(max(100, int(mn)), min(10000, int(mx)))
                self.spin_ct.setValue(max(100, min(10000, int(df))))
        except Exception as e:
            print("Aviso camera_controls:", e)

    def _wb_ui_update(self, mode_text: str):
        self.spin_ct.setEnabled(mode_text == "Kelvin (CT)")
        self.cmb_awb_preset.setEnabled(mode_text == "Preset de escena")
        manual = (mode_text == "Manual gains")
        self.spin_awb_r.setEnabled(manual); self.spin_awb_b.setEnabled(manual)

    def _set_status(self, msg: str):
        self.status.setText(msg)
        print(msg)

    # ---------- Aplicar Manual ----------
    def _apply_all_manual(self):
        exp_us = int(self.spin_exp_us.value())
        gain   = float(self.spin_gain.value())
        lens   = float(self.spin_lens.value())
        mode   = self.cmb_wb_mode.currentText()

        controls = {
            "AeEnable": False, "AfMode": 0,
            "ExposureTime": exp_us, "AnalogueGain": gain, "LensPosition": lens
        }
        if exp_us > 20000:
            controls["FrameDurationLimits"] = (exp_us, exp_us)

        try:
            if mode == "Auto":
                controls.update({"AwbEnable": True, "AwbMode": self._awb_mode_map["Auto"]})
            elif mode == "Kelvin (CT)":
                controls.update({"AwbEnable": False, "ColourTemperature": int(self.spin_ct.value())})
            elif mode == "Preset de escena":
                preset = self.cmb_awb_preset.currentText()
                controls.update({"AwbEnable": True, "AwbMode": int(self._awb_mode_map.get(preset, 0))})
            else:
                r = float(self.spin_awb_r.value()); b = float(self.spin_awb_b.value())
                controls.update({"AwbEnable": False})
                try:
                    controls["ColourGains"] = (r, b)
                except Exception:
                    controls["AwbRedGain"] = r; controls["AwbBlueGain"] = b

            self.picam2.set_controls(controls)
            self._set_status("Parámetros manuales aplicados.")
        except Exception as e:
            self._set_status(f"Error al aplicar manual: {e}")

    # ---------- Captura: helpers UI ----------
    def _select_base_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta base")
        if path:
            self.le_base.setText(path)
            self._update_effective_path()
            self._set_status(f"Carpeta base seleccionada: {path}")

    def _create_subfolder(self):
        eff = self._effective_path()
        if not eff:
            self._set_status("Seleccione carpeta base primero.")
            return
        try:
            os.makedirs(eff, exist_ok=True)
            self._set_status(f"Carpeta creada/ok: {eff}")
            self._update_effective_path()
        except Exception as e:
            self._set_status(f"Error al crear carpeta: {e}")

    def _open_effective_path(self):
        eff = self._effective_path()
        if not eff:
            self._set_status("Ruta efectiva vacía.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(eff))

    def _effective_path(self) -> str:
        base = self.le_base.text().strip()
        sub  = self.le_sub.text().strip()
        if not base:
            return ""
        return os.path.join(base, sub) if sub else base

    def _update_effective_path(self):
        self.le_effective.setText(self._effective_path())

    # ---------- Validación de nombre ----------
    _INVALID_RE = re.compile(r'[<>:"\\|?*\n\r]')

    def _build_target_filepath(self):
        dest = self._effective_path()
        if not dest:
            raise RuntimeError("Seleccione/cree la carpeta destino.")
        name = self.le_filename.text().strip()
        if not name:
            raise RuntimeError("Ingrese un nombre de archivo.")
        if self._INVALID_RE.search(name) or "/" in name:
            raise RuntimeError("Nombre de archivo inválido.")
        if not name.lower().endswith(".dng"):
            name += ".dng"
        return os.path.join(dest, name)

    # ---------- Captura RAW en QThread (DNG) ----------
    def _capture_raw_dng(self):
        try:
            dest = self._effective_path()
            if not dest:
                raise RuntimeError("Seleccione la carpeta base y/o cree la subcarpeta.")
            os.makedirs(dest, exist_ok=True)

            filepath = self._build_target_filepath()
            if os.path.exists(filepath):
                raise RuntimeError("El archivo ya existe. Escriba otro nombre.")

            self.btn_capture.setEnabled(False)
            self._set_status("Capturando RAW (DNG)…")

            self._cap_thread = QThread()
            self._cap_worker = RawCaptureWorker(self.picam2, self.still_config, self.preview_config, filepath)
            self._cap_worker.moveToThread(self._cap_thread)

            self._cap_thread.started.connect(self._cap_worker.run)
            self._cap_worker.finished.connect(self._on_capture_ok)
            self._cap_worker.failed.connect(self._on_capture_fail)

            self._cap_worker.finished.connect(self._cap_thread.quit)
            self._cap_worker.failed.connect(self._cap_thread.quit)
            self._cap_worker.finished.connect(self._cap_worker.deleteLater)
            self._cap_worker.failed.connect(self._cap_worker.deleteLater)
            self._cap_thread.finished.connect(self._cap_thread.deleteLater)

            self._cap_thread.start()

        except Exception as e:
            self._set_status(f"Error de captura: {e}")
            self.btn_capture.setEnabled(True)

    def _on_capture_ok(self, filepath: str):
        self._set_status(f"Guardado: {filepath}")
        self.qpicamera2.update()
        self.btn_capture.setEnabled(True)

    def _on_capture_fail(self, msg: str):
        self._set_status(f"Error de captura: {msg}")
        self.qpicamera2.update()
        self.btn_capture.setEnabled(True)

    # ---------- Callback de preview: copia el frame 'main' ----------
    def _frame_callback(self, request):
        try:
            arr = request.make_array("main")
            if isinstance(arr, np.ndarray) and arr.size > 0:
                with self._frame_lock:
                    self._latest_rgb = arr.copy()
        except Exception:
            pass

    # ---------- Helpers de frame no bloqueante ----------
    def _estimate_frame_timeout_ms(self) -> int:
        try:
            exp_us = int(self.spin_exp_us.value())
            t_ms = int(max(30, min(500, 2 * (exp_us / 1000.0) + 30)))
            return t_ms
        except Exception:
            return 120

    def _grab_latest_frame(self) -> np.ndarray:
        with self._frame_lock:
            if isinstance(self._latest_rgb, np.ndarray) and self._latest_rgb.size > 0:
                return self._latest_rgb.copy()

        timeout_ms = self._estimate_frame_timeout_ms()
        attempts = 2
        for _ in range(attempts):
            req = None
            try:
                req = self.picam2.capture_request(timeout=timeout_ms)
                if req is None:
                    continue
                try:
                    arr = req.make_array("main")
                except Exception:
                    arr = req.make_array()
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    return arr
            except Exception:
                pass
            finally:
                try:
                    if req is not None:
                        req.release()
                except Exception:
                    pass
        return None

    # ---------- Histograma (snapshot) ----------
    def _update_histogram_snapshot(self):
        frame = self._grab_latest_frame()
        if frame is None:
            self._set_status("No hay frame disponible todavía. Intenta de nuevo.")
            return
        try:
            y = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)[:, :, 0]
            y_min = int(np.min(y)); y_max = int(np.max(y))
            y_mean = float(np.mean(y)); y_std = float(np.std(y))

            # usar tamaño fijo del QLabel (sin reescalar)
            hist_img_bgr = self._render_hist_image(y, width=self._hist_w, height=self._hist_h)
            pix = self._np_bgr_to_qpixmap(hist_img_bgr)
            self.hist_label.setPixmap(pix)

            self.lbl_hist_stats.setText(
                f"Estadísticos: min={y_min}  max={y_max}  media={y_mean:.1f}  std={y_std:.1f}"
            )
            self._set_status("Histograma actualizado.")
        except Exception as e:
            self._set_status(f"Error calculando histograma: {e}")

    # ---------- Render del histograma con ejes/etiquetas (sin solapes) ----------
    def _render_hist_image(self, y_img: np.ndarray, width=360, height=260) -> np.ndarray:
        """
        Histograma (256 bins) normalizado con ejes y etiquetas.
        X: Intensidad (0-255). Y: Frecuencia relativa (0.0-1.0).
        Dibuja 'Frecuencia relativa' en vertical para evitar solapes con el título.
        """
        hist = cv2.calcHist([y_img], [0], None, [256], [0, 256]).ravel()
        hist = hist / np.max(hist) if np.max(hist) > 0 else hist

        H, W = height, width
        canvas = np.full((H, W, 3), 255, np.uint8)

        # Márgenes amplios
        left_margin, right_margin = 50, 28
        top_margin, bottom_margin = 52, 48
        usable_w = max(1, W - left_margin - right_margin)
        usable_h = max(1, H - top_margin - bottom_margin)

        # Ejes
        x0, y0 = left_margin, H - bottom_margin
        x1, y1 = W - right_margin, top_margin
        cv2.line(canvas, (x0, y0), (x1, y0), (0, 0, 0), 1)  # X
        cv2.line(canvas, (x0, y0), (x0, y1), (0, 0, 0), 1)  # Y

        # Curva
        xs = np.linspace(x0, x1 - 1, 256).astype(int)
        ys = (y1 + usable_h - (hist * usable_h)).astype(int)
        for i in range(1, 256):
            cv2.line(canvas, (xs[i-1], ys[i-1]), (xs[i], ys[i]), (0, 0, 0), 2)

        # Ticks Y (0.0, 0.5, 1.0)
        for val in [0.0, 0.5, 1.0]:
            y_pos = int(y1 + usable_h - (val * usable_h))
            cv2.line(canvas, (x0 - 5, y_pos), (x0, y_pos), (0, 0, 0), 1)
            cv2.putText(canvas, f"{val:.1f}", (8, y_pos + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA)

        # Ticks X (0, 128, 255)
        for val, label in [(0, "0"), (128, "128"), (255, "255")]:
            x_pos = int(x0 + (val / 255.0) * usable_w)
            cv2.line(canvas, (x_pos, y0), (x_pos, y0 + 5), (0, 0, 0), 1)
            cv2.putText(canvas, label, (x_pos - 10, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA)

        # Etiqueta X (centrada y baja para no pisar ticks)
        cv2.putText(canvas, "Intensidad (0-255)",
                    (x0 + (usable_w // 2) - 70, H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

        # Etiqueta Y vertical
        label = "Frecuencia relativa"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        txt = np.full((th + 6, tw + 6, 3), 255, np.uint8)
        cv2.putText(txt, label, (3, th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        txt = cv2.rotate(txt, cv2.ROTATE_90_COUNTERCLOCKWISE)
        thv, twv = txt.shape[:2]
        y_top = y1 + (usable_h - thv) // 2
        x_left = max(4, x0 - 35)
        canvas[y_top:y_top + thv, x_left:x_left + twv] = txt

        # Título centrado
        title = "Histograma Y (snapshot)"
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)
        cv2.putText(canvas, title, ((W - tw) // 2, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)

        return canvas

    def _np_bgr_to_qpixmap(self, img_bgr: np.ndarray) -> QPixmap:
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("Se esperaba imagen BGR de 3 canales.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    # ---------- Eventos ----------
    def showEvent(self, event):
        if not self._started:
            try:
                self.picam2.post_callback = self._frame_callback
            except Exception:
                try:
                    self.picam2.set_post_callback(self._frame_callback)
                except Exception:
                    pass

            self.picam2.start()
            self.qpicamera2.update()
            try:
                self._apply_all_manual()
            except Exception as e:
                self._set_status(f"Init manual error: {e}")
            self._started = True
        super().showEvent(event)

    def closeEvent(self, event):
        try:
            self.picam2.stop()
        except Exception:
            pass
        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    w = PreviewWindow()
    w.show()
    app.exec_()