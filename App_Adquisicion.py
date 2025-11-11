#!/usr/bin/python3
# ------------------------------------------------------------
# IMX708 GUI + Modo OpenCV
# - Preview RGB (fluido, evita pantalla negra)
# - Manual: ExposureTime, AnalogueGain, LensPosition, WB
# - Captura: carpeta base + subcarpeta + nombre libre + Tomar RAW (DNG)
# - Modo OpenCV: trackbars (Exposure/Gain), histograma luminancia,
#                guardar RAW TIFF (sin comprimir) con tecla 's'
# - Hilos (QThread) para captura DNG y para vista OpenCV (no congela UI)
# ------------------------------------------------------------

import os
import re
import time
import numpy as np

# ---- opcionales para guardar TIFF RAW ----
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
    QDoubleSpinBox, QSpinBox, QLineEdit, QFileDialog
)
from PyQt5.QtGui import QDesktopServices

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


# ========= Worker OpenCV: trackbars + hist + guardar TIFF RAW =========
class OpenCVPreviewWorker(QObject):
    finished = pyqtSignal()
    failed   = pyqtSignal(str)

    def __init__(self, picam2, get_effective_path, get_filename, get_current_controls):
        """
        get_effective_path(): str  -> carpeta destino actual
        get_filename(): str        -> nombre base (sin extensión deseable)
        get_current_controls(): dict -> {'ExposureTime':int,'AnalogueGain':float}
        """
        super().__init__()
        self.picam2 = picam2
        self.get_effective_path = get_effective_path
        self.get_filename = get_filename
        self.get_current_controls = get_current_controls

    # --- helpers ---
    def _ensure_dir(self, path):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass

    def _build_tiff_path(self):
        base = self.get_effective_path()
        name = self.get_filename().strip() or f"raw_{int(time.time())}"
        if not name.lower().endswith(".tiff") and not name.lower().endswith(".tif"):
            name += ".tiff"
        self._ensure_dir(base)
        return os.path.join(base, name)

    def _draw_hist(self, y_img):
        # y_img: 8-bit luminance
        hist = cv2.calcHist([y_img], [0], None, [256], [0, 256]).ravel()
        hist = hist / (hist.max() + 1e-9)
        H, W = 200, 256
        canvas = np.full((H, W, 3), 255, np.uint8)
        for x, v in enumerate(hist):
            h = int(v * (H - 10))
            cv2.line(canvas, (x, H-1), (x, H-1-h), (0, 0, 0), 1)
        cv2.putText(canvas, "Hist Y", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        return canvas

    def _save_raw_tiff(self):
        # Capturamos el buffer RAW (Bayer 10/12 -> np.uint16 normalmente)
        req = None
        try:
            req = self.picam2.capture_request()
            raw = req.make_array("raw")  # ndarray (altura, ancho) sin demosaico
        finally:
            if req is not None:
                req.release()

        if raw is None:
            raise RuntimeError("No se pudo leer el buffer RAW para guardar TIFF.")

        path = self._build_tiff_path()
        if _HAS_TIFF:
            # Guardar sin comprimir, manteniendo el mosaico Bayer
            # Nota: photometric='cfa' marca el patrón CFA en el TIFF.
            tiff.imwrite(path, raw, dtype=raw.dtype, photometric='cfa')
        else:
            # Fallback: guardar .npy para no perder datos
            np.save(path.replace(".tiff", ".npy"), raw)

        return path

    def run(self):
        try:
            win = "OpenCV Live"
            win_hist = "Histograma Y"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.namedWindow(win_hist, cv2.WINDOW_NORMAL)

            # Trackbars iniciales a partir de los controles actuales
            current = self.get_current_controls()
            init_exp = int(current.get("ExposureTime", 10000))
            init_gain = float(current.get("AnalogueGain", 1.12))

            # Escalas: exposición directamente en µs (hasta 1e6); ganancia *100
            cv2.createTrackbar("Exposure (us)", win, max(10, min(1_000_000, init_exp)), 1_000_000, lambda v: None)
            cv2.createTrackbar("Gain x100",     win, int(round(init_gain * 100)),        1600,        lambda v: None)

            cv2.resizeWindow(win, 960, 540)
            cv2.resizeWindow(win_hist, 400, 260)

            while True:
                # Leer trackbars
                exp_us = cv2.getTrackbarPos("Exposure (us)", win)
                gain_x100 = cv2.getTrackbarPos("Gain x100", win)
                exp_us = max(10, int(exp_us))
                gain = max(1.12, gain_x100 / 100.0)

                # Aplicar controles (AE off)
                try:
                    controls = {"AeEnable": False, "ExposureTime": exp_us, "AnalogueGain": float(gain)}
                    if exp_us > 20000:
                        controls["FrameDurationLimits"] = (exp_us, exp_us)
                    self.picam2.set_controls(controls)
                except Exception:
                    pass

                # Capturar frame para preview/hist (RGB)
                try:
                    frame = self.picam2.capture_array()  # RGB frame
                except Exception:
                    # Pequeño retry si hubo switching
                    time.sleep(0.01)
                    continue

                # Vista y luminancia
                rgb = frame  # Picamera2 devuelve RGB por defecto
                # Y de YCrCb (8 bits)
                y = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0]
                hist_img = self._draw_hist(y)

                # Mostrar
                cv2.imshow(win, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cv2.imshow(win_hist, hist_img)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q') or k == 27:
                    break
                if k == ord('s'):
                    # Guardar RAW como TIFF sin comprimir
                    try:
                        path = self._save_raw_tiff()
                        print(f"[OpenCV] Guardado RAW TIFF: {path if _HAS_TIFF else path.replace('.tiff','.npy')}")
                    except Exception as e:
                        print(f"[OpenCV] Error guardando TIFF: {e}")

                # Si la ventana se cierra con la X
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break

            cv2.destroyAllWindows()
            self.finished.emit()

        except Exception as e:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            self.failed.emit(str(e))


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMX708 - Manual + Captura RAW (DNG)")
        self.resize(1500, 900)

        # ===== Cámara y PREVIEW (RGB) =====
        self.picam2 = Picamera2()
        self.preview_config = self.picam2.create_preview_configuration(
            main={"size": (1536, 864)},  # preview fluido y seguro
            buffer_count=3
        )
        self.picam2.configure(self.preview_config)

        # STILL config (RAW nativo 4608x2592, SRGGB10)
        self.still_config = self.picam2.create_still_configuration(
            raw={"size": (4608, 2592), "format": "SRGGB10"},
            buffer_count=1
        )

        self.qpicamera2 = QGlPicamera2(self.picam2, width=1280, height=720, keep_ar=True)

        # ===== Panel lateral =====
        self.side_panel = QFrame(); self.side_panel.setFrameShape(QFrame.StyledPanel); self.side_panel.setFixedWidth(460)
        panel_layout = QVBoxLayout(); panel_layout.setSpacing(8)

        title = QLabel("Panel de Control")
        title.setAlignment(Qt.AlignCenter); title.setStyleSheet("font-weight:600; font-size:15pt;")
        panel_layout.addWidget(title)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

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

        # ================== Pestaña: OpenCV (preview+hist) ==================
        self.page_cv = QWidget()
        fl = QFormLayout(self.page_cv)
        self.btn_cv_open = QPushButton("Abrir vista OpenCV (trackbars + histograma)")
        self.lbl_cv_hint = QLabel("Teclas: 's' guardar RAW TIFF · 'q'/ESC salir")
        self.lbl_cv_hint.setStyleSheet("color:#555;")
        fl.addRow(self.btn_cv_open)
        fl.addRow(self.lbl_cv_hint)
        self.btn_cv_open.clicked.connect(self._open_opencv_worker)

        # ================== Pestaña: Captura ==================
        self.page_capture = QWidget()
        fc = QFormLayout(self.page_capture)

        hb_base = QHBoxLayout()
        self.le_base = QLineEdit(); self.le_base.setPlaceholderText("Selecciona carpeta base…"); self.le_base.setReadOnly(True)
        self.btn_base = QPushButton("Seleccionar…")
        hb_base.addWidget(self.le_base, 1); hb_base.addWidget(self.btn_base, 0)
        fc.addRow("Carpeta base", hb_base)

        hb_sub = QHBoxLayout()
        self.le_sub = QLineEdit(); self.le_sub.setPlaceholderText("Nombre de subcarpeta (opcional)")
        self.btn_sub_create = QPushButton("Crear")
        hb_sub.addWidget(self.le_sub, 1); hb_sub.addWidget(self.btn_sub_create, 0)
        fc.addRow("Subcarpeta nueva", hb_sub)

        hb_eff = QHBoxLayout()
        self.le_effective = QLineEdit(); self.le_effective.setReadOnly(True)
        self.btn_open = QPushButton("Abrir en explorador")
        hb_eff.addWidget(self.le_effective, 1); hb_eff.addWidget(self.btn_open, 0)
        fc.addRow("Destino efectivo", hb_eff)

        self.le_filename = QLineEdit(); self.le_filename.setPlaceholderText("Nombre de archivo (sin o con .dng / .tiff)")
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
        self.toolbox.addItem(self.page_cv,      "Vista OpenCV (histograma)")
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

        # Punteros de hilos
        self._cap_thread = None
        self._cap_worker = None
        self._cv_thread = None
        self._cv_worker = None

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

    # ---------- OpenCV Worker launcher ----------
    def _open_opencv_worker(self):
        # Lanza la ventana OpenCV en hilo aparte
        try:
            if self._cv_thread is not None:
                self._set_status("Vista OpenCV ya está activa.")
                return

            # Helpers para que el worker use la ruta/nombre actuales
            def _get_effective_path():
                return self._effective_path() or os.path.expanduser("~/")

            def _get_filename():
                return self.le_filename.text().strip() or "captura_raw"

            def _get_current_controls():
                return {
                    "ExposureTime": int(self.spin_exp_us.value()),
                    "AnalogueGain": float(self.spin_gain.value())
                }

            self._cv_thread = QThread()
            self._cv_worker = OpenCVPreviewWorker(self.picam2, _get_effective_path, _get_filename, _get_current_controls)
            self._cv_worker.moveToThread(self._cv_thread)

            self._cv_thread.started.connect(self._cv_worker.run)
            self._cv_worker.finished.connect(self._on_cv_finished)
            self._cv_worker.failed.connect(self._on_cv_failed)

            self._cv_worker.finished.connect(self._cv_thread.quit)
            self._cv_worker.failed.connect(self._cv_thread.quit)
            self._cv_worker.finished.connect(self._cv_worker.deleteLater)
            self._cv_worker.failed.connect(self._cv_worker.deleteLater)
            self._cv_thread.finished.connect(self._cv_thread.deleteLater)

            self._cv_thread.start()
            self._set_status("Vista OpenCV iniciada (teclas: s=guardar TIFF, q/ESC=salir).")

        except Exception as e:
            self._set_status(f"Error iniciando vista OpenCV: {e}")
            self._cv_thread = None
            self._cv_worker = None

    def _on_cv_finished(self):
        self._set_status("Vista OpenCV cerrada.")
        self._cv_thread = None
        self._cv_worker = None

    def _on_cv_failed(self, msg: str):
        self._set_status(f"Vista OpenCV error: {msg}")
        self._cv_thread = None
        self._cv_worker = None

    # ---------- Eventos ----------
    def showEvent(self, event):
        if not self._started:
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
