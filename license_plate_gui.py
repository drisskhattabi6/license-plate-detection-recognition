"""
License Plate Detection and Recognition GUI

Features
--------
- Auto-load YOLO detector from models/Detect_Plate_Model.pt if available
- Load local PaddleOCR model from models/latin_PP-OCRv5_mobile_rec
- Process one image, many images, or one video
- Detect multiple license plates
- Recognize plate text using PaddleOCR
- Save annotated outputs to output/
- Save CSV reports
- Stabilize video OCR with IoU-based temporal voting
- Large preview area
- Previous / Next navigation for multiple processed images

Run
---
python license_plate_gui.py
"""

from __future__ import annotations

import os
import csv
import queue
import threading
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

# Reduce noisy native logs before imports
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("FLAGS_logtostderr", "0")

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# IMPORTANT ON WINDOWS:
# Import torch BEFORE paddle/paddleocr
import torch
from ultralytics import YOLO
import paddle
from paddleocr import TextRecognition


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
DETECTOR_MODEL_PATH = MODELS_DIR / "Detect_Plate_Model.pt"
OCR_MODEL_DIR = MODELS_DIR / "latin_PP-OCRv5_mobile_rec"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PLATE_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

IMAGE_FILETYPES = [
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.JPG *.JPEG *.PNG *.BMP *.WEBP"),
    ("All files", "*.*"),
]

VIDEO_FILETYPES = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.mpeg *.mpg *.wmv *.webm"),
    ("All files", "*.*"),
]

COLORS = {
    "bg": "#0B1220",
    "panel": "#111827",
    "panel_alt": "#1F2937",
    "text": "#E5E7EB",
    "muted": "#94A3B8",
    "accent": "#22C55E",
    "accent_dark": "#16A34A",
    "blue": "#0EA5E9",
    "blue_dark": "#0284C7",
    "orange": "#F59E0B",
    "orange_dark": "#D97706",
    "red": "#EF4444",
    "red_dark": "#DC2626",
    "border": "#334155",
    "preview_bg": "#050A14",
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def normalize_plate(text: Any) -> str:
    if text is None:
        return ""
    return "".join(ch for ch in str(text).upper() if ch in PLATE_CHARACTERS)


def crop_with_padding(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    pad_ratio: float = 0.08,
) -> np.ndarray | None:
    height, width = image.shape[:2]

    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)

    pad_x = int(box_width * pad_ratio)
    pad_y = int(box_height * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def resize_keep_aspect(
    image: np.ndarray,
    target_height: int = 64,
    min_width: int = 192,
) -> np.ndarray | None:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return None

    scale = target_height / height
    new_width = max(int(width * scale), min_width)
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA

    return cv2.resize(image, (new_width, target_height), interpolation=interpolation)


def preprocess_plate(image_bgr: np.ndarray) -> np.ndarray | None:
    if image_bgr is None or image_bgr.size == 0:
        return None

    resized = resize_keep_aspect(image_bgr)
    if resized is None:
        return None

    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    enhanced = cv2.merge([l_channel, a_channel, b_channel])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)

    return sharpened


def calculate_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def best_track_text(track: dict[str, Any]) -> tuple[str, float]:
    votes = track["text_votes"]
    if not votes:
        return "", 0.0

    text = max(votes, key=votes.get)
    total_weight = sum(votes.values())
    stability = votes[text] / total_weight if total_weight > 0 else 0.0
    return text, stability


# -----------------------------------------------------------------------------
# Main GUI
# -----------------------------------------------------------------------------
class LicensePlateApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("License Plate Detection and Recognition")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 720)
        self.root.configure(bg=COLORS["bg"])

        self.detector: YOLO | None = None
        self.ocr_model: TextRecognition | None = None
        self.model_path: str = ""
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.events: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.busy = False
        self.preview_photo: ImageTk.PhotoImage | None = None

        self.conf_var = tk.StringVar(value="0.40")
        self.ocr_interval_var = tk.StringVar(value="5")
        self.pad_var = tk.StringVar(value="0.08")
        self.status_var = tk.StringVar(value="Starting...")
        self.model_var = tk.StringVar(value="Detector: not loaded")
        self.output_var = tk.StringVar(value=f"Output: {self.output_dir}")
        self.preview_title_var = tk.StringVar(value="No preview yet")
        self.preview_counter_var = tk.StringVar(value="")

        self.preview_items: list[tuple[str, np.ndarray]] = []
        self.preview_index = -1

        self._configure_theme()
        self._build_ui()

        self.root.after(100, self._process_events)
        self.root.after(150, self._startup_load_local_models)

    # -------------------------------------------------------------------------
    # Theme
    # -------------------------------------------------------------------------
    def _configure_theme(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=COLORS["bg"], foreground=COLORS["text"])
        style.configure("TFrame", background=COLORS["bg"])
        style.configure(
            "TLabelframe",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            bordercolor=COLORS["border"],
            relief="flat",
        )
        style.configure(
            "TLabelframe.Label",
            background=COLORS["panel"],
            foreground=COLORS["text"],
        )
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
        style.configure("Muted.TLabel", background=COLORS["bg"], foreground=COLORS["muted"])
        style.configure(
            "TEntry",
            fieldbackground=COLORS["panel_alt"],
            foreground=COLORS["text"],
            insertcolor=COLORS["text"],
            bordercolor=COLORS["border"],
            padding=6,
        )
        style.configure(
            "Accent.TButton",
            background=COLORS["accent"],
            foreground="#FFFFFF",
            borderwidth=0,
            padding=8,
        )
        style.map("Accent.TButton", background=[("active", COLORS["accent_dark"])])

        style.configure(
            "Blue.TButton",
            background=COLORS["blue"],
            foreground="#FFFFFF",
            borderwidth=0,
            padding=8,
        )
        style.map("Blue.TButton", background=[("active", COLORS["blue_dark"])])

        style.configure(
            "Warn.TButton",
            background=COLORS["orange"],
            foreground="#111827",
            borderwidth=0,
            padding=8,
        )
        style.map("Warn.TButton", background=[("active", COLORS["orange_dark"])])

        style.configure(
            "Danger.TButton",
            background=COLORS["red"],
            foreground="#FFFFFF",
            borderwidth=0,
            padding=8,
        )
        style.map("Danger.TButton", background=[("active", COLORS["red_dark"])])

        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=COLORS["panel_alt"],
            background=COLORS["accent"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["accent"],
            darkcolor=COLORS["accent"],
        )

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Header
        header = ttk.Frame(self.root, padding=(10, 8, 10, 2))
        header.pack(fill=tk.X)

        ttk.Label(
            header,
            text="License Plate Detection and Recognition",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            header,
            text="YOLO Detector + PaddleOCR • Images • Multiple Images • Videos",
            style="Muted.TLabel",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(1, 4))

        # Toolbar
        toolbar = ttk.Frame(self.root, padding=(10, 2, 10, 4))
        toolbar.pack(fill=tk.X)

        self.load_model_button = ttk.Button(
            toolbar,
            text="Load Detector Model",
            style="Blue.TButton",
            command=self.choose_model,
        )
        self.load_model_button.grid(row=0, column=0, padx=(0, 6), pady=2, sticky="w")

        ttk.Button(
            toolbar,
            text="Select Image(s)",
            style="Accent.TButton",
            command=self.choose_images,
        ).grid(row=0, column=1, padx=6, pady=2, sticky="w")

        ttk.Button(
            toolbar,
            text="Select Video",
            style="Accent.TButton",
            command=self.choose_video,
        ).grid(row=0, column=2, padx=6, pady=2, sticky="w")

        ttk.Button(
            toolbar,
            text="Output Folder",
            style="Blue.TButton",
            command=self.choose_output_folder,
        ).grid(row=0, column=3, padx=6, pady=2, sticky="w")

        ttk.Button(
            toolbar,
            text="Open Output",
            style="Blue.TButton",
            command=self.open_output_folder,
        ).grid(row=0, column=4, padx=6, pady=2, sticky="w")

        ttk.Button(
            toolbar,
            text="Stop",
            style="Danger.TButton",
            command=self.request_stop,
        ).grid(row=0, column=5, padx=6, pady=2, sticky="w")

        # Settings
        settings = ttk.LabelFrame(self.root, text="Settings", padding=(10, 8))
        settings.pack(fill=tk.X, padx=10, pady=(2, 4))

        ttk.Label(settings, text="Confidence").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings, textvariable=self.conf_var, width=7).grid(row=0, column=1, padx=(6, 16))

        ttk.Label(settings, text="OCR Interval").grid(row=0, column=2, sticky="w")
        ttk.Entry(settings, textvariable=self.ocr_interval_var, width=7).grid(row=0, column=3, padx=(6, 16))

        ttk.Label(settings, text="Padding").grid(row=0, column=4, sticky="w")
        ttk.Entry(settings, textvariable=self.pad_var, width=7).grid(row=0, column=5, padx=(6, 16))

        # Info
        info = ttk.Frame(self.root, padding=(10, 2, 10, 4))
        info.pack(fill=tk.X)

        ttk.Label(info, textvariable=self.model_var, style="Muted.TLabel").pack(anchor="w")
        ttk.Label(info, textvariable=self.output_var, style="Muted.TLabel").pack(anchor="w")

        self.status_label = tk.Label(
            info,
            textvariable=self.status_var,
            bg=COLORS["bg"],
            fg=COLORS["blue"],
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        self.status_label.pack(fill=tk.X, pady=(2, 0))

        self.progress = ttk.Progressbar(self.root, orient="horizontal", mode="determinate")
        self.progress.pack(fill=tk.X, padx=10, pady=(0, 4))

        # Preview area fills almost all remaining space
        preview_container = ttk.Frame(self.root, padding=(6, 2, 6, 6))
        preview_container.pack(fill=tk.BOTH, expand=True)

        preview_top = ttk.Frame(preview_container)
        preview_top.pack(fill=tk.X, pady=(0, 4))

        self.prev_button = ttk.Button(
            preview_top,
            text="◀ Previous",
            style="Blue.TButton",
            command=self.show_previous_preview,
        )
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = ttk.Button(
            preview_top,
            text="Next ▶",
            style="Blue.TButton",
            command=self.show_next_preview,
        )
        self.next_button.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(
            preview_top,
            textvariable=self.preview_counter_var,
            style="Muted.TLabel",
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT, padx=(12, 0))

        ttk.Label(
            preview_top,
            textvariable=self.preview_title_var,
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.RIGHT)

        self.preview_label = tk.Label(
            preview_container,
            text="No preview yet",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted"],
            font=("Segoe UI", 14),
            anchor="center",
            justify="center",
            bd=0,
            highlightthickness=0,
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        self._update_preview_nav_buttons()

    # -------------------------------------------------------------------------
    # Startup
    # -------------------------------------------------------------------------
    def _startup_load_local_models(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_var.set(f"Output: {self.output_dir}")

        if DETECTOR_MODEL_PATH.is_file():
            self.load_model_button.grid_remove()
            self.model_var.set(f"Detector: {DETECTOR_MODEL_PATH.name} (loading...)")
            self.status_var.set("Loading local detector and OCR model...")
            self._start_model_loading(str(DETECTOR_MODEL_PATH))
        else:
            self.model_var.set("Detector: not found in models/")
            self.status_var.set("Detector not found. Please load a .pt model.")

    # -------------------------------------------------------------------------
    # General helpers
    # -------------------------------------------------------------------------
    def set_busy(self, value: bool) -> None:
        self.busy = value

    def safe_settings(self) -> tuple[float, int, float] | None:
        try:
            confidence = float(self.conf_var.get())
            interval = int(self.ocr_interval_var.get())
            pad_ratio = float(self.pad_var.get())
        except ValueError:
            messagebox.showerror("Invalid settings", "Confidence, OCR interval, and padding must be numeric.")
            return None

        if not 0.0 <= confidence <= 1.0:
            messagebox.showerror("Invalid confidence", "Confidence must be between 0 and 1.")
            return None
        if interval < 1:
            messagebox.showerror("Invalid OCR interval", "OCR interval must be at least 1.")
            return None
        if not 0.0 <= pad_ratio <= 0.5:
            messagebox.showerror("Invalid padding", "Padding must be between 0 and 0.5.")
            return None

        return confidence, interval, pad_ratio

    def ensure_ready(self) -> bool:
        if self.busy:
            messagebox.showwarning("Busy", "Wait for the current task to finish or press Stop.")
            return False
        if self.detector is None or self.ocr_model is None:
            messagebox.showwarning("Models not loaded", "Detector and OCR model must be loaded first.")
            return False
        return True

    def request_stop(self) -> None:
        if self.busy:
            self.stop_event.set()
            self.status_var.set("Stopping...")

    def choose_output_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select output folder", initialdir=str(self.output_dir))
        if folder:
            self.output_dir = Path(folder)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.output_var.set(f"Output: {self.output_dir}")

    def open_output_folder(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(self.output_dir)  # type: ignore[attr-defined]
        except Exception:
            messagebox.showinfo("Output Folder", str(self.output_dir))

    # -------------------------------------------------------------------------
    # Preview helpers
    # -------------------------------------------------------------------------
    def clear_preview_items(self) -> None:
        self.preview_items = []
        self.preview_index = -1
        self.preview_title_var.set("No preview yet")
        self.preview_counter_var.set("")
        self.preview_photo = None
        self.preview_label.configure(image="", text="No preview yet")
        self._update_preview_nav_buttons()

    def add_preview_item(self, title: str, frame_bgr: np.ndarray) -> None:
        self.preview_items.append((title, frame_bgr.copy()))
        self.preview_index = len(self.preview_items) - 1
        self._show_current_preview()

    def show_previous_preview(self) -> None:
        if not self.preview_items:
            return
        self.preview_index = max(0, self.preview_index - 1)
        self._show_current_preview()

    def show_next_preview(self) -> None:
        if not self.preview_items:
            return
        self.preview_index = min(len(self.preview_items) - 1, self.preview_index + 1)
        self._show_current_preview()

    def _update_preview_nav_buttons(self) -> None:
        has_items = len(self.preview_items) > 0
        at_start = self.preview_index <= 0
        at_end = self.preview_index >= len(self.preview_items) - 1 if has_items else True

        self.prev_button.state(["disabled"] if (not has_items or at_start) else ["!disabled"])
        self.next_button.state(["disabled"] if (not has_items or at_end) else ["!disabled"])

    def _show_current_preview(self) -> None:
        if not self.preview_items or self.preview_index < 0:
            self.preview_label.configure(image="", text="No preview yet")
            self.preview_title_var.set("No preview yet")
            self.preview_counter_var.set("")
            self._update_preview_nav_buttons()
            return

        title, frame_bgr = self.preview_items[self.preview_index]
        self.preview_title_var.set(title)
        self.preview_counter_var.set(f"{self.preview_index + 1} / {len(self.preview_items)}")
        self.show_preview(frame_bgr)
        self._update_preview_nav_buttons()

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------
    def choose_model(self) -> None:
        if self.busy:
            messagebox.showwarning("Busy", "Wait for the current task to finish.")
            return

        path = filedialog.askopenfilename(
            title="Select YOLO detector (.pt)",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return

        self._start_model_loading(path)

    def _start_model_loading(self, path: str) -> None:
        self.stop_event.clear()
        self.set_busy(True)
        self.status_var.set("Loading detector and OCR model...")
        threading.Thread(target=self._load_models_worker, args=(path,), daemon=True).start()

    def _load_models_worker(self, path: str) -> None:
        try:
            detector = YOLO(path)

            paddle_device = "gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
            try:
                paddle.set_device(paddle_device)
            except Exception:
                paddle_device = "cpu"
                paddle.set_device("cpu")

            if not OCR_MODEL_DIR.is_dir():
                raise FileNotFoundError(
                    f"OCR model folder not found:\n{OCR_MODEL_DIR}\n\n"
                    "Expected local OCR model in models/latin_PP-OCRv5_mobile_rec/"
                )

            required_files = [
                OCR_MODEL_DIR / "inference.json",
                OCR_MODEL_DIR / "inference.pdiparams",
                OCR_MODEL_DIR / "inference.yml",
            ]
            missing_files = [p.name for p in required_files if not p.is_file()]
            if missing_files:
                raise FileNotFoundError(
                    f"OCR model folder is incomplete.\nMissing: {', '.join(missing_files)}"
                )

            ocr_model = TextRecognition(
                model_name="latin_PP-OCRv5_mobile_rec",
                model_dir=str(OCR_MODEL_DIR),
                device=paddle_device,
            )

            self.events.put(("models_loaded", detector, ocr_model, path, paddle_device))
        except Exception as exc:
            self.events.put(("error", f"Could not load models:\n{exc}"))

    # -------------------------------------------------------------------------
    # Input selection
    # -------------------------------------------------------------------------
    def choose_images(self) -> None:
        if not self.ensure_ready():
            return

        settings = self.safe_settings()
        if settings is None:
            return

        paths = filedialog.askopenfilenames(title="Select image(s)", filetypes=IMAGE_FILETYPES)
        if not paths:
            return

        confidence, _, padding = settings
        self.stop_event.clear()
        self.set_busy(True)
        self.clear_preview_items()
        self.progress.configure(maximum=len(paths), value=0)
        self.status_var.set(f"Processing {len(paths)} image(s)...")

        threading.Thread(
            target=self._process_images_worker,
            args=(list(paths), confidence, padding),
            daemon=True,
        ).start()

    def choose_video(self) -> None:
        if not self.ensure_ready():
            return

        settings = self.safe_settings()
        if settings is None:
            return

        path = filedialog.askopenfilename(title="Select video", filetypes=VIDEO_FILETYPES)
        if not path:
            return

        confidence, interval, padding = settings
        self.stop_event.clear()
        self.set_busy(True)
        self.clear_preview_items()
        self.status_var.set("Opening video...")

        threading.Thread(
            target=self._process_video_worker,
            args=(path, confidence, interval, padding),
            daemon=True,
        ).start()

    # -------------------------------------------------------------------------
    # OCR extraction
    # -------------------------------------------------------------------------
    @staticmethod
    def _extract_ocr_fields(result: Any) -> tuple[str, float]:
        data: Any = result

        if not isinstance(data, dict):
            for method_name in ("to_dict", "json"):
                method = getattr(data, method_name, None)
                if callable(method):
                    try:
                        data = method()
                        break
                    except Exception:
                        pass

        if not isinstance(data, dict):
            return "", 0.0

        payload = data.get("res", data)
        if not isinstance(payload, dict):
            return "", 0.0

        text = payload.get("rec_text", "")
        score = payload.get("rec_score", 0.0)

        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        if isinstance(score, (list, tuple, np.ndarray)):
            score = score[0] if len(score) else 0.0

        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        return normalize_plate(text), score

    def recognize_plate(self, plate_bgr: np.ndarray) -> tuple[str, float]:
        if plate_bgr is None or plate_bgr.size == 0 or self.ocr_model is None:
            return "", 0.0

        try:
            output = list(self.ocr_model.predict(input=plate_bgr, batch_size=1))
            if not output:
                return "", 0.0
            return self._extract_ocr_fields(output[0])
        except Exception:
            return "", 0.0

    # -------------------------------------------------------------------------
    # Detection
    # -------------------------------------------------------------------------
    def detect_plates(self, frame_bgr: np.ndarray, confidence: float) -> list[dict[str, Any]]:
        if self.detector is None:
            return []

        result = self.detector.predict(source=frame_bgr, conf=confidence, verbose=False)[0]
        detections: list[dict[str, Any]] = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "detection_confidence": float(score),
            })

        return detections

    @staticmethod
    def draw_plate(
        image: np.ndarray,
        bbox: list[int],
        text: str,
        det_confidence: float,
        ocr_confidence: float,
        stability: float | None = None,
        track_id: int | None = None,
    ) -> None:
        x1, y1, x2, y2 = bbox
        recognized = bool(text)
        color = (0, 220, 0) if recognized else (0, 165, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        plate_h = max(20, y2 - y1)
        plate_w = max(20, x2 - x1)

        if recognized:
            if stability is None:
                label = f"{text} | D:{det_confidence:.2f} O:{ocr_confidence:.2f}"
            else:
                label = f"{text} | D:{det_confidence:.2f} S:{stability:.2f}"
        else:
            label = f"Plate #{track_id}" if track_id is not None else f"Plate | D:{det_confidence:.2f}"

        font_scale = max(0.85, min(1.25, plate_w / 220.0, plate_h / 22.0))
        thickness = max(2, int(font_scale * 2.3))
        padding = 8

        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )

        label_x1 = max(0, x1)
        label_x2 = min(image.shape[1] - 1, label_x1 + label_width + padding * 2)

        label_y2 = max(label_height + baseline + padding * 2, y1)
        label_y1 = max(0, label_y2 - (label_height + baseline + padding * 2))

        cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), color, -1)

        text_x = label_x1 + padding
        text_y = label_y2 - baseline - padding

        cv2.putText(
            image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    # -------------------------------------------------------------------------
    # Image processing
    # -------------------------------------------------------------------------
    def _process_images_worker(self, paths: list[str], confidence: float, padding: float) -> None:
        rows: list[dict[str, Any]] = []

        try:
            for image_index, path_string in enumerate(paths, start=1):
                if self.stop_event.is_set():
                    break

                path = Path(path_string)
                image = cv2.imread(str(path))
                if image is None:
                    self.events.put(("progress", image_index))
                    continue

                annotated = image.copy()
                detections = self.detect_plates(image, confidence)
                recognized_count = 0

                for plate_index, detection in enumerate(detections, start=1):
                    bbox = detection["bbox"]
                    crop = crop_with_padding(image, *bbox, pad_ratio=padding)
                    processed = preprocess_plate(crop) if crop is not None else None
                    text, ocr_conf = self.recognize_plate(processed) if processed is not None else ("", 0.0)

                    if text:
                        recognized_count += 1

                    self.draw_plate(
                        annotated,
                        bbox,
                        text,
                        detection["detection_confidence"],
                        ocr_conf,
                    )

                    rows.append({
                        "input_file": str(path),
                        "item_type": "image",
                        "frame": "",
                        "plate_index": plate_index,
                        "track_id": "",
                        "recognized_plate": text,
                        "detection_confidence": round(detection["detection_confidence"], 5),
                        "ocr_confidence": round(ocr_conf, 5),
                        "stability": "",
                        "ocr_samples": "",
                    })

                output_path = self.output_dir / f"{path.stem}_detected.jpg"
                cv2.imwrite(str(output_path), annotated)

                title = f"{path.name} | Plates: {len(detections)} | Recognized: {recognized_count}"
                self.events.put(("add_preview_item", title, annotated))
                self.events.put(("progress", image_index))
                self.events.put(("status", f"Processed {image_index}/{len(paths)} image(s)..."))

            report_path = self.output_dir / "image_plate_results.csv"
            self._write_csv(report_path, rows)

            stopped = self.stop_event.is_set()
            msg = (
                f"{'Stopped' if stopped else 'Finished'} image processing.\n"
                f"Saved report:\n{report_path}"
            )
            self.events.put(("done", msg))
        except Exception as exc:
            self.events.put(("error", f"Image processing failed:\n{exc}"))

    # -------------------------------------------------------------------------
    # Video processing
    # -------------------------------------------------------------------------
    def _process_video_worker(
        self,
        video_path_string: str,
        confidence: float,
        ocr_interval: int,
        padding: float,
    ) -> None:
        video_path = Path(video_path_string)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            self.events.put(("error", f"Could not open video:\n{video_path}"))
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = fps if fps > 0 else 30.0

        output_path = self.output_dir / f"{video_path.stem}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

        if not writer.isOpened():
            cap.release()
            self.events.put(("error", f"Could not create output video:\n{output_path}"))
            return

        self.events.put(("video_started", total_frames, f"Video: {frame_width}x{frame_height} @ {fps:.2f} FPS"))

        tracks: dict[int, dict[str, Any]] = {}
        completed_tracks: list[dict[str, Any]] = []
        next_track_id = 1
        frame_number = 0
        summary_rows: list[dict[str, Any]] = []
        start = time.perf_counter()

        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break

                frame_number += 1
                annotated = frame.copy()
                detections = self.detect_plates(frame, confidence)
                matched_ids: set[int] = set()

                for detection in detections:
                    bbox = detection["bbox"]
                    best_id = None
                    best_iou = 0.0

                    for track_id, track in tracks.items():
                        if track_id in matched_ids:
                            continue
                        iou = calculate_iou(bbox, track["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_id = track_id

                    if best_id is not None and best_iou >= 0.30:
                        track_id = best_id
                        track = tracks[track_id]
                        track["bbox"] = bbox
                        track["last_seen"] = frame_number
                        track["detection_confidence"] = detection["detection_confidence"]
                    else:
                        track_id = next_track_id
                        next_track_id += 1
                        track = {
                            "track_id": track_id,
                            "bbox": bbox,
                            "first_seen": frame_number,
                            "last_seen": frame_number,
                            "last_ocr_frame": -ocr_interval,
                            "detection_confidence": detection["detection_confidence"],
                            "latest_ocr_confidence": 0.0,
                            "text_votes": defaultdict(float),
                            "ocr_samples": 0,
                        }
                        tracks[track_id] = track

                    matched_ids.add(track_id)
                    detection["track_id"] = track_id

                expired = [
                    track_id
                    for track_id, track in tracks.items()
                    if frame_number - track["last_seen"] > 30
                ]
                for track_id in expired:
                    completed_tracks.append(tracks.pop(track_id))

                for detection in detections:
                    track = tracks[detection["track_id"]]
                    if frame_number - track["last_ocr_frame"] < ocr_interval:
                        continue

                    crop = crop_with_padding(frame, *detection["bbox"], pad_ratio=padding)
                    processed = preprocess_plate(crop) if crop is not None else None
                    if processed is None:
                        continue

                    text, ocr_conf = self.recognize_plate(processed)
                    track["last_ocr_frame"] = frame_number
                    track["latest_ocr_confidence"] = ocr_conf

                    if text:
                        track["text_votes"][text] += ocr_conf * detection["detection_confidence"]
                        track["ocr_samples"] += 1

                for detection in detections:
                    track = tracks[detection["track_id"]]
                    text, stability = best_track_text(track)

                    self.draw_plate(
                        annotated,
                        detection["bbox"],
                        text,
                        detection["detection_confidence"],
                        track["latest_ocr_confidence"],
                        stability=stability,
                        track_id=detection["track_id"],
                    )

                elapsed = time.perf_counter() - start
                processing_fps = frame_number / elapsed if elapsed > 0 else 0.0
                overlay = f"Frame {frame_number}/{total_frames} | {processing_fps:.1f} FPS"
                cv2.putText(
                    annotated,
                    overlay,
                    (15, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.95,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                writer.write(annotated)

                if frame_number % 5 == 0 or frame_number == total_frames:
                    self.events.put(("preview_video_frame", annotated, video_path.name, frame_number, total_frames))

                if frame_number % 10 == 0 or frame_number == total_frames:
                    self.events.put(("progress", frame_number))
                    self.events.put(("status", f"Frame {frame_number}/{total_frames} | Active tracks: {len(tracks)}"))

            completed_tracks.extend(tracks.values())
            cap.release()
            writer.release()

            for track in completed_tracks:
                text, stability = best_track_text(track)
                summary_rows.append({
                    "input_file": str(video_path),
                    "item_type": "video",
                    "frame": f"{track['first_seen']}-{track['last_seen']}",
                    "plate_index": "",
                    "track_id": track["track_id"],
                    "recognized_plate": text,
                    "detection_confidence": round(track["detection_confidence"], 5),
                    "ocr_confidence": round(track["latest_ocr_confidence"], 5),
                    "stability": round(stability, 5),
                    "ocr_samples": track["ocr_samples"],
                })

            report_path = self.output_dir / f"{video_path.stem}_plate_results.csv"
            self._write_csv(report_path, summary_rows)

            stopped = self.stop_event.is_set()
            final_message = (
                f"Video processing {'stopped' if stopped else 'completed'}.\n"
                f"Saved video:\n{output_path}\n\n"
                f"Saved report:\n{report_path}"
            )
            self.events.put(("done", final_message))

        except Exception as exc:
            cap.release()
            writer.release()
            self.events.put(("error", f"Video processing failed:\n{exc}"))

    # -------------------------------------------------------------------------
    # CSV writer
    # -------------------------------------------------------------------------
    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        base_fields = [
            "input_file",
            "item_type",
            "frame",
            "plate_index",
            "track_id",
            "recognized_plate",
            "detection_confidence",
            "ocr_confidence",
            "stability",
            "ocr_samples",
        ]

        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=base_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    # -------------------------------------------------------------------------
    # Event processing
    # -------------------------------------------------------------------------
    def _process_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]

                if kind == "models_loaded":
                    _, detector, ocr_model, path, paddle_device = event
                    self.detector = detector
                    self.ocr_model = ocr_model
                    self.model_path = path
                    self.model_var.set(f"Detector: {Path(path).name} | OCR device: {paddle_device}")
                    self.status_var.set("Models loaded. Ready.")
                    self.set_busy(False)
                    self.load_model_button.grid_remove()

                elif kind == "add_preview_item":
                    _, title, frame = event
                    self.add_preview_item(title, frame)

                elif kind == "preview_video_frame":
                    _, frame, video_name, frame_number, total_frames = event
                    self.preview_items = [(f"{video_name} | Frame {frame_number}/{total_frames}", frame.copy())]
                    self.preview_index = 0
                    self._show_current_preview()

                elif kind == "progress":
                    self.progress.configure(value=event[1])

                elif kind == "video_started":
                    _, total_frames, description = event
                    self.progress.configure(maximum=max(total_frames, 1), value=0)
                    self.status_var.set(description)

                elif kind == "status":
                    self.status_var.set(event[1])

                elif kind == "done":
                    self.status_var.set("Finished.")
                    self.set_busy(False)
                    self.stop_event.clear()
                    messagebox.showinfo("Finished", event[1])

                elif kind == "error":
                    self.status_var.set("Error.")
                    self.set_busy(False)
                    self.stop_event.clear()
                    if self.detector is None:
                        self.load_model_button.grid()
                    messagebox.showerror("Error", event[1])

        except queue.Empty:
            pass

        self.root.after(80, self._process_events)

    # -------------------------------------------------------------------------
    # Preview rendering
    # -------------------------------------------------------------------------
    def show_preview(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None or frame_bgr.size == 0:
            return

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        self.preview_label.update_idletasks()
        max_width = max(300, self.preview_label.winfo_width())
        max_height = max(300, self.preview_label.winfo_height())

        scale = min(max_width / image.width, max_height / image.height, 1.0)
        new_size = (
            max(1, int(image.width * scale)),
            max(1, int(image.height * scale)),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)

        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_photo, text="")



if __name__ == "__main__":
    app_root = tk.Tk()
    LicensePlateApp(app_root)
    app_root.mainloop()