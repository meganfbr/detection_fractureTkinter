import os
import sys
import json
import numpy as np
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from tensorflow.keras.models import load_model
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image

# ---------------- KONFIGURASI ----------------
MODELS_DIR = "."
MODEL1_PATH = os.path.join(MODELS_DIR, "model1_xray_filterCNN.keras")
MODEL2_PATH = os.path.join(MODELS_DIR, "fractureCNNRev_model.keras")
CLASS1_JSON = os.path.join(MODELS_DIR, "class_names_model1.json")
CLASS2_JSON = os.path.join(MODELS_DIR, "class_names_model2.json")
CM_PNG_PATH = os.path.join(MODELS_DIR, "confusion_matrix_final.png")

IMG_DISPLAY_SIZE = (450, 450)
# ---------------------------------------------

def safe_load_json(path, default=None):
    if not path:
        return default
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Peringatan: gagal memuat JSON {path}: {e}")
    return default

def safe_get_classname(class_names_obj, idx):
    if class_names_obj is None:
        return str(idx)
    try:
        if isinstance(class_names_obj, dict):
            sidx = str(idx)
            if sidx in class_names_obj:
                return class_names_obj[sidx]
            for k, v in class_names_obj.items():
                try:
                    if int(k) == idx:
                        return v
                except:
                    pass
            vals = list(class_names_obj.values())
            return vals[idx] if idx < len(vals) else str(idx)
        elif isinstance(class_names_obj, list):
            return class_names_obj[idx] if idx < len(class_names_obj) else str(idx)
        else:
            return str(idx)
    except Exception:
        return str(idx)

def preprocess_image(img_path, model_name):
    img = Image.open(img_path).convert("RGB")
    if model_name == "model1":
        img = img.resize((224, 224))
    elif model_name == "model2":
        img = img.resize((256, 256))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def safe_load_model_with_compat(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")
    try:
        return load_model(path, compile=False)
    except Exception as e1:
        from tensorflow.keras.layers import (
            InputLayer, BatchNormalization, Conv2D, Dense, MaxPooling2D, Flatten
        )
        return load_model(
            path,
            compile=False,
            custom_objects={
                "InputLayer": InputLayer,
                "BatchNormalization": BatchNormalization,
                "Conv2D": Conv2D,
                "Dense": Dense,
                "MaxPooling2D": MaxPooling2D,
                "Flatten": Flatten,
            },
            safe_mode=False
        )

# ================================================================
class XrayClassifierApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klasifikasi Citra X-Ray (2 Model Pipeline)")
        self.resize(950, 720)

        # Memuat model
        try:
            self.model1 = safe_load_model_with_compat(MODEL1_PATH)
            print("✅ Model 1 berhasil dimuat")
        except Exception as e:
            QtWidgets.QMessage
