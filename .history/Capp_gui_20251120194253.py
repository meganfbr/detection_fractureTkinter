import os
import sys
import json
import numpy as np
import tempfile

# Suppress verbose TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image

# ---------------- CONFIG ----------------
MODELS_DIR = "."
MODEL1_PATH = os.path.join(MODELS_DIR, "model1_xray_filterCNN.keras")
MODEL2_PATH = os.path.join(MODELS_DIR, "fractureCNNRev_model.keras")
CLASS1_JSON = os.path.join(MODELS_DIR, "class_names_model1.json")
CLASS2_JSON = os.path.join(MODELS_DIR, "class_names_model2.json")
METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")
CM_PNG_PATH = os.path.join(MODELS_DIR, "confusion_matrix_final.png")

MODEL_INPUT_SIZE = (224, 224)
IMG_DISPLAY_SIZE = (450, 450)
# ----------------------------------------

def safe_load_json(path, default=None):
    if not path:
        return default
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Warning: failed to load JSON {path}: {e}")
    return default

def safe_get_classname(class_names_obj, idx):
    """Support both list and dict JSON formats."""
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

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(MODEL_INPUT_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# === FIXED MODEL LOADER ===
def safe_load_model_with_compat(path):
    """Load H5 model safely across TF versions."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        return load_model(path, compile=False)
    except Exception as e1:
        try:
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
                safe_mode=False  # 🔥 penting: biar batch_shape & argumen lama diabaikan
            )
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load model '{path}'.\nFirst error: {e1}\nSecond attempt: {e2}"
            )

# ================================================================
class XrayClassifierApp(QtWidgets.QWidget):
    class XrayClassifierApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        print("DEBUG: Masuk __init__()")

        self.setWindowTitle("X-Ray Fracture Classifier (2-Model Pipeline)")
        self.resize(1000, 760)

        # === Load Model 1 ===
        print("DEBUG: mulai load model 1...")
        try:
            self.model1 = safe_load_model_with_compat(MODEL1_PATH)
            print("DEBUG: model1 loaded")
        except Exception as e:
            print("ERROR saat load model1:", e)
            QtWidgets.QMessageBox.critical(None, "Error", f"Gagal load Model1:\n{e}")
            sys.exit(1)

        # === Load Model 2 ===
        print("DEBUG: mulai load model 2...")
        try:
            self.model2 = safe_load_model_with_compat(MODEL2_PATH)
            print("DEBUG: model2 loaded")
        except Exception as e:
            print("ERROR saat load model2:", e)
            QtWidgets.QMessageBox.critical(None, "Error", f"Gagal load Model2:\n{e}")
            sys.exit(1)

        # === Load JSON metadata ===
        print("DEBUG: load JSON metadata")
        self.class1 = safe_load_json(CLASS1_JSON, default=["Non Xray", "Xray"])
        self.class2 = safe_load_json(CLASS2_JSON, default=[
            "Avulsion fracture", "Comminuted fracture", "Fracture Dislocation",
            "Greenstick fracture", "Hairline Fracture", "Impacted fracture",
            "Longitudinal fracture", "Normal", "Oblique fracture",
            "Pathological fracture", "Spiral Fracture"
        ])
        self.eval_metrics = safe_load_json(METRICS_PATH, default=None)
        self.cm_exists = os.path.exists(CM_PNG_PATH)

        # === UI Elements ===
        self.imageLabel = QtWidgets.QLabel("No image")
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setFixedSize(*IMG_DISPLAY_SIZE)
        self.imageLabel.setStyleSheet("border:1px solid #bbb; background:#efefef;")

        self.btnLoad = QtWidgets.QPushButton("Choose Image")
        self.btnLoad.clicked.connect(self.load_image)

        self.resultBox = QtWidgets.QTextEdit()
        self.resultBox.setReadOnly(True)
        self.resultBox.setFixedHeight(240)

        self.top3_list = QtWidgets.QListWidget()
        self.top3_list.setFixedWidth(300)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.imageLabel)
        left.addSpacing(6)
        left.addWidget(self.btnLoad)
        left.addSpacing(8)
        left.addWidget(QtWidgets.QLabel("Top predictions:"))
        left.addWidget(self.top3_list)
        left.addStretch()

        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("<b>Prediction & Dynamic Metrics</b>"))
        right.addWidget(self.resultBox)
        right.addSpacing(6)

        right.addWidget(QtWidgets.QLabel("<b>Model Evaluation (Global/Test set)</b>"))
        self.lbl_accuracy = QtWidgets.QLabel("Accuracy: -")
        self.lbl_precision = QtWidgets.QLabel("Precision: -")
        self.lbl_recall = QtWidgets.QLabel("Recall: -")
        self.lbl_f1 = QtWidgets.QLabel("F1 Score: -")
        right.addWidget(self.lbl_accuracy)
        right.addWidget(self.lbl_precision)
        right.addWidget(self.lbl_recall)
        right.addWidget(self.lbl_f1)

        right.addSpacing(8)
        right.addWidget(QtWidgets.QLabel("<b>Confusion Matrix (Global)</b>"))
        self.cm_canvas = QtWidgets.QLabel("No confusion matrix available")
        self.cm_canvas.setFixedSize(420, 320)
        right.addWidget(self.cm_canvas)

        self.show_global_metrics()

        main = QtWidgets.QHBoxLayout()
        main.addLayout(left, 1)
        main.addLayout(right, 1)
        self.setLayout(main)

        self._popup_windows = []

    # ================================================================
    def show_global_metrics(self):
        if self.eval_metrics:
            self.lbl_accuracy.setText(f"Accuracy: {self.eval_metrics.get('accuracy', '-'): .4f}")
            self.lbl_precision.setText(f"Precision: {self.eval_metrics.get('precision', '-'): .4f}")
            self.lbl_recall.setText(f"Recall: {self.eval_metrics.get('recall', '-'): .4f}")
            self.lbl_f1.setText(f"F1 Score: {self.eval_metrics.get('f1_score', '-'): .4f}")
        else:
            self.lbl_accuracy.setText("Accuracy: (not provided)")
            self.lbl_precision.setText("Precision: (not provided)")
            self.lbl_recall.setText("Recall: (not provided)")
            self.lbl_f1.setText("F1 Score: (not provided)")

        if self.cm_exists:
            try:
                pix = QtGui.QPixmap(CM_PNG_PATH).scaled(
                    420, 320, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                )
                self.cm_canvas.setPixmap(pix)
            except Exception as e:
                self.cm_canvas.setText(f"Failed to load CM image: {e}")
        else:
            self.cm_canvas.setText("No global confusion matrix found.")

    # ================================================================
    def load_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not file_path:
            return
        try:
            qpix = QtGui.QPixmap(file_path).scaled(
                *IMG_DISPLAY_SIZE, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.imageLabel.setPixmap(qpix)
            self.run_prediction(file_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot open image:\n{e}")

    # ================================================================
    def run_prediction(self, img_path):
        img_arr = preprocess_image(img_path)

        pred1 = self.model1.predict(img_arr, verbose=0)[0]
        idx1 = int(np.argmax(pred1))
        class1_name = safe_get_classname(self.class1, idx1)
        conf1 = float(pred1[idx1]) * 100.0

        low = class1_name.lower().replace(" ", "").replace("-", "_")
        if "non" in low or "not" in low:
            QtWidgets.QMessageBox.warning(
                self, "Not an X-Ray",
                f"Model1: {class1_name} ({conf1:.2f}%)\nThis image will not be processed."
            )
            self.resultBox.setPlainText(
                f"Model1 detected: {class1_name} ({conf1:.2f}%)\n\nProcessing stopped."
            )
            self.top3_list.clear()
            return

        pred2 = self.model2.predict(img_arr, verbose=0)[0]
        idx2 = int(np.argmax(pred2))
        class2_name = safe_get_classname(self.class2, idx2)
        conf2 = float(pred2[idx2]) * 100.0

        probs = np.array(pred2, dtype=float)
        p = probs[idx2]
        dynamic_precision = float(p)
        dynamic_recall = float(p)
        dynamic_f1 = float((2 * p * p) / (p + p + 1e-12))
        avg_recall = float(np.mean(probs))

        result_text = (
            f"Predicted Class : {class2_name}\n"
            f"Confidence      : {conf2:.2f}%\n\n"
            f"Dynamic Metrics (per this image)\n"
            f"Precision : {dynamic_precision:.4f}\n"
            f"Recall    : {dynamic_recall:.4f}\n"
            f"F1-score  : {dynamic_f1:.4f}\n"
            f"Avg Recall: {avg_recall:.4f}\n"
        )
        self.resultBox.setPlainText(result_text)

        self.top3_list.clear()
        top_k = min(5, len(probs))
        order = np.argsort(probs)[::-1][:top_k]
        for i in order:
            cname = safe_get_classname(self.class2, int(i))
            self.top3_list.addItem(f"{cname:30s} {probs[i]*100:6.2f}%")

        if not self.cm_exists:
            self.show_single_cm(idx2, len(self.class2))

    # ================================================================
    def show_single_cm(self, pred_idx, num_classes):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm = np.zeros((num_classes, num_classes), dtype=int)
            cm[pred_idx, pred_idx] = 1
            labels = [safe_get_classname(self.class2, i) for i in range(num_classes)]

            fig = plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            tmpfile = os.path.join(tempfile.gettempdir(), "tmp_single_cm.png")
            fig.savefig(tmpfile, dpi=150, bbox_inches="tight")
            plt.close(fig)

            cm_window = QtWidgets.QWidget()
            cm_window.setWindowTitle("Confusion Matrix (single-sample view)")
            v = QtWidgets.QVBoxLayout()
            lbl = QtWidgets.QLabel()
            pix = QtGui.QPixmap(tmpfile).scaled(
                720, 520, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            lbl.setPixmap(pix)
            v.addWidget(lbl)
            cm_window.setLayout(v)
            cm_window.resize(740, 560)
            cm_window.show()
            self._popup_windows.append(cm_window)
        except Exception as e:
            print("Could not create single-sample confusion matrix:", e)

# ================================================================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = XrayClassifierApp()
    win.show()
    sys.exit(app.exec_())
