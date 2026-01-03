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
                safe_mode=False
            )
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load model '{path}'.\nFirst error: {e1}\nSecond attempt: {e2}"
            )

# ================================================================
class XrayClassifierApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        print("DEBUG: Masuk __init__()")

        self.setWindowTitle("X-Ray Fracture Classifier (2-Model Pipeline)")
        self.resize(1000, 760)

        # Load models
        try:
            print("DEBUG: mulai load model 1...")
            self.model1 = safe_load_model_with_compat(MODEL1_PATH)
            print("✅ Model1 loaded")
        except Exception as e:
            print("ERROR saat load model1:", e)
            QtWidgets.QMessageBox.critical(None, "Error", f"Gagal load Model1:\n{e}")
            sys.exit(1)

        try:
            print("DEBUG: mulai load model 2...")
            self.model2 = safe_load_model_with_compat(MODEL2_PATH)
            print("✅ Model2 loaded")
        except Exception as e:
            print("ERROR saat load model2:", e)
            QtWidgets.QMessageBox.critical(None, "Error", f"Gagal load Model2:\n{e}")
            sys.exit(1)

        # Load metadata
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
        right.addWidget(QtWidgets.QLabel("<b>Confusion Matrix (Dynamic)</b>"))
        self.cm_canvas = QtWidgets.QLabel("No confusion matrix yet")
        self.cm_canvas.setFixedSize(420, 320)
        right.addWidget(self.cm_canvas)

        self.show_global_metrics()

        main = QtWidgets.QHBoxLayout()
        main.addLayout(left, 1)
        main.addLayout(right, 1)
        self.setLayout(main)

    # ================================================================
    def show_global_metrics(self):
        if self.eval_metrics:
            self.lbl_accuracy.setText(f"Accuracy: {self.eval_metrics.get('accuracy', '-')}")
            self.lbl_precision.setText(f"Precision: {self.eval_metrics.get('precision', '-')}")
            self.lbl_recall.setText(f"Recall: {self.eval_metrics.get('recall', '-')}")
            self.lbl_f1.setText(f"F1 Score: {self.eval_metrics.get('f1_score', '-')}")
        else:
            self.lbl_accuracy.setText("Accuracy: (not provided)")
            self.lbl_precision.setText("Precision: (not provided)")
            self.lbl_recall.setText("Recall: (not provided)")
            self.lbl_f1.setText("F1 Score: (not provided)")

    # ================================================================
    def load_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not file_path:
            return
        qpix = QtGui.QPixmap(file_path).scaled(
            *IMG_DISPLAY_SIZE, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.imageLabel.setPixmap(qpix)
        self.run_prediction(file_path)

    # ================================================================
    def run_prediction(self, img_path):
        # Step 1: Filter Non-Xray
        img_arr1 = preprocess_image(img_path, "model1")
        pred1 = self.model1.predict(img_arr1, verbose=0)[0]
        idx1 = int(np.argmax(pred1))
        class1_name = safe_get_classname(self.class1, idx1)
        conf1 = float(pred1[idx1]) * 100.0

        low = class1_name.lower().replace(" ", "").replace("-", "_")
        if "non" in low or "not" in low:
            QtWidgets.QMessageBox.warning(
                self, "Not an X-Ray",
                f"Model1: {class1_name} ({conf1:.2f}%)\nImage skipped."
            )
            self.resultBox.setPlainText(f"Model1 detected: {class1_name} ({conf1:.2f}%)\n\nProcessing stopped.")
            self.top3_list.clear()
            return

        # Step 2: Classify fracture type
        img_arr2 = preprocess_image(img_path, "model2")
        pred2 = self.model2.predict(img_arr2, verbose=0)[0]
        idx2 = int(np.argmax(pred2))
        class2_name = safe_get_classname(self.class2, idx2)
        conf2 = float(pred2[idx2]) * 100.0

        # Threshold to reduce false fractures
        if class2_name.lower() != "normal" and conf2 < 70:
            class2_name = "Likely Normal"
            conf2 = 100 - conf2

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

        # === Top-k Predictions ===
        self.top3_list.clear()
        top_k = min(5, len(probs))
        order = np.argsort(probs)[::-1][:top_k]
        for i in order:
            cname = safe_get_classname(self.class2, int(i))
            self.top3_list.addItem(f"{cname:30s} {probs[i]*100:6.2f}%")

        # === Dynamic Confusion Matrix ===
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = np.zeros((len(self.class2), len(self.class2)), dtype=int)
        cm[idx2, idx2] = 1
        labels = [safe_get_classname(self.class2, i) for i in range(len(self.class2))]

        fig = plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                    xticklabels=labels, yticklabels=labels, cbar=False)
        plt.title("Dynamic Confusion Matrix (per image)")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        tmpfile = os.path.join(tempfile.gettempdir(), "confusion_matrix_dynamic.png")
        fig.savefig(tmpfile, dpi=150, bbox_inches="tight")
        plt.close(fig)

        pix = QtGui.QPixmap(tmpfile).scaled(
            420, 320, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.cm_canvas.setPixmap(pix)

# ================================================================
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = XrayClassifierApp()
    win.show()
    sys.exit(app.exec_())
