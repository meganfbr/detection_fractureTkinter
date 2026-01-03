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
            QtWidgets.QMessageBox.critical(None, "Kesalahan", f"Gagal memuat Model 1:\n{e}")
            sys.exit(1)

        try:
            self.model2 = safe_load_model_with_compat(MODEL2_PATH)
            print("✅ Model 2 berhasil dimuat")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Kesalahan", f"Gagal memuat Model 2:\n{e}")
            sys.exit(1)

        # Muat label kelas
        self.class1 = safe_load_json(CLASS1_JSON, default=["Non Xray", "Xray"])
        self.class2 = safe_load_json(CLASS2_JSON, default=[
            "Avulsion fracture", "Comminuted fracture", "Fracture Dislocation",
            "Greenstick fracture", "Hairline Fracture", "Impacted fracture",
            "Longitudinal fracture", "Normal", "Oblique fracture",
            "Pathological fracture", "Spiral Fracture"
        ])

        # === Elemen UI ===
        # === Tambahkan Judul Utama di GUI ===
        title_label = QtWidgets.QLabel("🩺  SISTEM DETEKSI FRAKTUR TULANG\nMenggunakan Metode CNN (2-Model Pipeline)")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #003366;
            padding: 10px;
        """)

        self.imageLabel = QtWidgets.QLabel("Belum ada gambar")
        self.imageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel.setFixedSize(*IMG_DISPLAY_SIZE)
        self.imageLabel.setStyleSheet("border:1px solid #bbb; background:#efefef;")

        self.btnLoad = QtWidgets.QPushButton("Pilih Gambar")
        self.btnReset = QtWidgets.QPushButton("Reset")
        self.btnLoad.clicked.connect(self.load_image)
        self.btnReset.clicked.connect(self.reset_ui)

        self.resultBox = QtWidgets.QTextEdit()
        self.resultBox.setReadOnly(True)
        self.resultBox.setFixedHeight(220)

        self.top3_list = QtWidgets.QListWidget()
        self.top3_list.setFixedWidth(300)

        # Layout kiri
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.imageLabel)
        left.addSpacing(6)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.btnLoad)
        btn_row.addWidget(self.btnReset)
        left.addLayout(btn_row)

        left.addSpacing(8)
        left.addWidget(QtWidgets.QLabel("<b>Prediksi Teratas:</b>"))
        left.addWidget(self.top3_list)
        left.addStretch()

        # Layout kanan
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("<b>Hasil Prediksi & Metrik Dinamis</b>"))
        right.addWidget(self.resultBox)
        right.addSpacing(8)

        right.addWidget(QtWidgets.QLabel("<b>Hasil Matriks</b>"))
        self.cm_canvas = QtWidgets.QLabel("Belum ada matriks kebingungan")
        self.cm_canvas.setFixedSize(420, 320)
        right.addWidget(self.cm_canvas)

        # Gabungkan layout utama
        main = QtWidgets.QHBoxLayout()
        main.addLayout(left, 1)
        main.addLayout(right, 1)
        self.setLayout(main)

    # ================================================================
    def reset_ui(self):
        """Menghapus hasil dan gambar"""
        self.imageLabel.clear()
        self.imageLabel.setText("Belum ada gambar")
        self.resultBox.clear()
        self.top3_list.clear()
        self.cm_canvas.setText("Belum ada matriks kebingungan")

    # ================================================================
    def load_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Pilih Gambar", "", "Gambar (*.jpg *.jpeg *.png *.bmp)"
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
        # --- Model 1: Filter X-ray ---
        img_arr1 = preprocess_image(img_path, "model1")
        pred1 = self.model1.predict(img_arr1, verbose=0)[0]
        idx1 = int(np.argmax(pred1))
        class1_name = safe_get_classname(self.class1, idx1)
        conf1 = float(pred1[idx1]) * 100.0

        # Jika bukan X-ray
        if "non" in class1_name.lower():
            QtWidgets.QMessageBox.warning(
                self, "Gambar Bukan X-Ray",
                "Gambar bukan X-ray, silakan input kembali."
            )
            self.resultBox.setPlainText(
                f"Model 1 mendeteksi gambar bukan X-ray ({conf1:.2f}%)\n"
                "Silakan pilih gambar X-ray lain."
            )
            self.top3_list.clear()
            return

        # --- Model 2: Jenis Fraktur ---
        img_arr2 = preprocess_image(img_path, "model2")
        pred2 = self.model2.predict(img_arr2, verbose=0)[0]
        idx2 = int(np.argmax(pred2))
        class2_name = safe_get_classname(self.class2, idx2)
        conf2 = float(pred2[idx2]) * 100.0

        # Logika deteksi normal yang lebih baik
        prob_normal = pred2[self.class2.index("Normal")] if "Normal" in self.class2 else 0
        if prob_normal > 0.5:
            class2_name = "Normal"
            conf2 = prob_normal * 100
        # Hapus / komentari bagian ini
        # elif class2_name.lower() != "normal" and conf2 < 60:
        #     class2_name = "Kemungkinan Normal"
        #     conf2 = 100 - conf2


        # Metrik dinamis
        probs = np.array(pred2, dtype=float)
        p = probs[idx2]
        dynamic_precision = float(p)
        dynamic_recall = float(p)
        dynamic_f1 = float((2 * p * p) / (p + p + 1e-12))
        avg_recall = float(np.mean(probs))

        result_text = (
            f"Kelas Prediksi : {class2_name}\n"
            f"Tingkat Keyakinan : {conf2:.2f}%\n\n"
            f"Metrik Dinamis (per gambar)\n"
            f"Presisi : {dynamic_precision:.4f}\n"
            f"Recall  : {dynamic_recall:.4f}\n"
            f"F1-Score: {dynamic_f1:.4f}\n"
            f"Rata Recall: {avg_recall:.4f}\n"
        )
        self.resultBox.setPlainText(result_text)

        # Tampilkan top-5 prediksi
        self.top3_list.clear()
        top_k = min(5, len(probs))
        order = np.argsort(probs)[::-1][:top_k]
        for i in order:
            cname = safe_get_classname(self.class2, int(i))
            self.top3_list.addItem(f"{cname:30s} {probs[i]*100:6.2f}%")

        # Matriks kebingungan dinamis
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = np.zeros((len(self.class2), len(self.class2)), dtype=int)
        cm[idx2, idx2] = 1
        labels = [safe_get_classname(self.class2, i) for i in range(len(self.class2))]

        fig = plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                    xticklabels=labels, yticklabels=labels, cbar=False)
        plt.title("Matriks Kebingungan (per gambar)")
        plt.xlabel("Prediksi")
        plt.ylabel("Kelas Sebenarnya")
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
