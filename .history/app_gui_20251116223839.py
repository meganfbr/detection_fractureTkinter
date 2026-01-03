import sys
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PyQt5 import QtWidgets, QtGui, QtCore
from PIL import Image
import cv2

# ================= CONFIG ==================
MODEL1_PATH = "model1_xray_filter.h5"
MODEL2_PATH = "fractureCNNRev_model.h5"
CLASS1_JSON = "class_names_model1.json"
CLASS2_JSON = "class_names_fracture.json"
METRICS_PATH = "evaluation_metrics.json"
CM_PNG_PATH = "confusion_matrix_final.png"

MODEL_INPUT_SIZE = (224, 224)
IMG_DISPLAY_SIZE = (450, 450)
# ============================================


# Load model safely
def safe_load_model(path):
    try:
        return load_model(path)
    except Exception as e:
        QtWidgets.QMessageBox.critical(None, "Error", f"Gagal Load Model:\n{e}")
        return None


# Load JSON safely
def safe_load_json(path, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return default


# Preprocess for model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=MODEL_INPUT_SIZE)
    img_arr = image.img_to_array(img)
    img_arr = img_arr / 255.0
    return np.expand_dims(img_arr, axis=0)


# GUI Class
class XrayClassifierApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("X-Ray Fracture Classifier")
        self.resize(900, 700)

        # Load models
        self.model1 = safe_load_model(MODEL1_PATH)
        self.model2 = safe_load_model(MODEL2_PATH)

        # Load class names
        self.class1_names = safe_load_json(CLASS1_JSON)
        self.class2_names = safe_load_json(CLASS2_JSON)

        # Load evaluation metrics for display
        self.eval_metrics = safe_load_json(METRICS_PATH)

        # UI Elements
        self.imageLabel = QtWidgets.QLabel()
        self.imageLabel.setFixedSize(*IMG_DISPLAY_SIZE)
        self.imageLabel.setStyleSheet("border: 1px solid #bbb; background: #eee;")

        self.btnLoad = QtWidgets.QPushButton("Upload Image")
        self.btnLoad.clicked.connect(self.load_image)

        self.resultBox = QtWidgets.QTextEdit()
        self.resultBox.setReadOnly(True)
        self.resultBox.setFixedHeight(200)

        # Layouts
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.btnLoad)
        layout.addWidget(self.resultBox)

        self.setLayout(layout)

    # When selecting image
    def load_image(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.jpg *.png *.jpeg)"
        )
        if not file_path:
            return

        # Display image
        pixmap = QtGui.QPixmap(file_path)
        pixmap = pixmap.scaled(*IMG_DISPLAY_SIZE, QtCore.Qt.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)

        # Predict (Model 1 → Model 2)
        self.run_prediction(file_path)

    def run_prediction(self, img_path):
        img = preprocess_image(img_path)

        # MODEL 1 — Xray Filter
        pred1 = self.model1.predict(img)[0]
        idx1 = np.argmax(pred1)
        class1 = self.class1_names[str(idx1)]
        confidence1 = pred1[idx1] * 100

        # If non-xray → popup + stop
        if class1.lower() == "non_xray" or "non" in class1.lower():
            QtWidgets.QMessageBox.warning(
                self,
                "Not an X-Ray",
                "Gambar bukan X-ray.\nTidak dapat diproses."
            )
            return

        # MODEL 2 — Fracture classification
        pred2 = self.model2.predict(img)[0]
        idx2 = np.argmax(pred2)

        class2 = self.class2_names[str(idx2)]
        confidence2 = pred2[idx2] * 100

        # Show results
        result_text = f"""
Predicted Class: {class2}
Confidence: {confidence2:.2f} %

Precision: {self.eval_metrics['precision']:.2f}
Recall: {self.eval_metrics['recall']:.2f}
F1-score: {self.eval_metrics['f1_score']:.2f}
Avg Recall: {self.eval_metrics['avg_recall']:.2f}
"""

        self.resultBox.setText(result_text)

        # Show confusion matrix window
        self.show_confusion_matrix()

    def show_confusion_matrix(self):
        cm_window = QtWidgets.QWidget()
        cm_window.setWindowTitle("Confusion Matrix")
        layout = QtWidgets.QVBoxLayout()

        label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(CM_PNG_PATH)
        pixmap = pixmap.scaled(500, 500, QtCore.Qt.KeepAspectRatio)
        label.setPixmap(pixmap)

        layout.addWidget(label)
        cm_window.setLayout(layout)
        cm_window.resize(600, 600)
        cm_window.show()


# Run Application
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = XrayClassifierApp()
    win.show()
    sys.exit(app.exec_())
