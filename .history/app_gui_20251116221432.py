import os
import json
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# -------------------- CONFIG --------------------
MODELS_DIR = "models"  # ganti sesuai lokasi folder model-mu
MODEL1_PATH = os.path.join(MODELS_DIR, "model1_xray_filter.h5")
MODEL2_PATH = os.path.join(MODELS_DIR, "model_fracture.h5")
CLASS1_PATH = os.path.join(MODELS_DIR, "class_names_model1.json")
CLASS2_PATH = os.path.join(MODELS_DIR, "class_names_fracture.json")
METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")    # optional
CM_PNG_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")        # optional

IMG_DISPLAY_SIZE = (400, 400)
MODEL_INPUT_SIZE = (224, 224)  # sesuaikan bila modelmu beda
# ------------------------------------------------

# -------------------- HELPERS --------------------
def safe_load_json(path, default=None):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load JSON {path}: {e}")
            return default
    return default

def prepare_image_for_model(path, target_size=MODEL_INPUT_SIZE):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Gagal membaca gambar.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_array = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img_rgb

def draw_confusion_matrix_image(cm_array, labels, out_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels, cbar=True)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def compute_single_prediction_metrics(cm_array, pred_idx):
    """Compute precision/recall/f1 from a single-sample confusion matrix.
       Note: This is *per-sample* and not representative of model performance."""
    # per-class precision: TP / (TP + FP) where FP = sum column - TP
    # per-class recall:    TP / (TP + FN) where FN = sum row - TP
    eps = 1e-12
    tp = cm_array[pred_idx, pred_idx]
    col_sum = cm_array[:, pred_idx].sum()
    row_sum = cm_array[pred_idx, :].sum()
    precision = float(tp) / (float(col_sum) + eps)
    recall = float(tp) / (float(row_sum) + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    avg_recall = float(np.diag(cm_array).sum()) / (cm_array.sum() + eps)  # trivial here
    return precision, recall, avg_recall, f1
# -------------------------------------------------

# -------------------- LOAD MODELS & METADATA --------------------
# Load class names
class_names_1 = safe_load_json(CLASS1_PATH, default=["Non Xray", "Xray"])
class_names_2 = safe_load_json(CLASS2_PATH, default=[
    "Avulsion fracture", "Comminuted fracture", "Fracture Dislocation",
    "Greenstick fracture", "Hairline Fracture", "Impacted fracture",
    "Longitudinal fracture", "Normal", "Oblique fracture",
    "Pathological fracture", "Spiral fracture"
])

# Load global evaluation metrics if exist (these are fixed values from training eval)
eval_metrics = safe_load_json(METRICS_PATH, default=None)

# Try load confusion matrix image path
cm_image_exists = os.path.exists(CM_PNG_PATH)

# Load models (with checks)
try:
    model1 = load_model(MODEL1_PATH)
    print("Loaded Model1 (X-ray filter).")
except Exception as e:
    messagebox.showerror("Error", f"Failed to load Model1: {e}")
    raise

try:
    model2 = load_model(MODEL2_PATH)
    print("Loaded Model2 (fracture classifier).")
except Exception as e:
    messagebox.showerror("Error", f"Failed to load Model2: {e}")
    raise
# ---------------------------------------------------------------

# -------------------- BUILD GUI --------------------
root = tk.Tk()
root.title("🦴 Bone Fracture Classification - 2-Model Pipeline")
root.geometry("1200x820")
root.configure(bg="#f5f5f5")

# Left: controls + image
left_frame = tk.Frame(root, bg="#f5f5f5", padx=10, pady=10)
left_frame.grid(row=0, column=0, sticky="nsew")

tk.Label(left_frame, text="📤 Upload X-ray Image", font=("Arial", 16, "bold"), bg="#f5f5f5").pack(pady=(5,10))
btn_upload = tk.Button(left_frame, text="Choose Image", font=("Arial", 12, "bold"), bg="#4CAF50", fg="white")
btn_upload.pack(pady=(0,10))

img_panel = tk.Label(left_frame, text="No image", bg="#e0e0e0", width=IMG_DISPLAY_SIZE[0]//10, height=IMG_DISPLAY_SIZE[1]//20,
                     relief="sunken", bd=2)
img_panel.pack(pady=10)

# Right: results + evaluation
right_frame = tk.Frame(root, bg="#f8f9fa", padx=12, pady=12, relief="groove", bd=1)
right_frame.grid(row=0, column=1, sticky="nsew")

tk.Label(right_frame, text="🧠 Prediction Result", font=("Arial", 14, "bold"), bg="#f8f9fa").pack(anchor="w")
pred_text = tk.Label(right_frame, text="No prediction yet", font=("Arial", 12), bg="#f8f9fa", justify="left")
pred_text.pack(anchor="w", pady=(6,12))

tk.Label(right_frame, text="📊 Model Evaluation (global)", font=("Arial", 13, "bold"), bg="#f8f9fa").pack(anchor="w")
metrics_frame = tk.Frame(right_frame, bg="#f8f9fa")
metrics_frame.pack(anchor="w", pady=(6,8))

lbl_accuracy = tk.Label(metrics_frame, text="Accuracy: -", font=("Arial", 11), bg="#f8f9fa")
lbl_precision = tk.Label(metrics_frame, text="Precision: -", font=("Arial", 11), bg="#f8f9fa")
lbl_recall = tk.Label(metrics_frame, text="Recall: -", font=("Arial", 11), bg="#f8f9fa")
lbl_avgrecall = tk.Label(metrics_frame, text="Average Recall: -", font=("Arial", 11), bg="#f8f9fa")
lbl_f1 = tk.Label(metrics_frame, text="F1 Score: -", font=("Arial", 11), bg="#f8f9fa")

lbl_accuracy.grid(row=0, column=0, sticky="w")
lbl_precision.grid(row=1, column=0, sticky="w")
lbl_recall.grid(row=2, column=0, sticky="w")
lbl_avgrecall.grid(row=3, column=0, sticky="w")
lbl_f1.grid(row=4, column=0, sticky="w")

# Confusion matrix display
tk.Label(right_frame, text="🗂 Confusion Matrix", font=("Arial", 13, "bold"), bg="#f8f9fa").pack(anchor="w", pady=(12,4))
cm_canvas = tk.Label(right_frame, bg="#f8f9fa", text="No confusion matrix available")
cm_canvas.pack()

# Make layout expand nicely
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# -------------------- UTILITY: load and show metrics (global) --------------------
def show_global_metrics():
    if eval_metrics:
        lbl_accuracy.config(text=f"Accuracy: {eval_metrics.get('accuracy', '-'): .4f}")
        lbl_precision.config(text=f"Precision: {eval_metrics.get('precision', '-'): .4f}")
        lbl_recall.config(text=f"Recall: {eval_metrics.get('recall', '-'): .4f}")
        lbl_avgrecall.config(text=f"Average Recall: {eval_metrics.get('average_recall', '-'): .4f}")
        lbl_f1.config(text=f"F1 Score: {eval_metrics.get('f1_score', '-'): .4f}")
        # show confusion matrix image if exists
        if cm_image_exists:
            try:
                cm_img = Image.open(CM_PNG_PATH).resize((600, 480))
                cm_img_tk = ImageTk.PhotoImage(cm_img)
                cm_canvas.config(image=cm_img_tk, text="")
                cm_canvas.image = cm_img_tk
            except Exception as e:
                cm_canvas.config(text=f"Failed to load confusion_matrix.png: {e}")
    else:
        lbl_accuracy.config(text="Accuracy: (not provided)")
        lbl_precision.config(text="Precision: (not provided)")
        lbl_recall.config(text="Recall: (not provided)")
        lbl_avgrecall.config(text="Average Recall: (not provided)")
        lbl_f1.config(text="F1 Score: (not provided)")
        cm_canvas.config(text="No global confusion matrix available\n(You can provide models/confusion_matrix.png)")

# Show global metrics on launch (if file present)
show_global_metrics()

# -------------------- PREDICTION PIPELINE --------------------
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if not file_path:
        return

    try:
        # prepare image
        img_array, img_rgb = prepare_image_for_model(file_path, target_size=MODEL_INPUT_SIZE)

        # MODEL 1: X-ray filter
        pred1 = model1.predict(img_array, verbose=0)[0]
        idx1 = int(np.argmax(pred1))
        class1 = class_names_1[idx1]
        conf1 = float(pred1[idx1]) * 100.0

        # display original image on left
        pil_disp = Image.fromarray(img_rgb).resize(IMG_DISPLAY_SIZE)
        tk_img = ImageTk.PhotoImage(pil_disp)
        img_panel.config(image=tk_img, text="")
        img_panel.image = tk_img

        if class1.lower() in ["non xray", "nonxray", "non_xray", "not xray", "not_xray"]:
            # Not an X-ray
            pred_text.config(text=f"Model1: {class1} ({conf1:.2f}%)\nResult: NOT X-RAY — stopped.", fg="orange")
            # still show global metrics if present
            return

        # Else it's X-ray -> MODEL 2
        pred2 = model2.predict(img_array, verbose=0)[0]
        idx2 = int(np.argmax(pred2))
        class2 = class_names_2[idx2]
        conf2 = float(pred2[idx2]) * 100.0

        # Display prediction text
        label_color = "#2e7d32" if class2.lower() == "normal" else "#d32f2f"
        pred_text.config(text=f"Predicted Class: {class2}\nConfidence: {conf2:.2f}%", fg=label_color)

        # Show global metrics (unchanged) if exist
        show_global_metrics()

        # If global confusion matrix not available, create single-sample CM for visualization
        if not cm_image_exists:
            cm = np.zeros((len(class_names_2), len(class_names_2)), dtype=int)
            # We don't know true label for uploaded image, so we show predicted diagonal highlight
            cm[idx2, idx2] = 1
            tmp_path = os.path.join(MODELS_DIR, "tmp_single_cm.png")
            draw_confusion_matrix_image(cm, class_names_2, tmp_path)
            cm_img = Image.open(tmp_path).resize((600, 480))
            cm_img_tk = ImageTk.PhotoImage(cm_img)
            cm_canvas.config(image=cm_img_tk, text="")
            cm_canvas.image = cm_img_tk
        # Done
    except Exception as e:
        messagebox.showerror("Error", f"Error during prediction:\n{e}")

# Bind button
btn_upload.config(command=upload_and_predict)

# -------------------- START APP --------------------
root.mainloop()
