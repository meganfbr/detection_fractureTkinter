import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import Tk, Button, Label, filedialog, Frame, messagebox
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import statistics

# ==================== KONFIGURASI ====================
MODEL_PATH = "fractureCNNRev_model.h5"
CLASS_NAMES_JSON = "class_names.json"
METRICS_JSON = "evaluation_metrics.json"
SAMPLE_XRAY_DIR = "Bone Break Classification/Normal"

default_class_names = [
    "Avulsion fracture", "Comminuted fracture", "Fracture Dislocation",
    "Greenstick fracture", "Hairline Fracture", "Impacted fracture",
    "Longitudinal fracture", "Normal", "Oblique fracture",
    "Pathological fracture", "Spiral fracture"
]

# ==================== INISIALISASI GUI ====================
root = Tk()
root.withdraw()

# ==================== LOAD CLASS NAMES ====================
if os.path.exists(CLASS_NAMES_JSON):
    try:
        with open(CLASS_NAMES_JSON, "r") as f:
            class_names = json.load(f)
        print("✅ Loaded class names from JSON file.")
    except Exception as e:
        print("⚠️ Gagal load class_names.json, pakai default. Error:", e)
        class_names = default_class_names
else:
    class_names = default_class_names

# ==================== LOAD MODEL ====================
if not os.path.exists(MODEL_PATH):
    messagebox.showerror("Error", f"Model file '{MODEL_PATH}' tidak ditemukan.")
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' tidak ditemukan.")
else:
    model = load_model(MODEL_PATH)
    print("✅ Model berhasil dimuat!")

NUM_CLASSES = len(class_names)

# ======================================================
# ⚙️ ANALISIS CITRA NORMAL UNTUK PARAMETER ADAPTIF
# ======================================================
def analyze_xray_samples(sample_dir, max_samples=50):
    sat_vals, bright_vals, contrast_vals = [], [], []
    img_files = []
    for root, _, files in os.walk(sample_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_files.append(os.path.join(root, f))
    img_files = img_files[:max_samples]
    if not img_files:
        return None

    for path in img_files:
        img = cv2.imread(path)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sat_vals.append(np.mean(hsv[:, :, 1]))
        bright_vals.append(np.mean(hsv[:, :, 2]))
        contrast_vals.append(np.std(gray))

    stats = {
        "sat_mean": statistics.median(sat_vals),
        "bright_mean": statistics.median(bright_vals),
        "contrast_mean": statistics.median(contrast_vals),
        "sat_std": statistics.pstdev(sat_vals),
        "bright_std": statistics.pstdev(bright_vals),
        "contrast_std": statistics.pstdev(contrast_vals)
    }
    print("✅ Adaptive X-ray threshold computed.")
    return stats

adaptive_stats = analyze_xray_samples(SAMPLE_XRAY_DIR)

# ======================================================
# ⚙️ DETEKSI GAMBAR X-RAY (ANTI FOTO SELFI, TOLERAN X-RAY BIRU)
# ======================================================
def is_xray_image(img_path):
    img = cv2.imread(img_path)
    if img is None or adaptive_stats is None:
        print("⚠️ Gagal membaca gambar untuk deteksi X-ray.")
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sat = np.mean(hsv[:, :, 1])
    bright = np.mean(hsv[:, :, 2])
    contrast = np.std(gray)
    mean_bgr = np.mean(img, axis=(0, 1))
    color_diff = np.max(mean_bgr) - np.min(mean_bgr)
    gray_mean = np.mean(gray)

    s_thr = adaptive_stats["sat_mean"] + 3.5 * adaptive_stats["sat_std"]
    b_thr_low = adaptive_stats["bright_mean"] - 3.5 * adaptive_stats["bright_std"]
    b_thr_high = adaptive_stats["bright_mean"] + 3.5 * adaptive_stats["bright_std"]
    c_thr_low = adaptive_stats["contrast_mean"] - 1.0 * adaptive_stats["contrast_std"]

    # 1️⃣ Tolak jika sangat berwarna
    if sat > s_thr or color_diff > 55:
        print("❌ Terlalu berwarna — bukan X-ray.")
        return False

    # 2️⃣ Tolak jika terlalu terang / gelap / tanpa kontras
    if bright < b_thr_low or bright > b_thr_high:
        print("❌ Terlalu terang atau terlalu gelap — bukan X-ray.")
        return False
    if contrast < c_thr_low:
        print("❌ Kontras terlalu rendah — kemungkinan foto biasa.")
        return False

    # 3️⃣ Tolak pola warna kulit
    if mean_bgr[2] > mean_bgr[1] > mean_bgr[0] and mean_bgr[2] - mean_bgr[0] > 20:
        print("❌ Warna kulit terdeteksi — bukan X-ray.")
        return False

    # 4️⃣ Deteksi X-ray kebiruan
    if mean_bgr[0] > mean_bgr[1] and mean_bgr[0] > mean_bgr[2]:
        blue_strength = mean_bgr[0] - ((mean_bgr[1] + mean_bgr[2]) / 2)
        if blue_strength < 40 and sat < 60:
            print("✅ X-ray kebiruan pudar terdeteksi.")
            return True
        else:
            print("❌ Biru terlalu jenuh — bukan X-ray medis.")
            return False

    # 5️⃣ Jika grayscale normal (abu-abu khas X-ray)
    if 25 < gray_mean < 230 and contrast > 15:
        print("✅ X-ray valid (grayscale).")
        return True

    print("❌ Tidak memenuhi karakteristik X-ray.")
    return False

def normalize_external_xray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    gray_norm = cv2.normalize(gray_eq, None, 0, 255, cv2.NORM_MINMAX)
    img_rgb = cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2RGB)
    return img_rgb


# ======================================================
# 🧠 UPLOAD & PREDIKSI (AUTO-NORMALISASI UNTUK GAMBAR LUAR DATASET)
# ======================================================
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    try:
        img_cv = cv2.imread(file_path)
        if img_cv is None:
            raise ValueError("Gagal membaca gambar.")

        # 🔍 Cek apakah gambar tampak seperti X-ray
        is_valid_xray = is_xray_image(file_path)

        # Jika bukan X-ray medis tapi masih bisa dikonversi → tetap proses
        if not is_valid_xray:
            print("⚠️ Gambar luar dataset - mencoba konversi grayscale untuk proses prediksi.")
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            img_cv = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        else:
            print("✅ Gambar valid X-ray - lanjut prediksi.")

        # 🔄 Ubah ke grayscale → RGB (biar seragam ke model)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # 🔁 Resize dan normalisasi
        img_resized = cv2.resize(img_rgb, (256, 256))
        img_array = np.expand_dims(img_resized / 255.0, axis=0)

        # 🧠 Prediksi
        preds = model.predict(img_array, verbose=0)[0]
        class_index = int(np.argmax(preds))
        predicted_class = class_names[class_index]
        confidence = float(preds[class_index]) * 100.0
        label_text = "Normal (Tidak Patah)" if predicted_class.lower() == "normal" else "Fracture (Patah Tulang)"

        # 🖼️ Tampilkan hasil gambar
        img_disp = Image.open(file_path).convert("RGB").resize((400, 400))
        img_disp = ImageTk.PhotoImage(img_disp)
        img_label.config(image=img_disp, text="")
        img_label.image = img_disp

        # 💬 Tampilkan hasil prediksi
        result_label.config(
            text=f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}%\nLabel: {label_text}",
            fg="#2e7d32" if predicted_class.lower() == "normal" else "#d32f2f"
        )

        # 📊 Confusion matrix mini
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        cm[class_index, class_index] = 1
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=False, fmt="d", linewidths=0.5)
        plt.title(f"Confusion Matrix (Predicted: {predicted_class})")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        tmp_path = "confusion_matrix_temp.png"
        plt.savefig(tmp_path, bbox_inches="tight", dpi=150)
        plt.close()

        cm_img = Image.open(tmp_path).resize((600, 480))
        cm_img = ImageTk.PhotoImage(cm_img)
        cm_label.config(image=cm_img, text="")
        cm_label.image = cm_img

    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan saat prediksi:\n{str(e)}")

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    try:
        # Baca dan konversi gambar
        img_cv = cv2.imread(file_path)
        if img_cv is None:
            raise ValueError("Gagal membaca gambar.")

        # Deteksi validitas X-ray
        is_valid_xray = is_xray_image(file_path)

        if not is_xray_image(file_path):
            messagebox.showwarning("Gambar Tidak Valid", "Gambar yang diunggah bukan X-ray! Silakan pilih gambar X-ray medis.")
            img_label.config(image="", text="❌ Input bukan X-ray", fg="red")
            result_label.config(text="")
            cm_label.config(image="", text="Masukan Gambar Untuk Melihat Matrix")
            return



        # 1️⃣ Deteksi apakah gambar dari luar dataset
        if "Bone Break Classification" not in file_path:
            print("🩻 Gambar dari luar dataset — lakukan normalisasi otomatis...")
            img_rgb = normalize_external_xray(img_cv)
        else:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # 2️⃣ Resize & normalisasi
        img_resized = cv2.resize(img_rgb, (256, 256))
        img_array = np.expand_dims(img_resized / 255.0, axis=0)

        # 3️⃣ Prediksi
        preds = model.predict(img_array, verbose=0)[0]
        class_index = int(np.argmax(preds))
        predicted_class = class_names[class_index]
        confidence = float(preds[class_index]) * 100.0
        label_text = "Normal (Tidak Patah)" if predicted_class.lower() == "normal" else "Fracture (Patah Tulang)"

        # 4️⃣ Tampilkan hasil ke GUI
        img_disp = Image.open(file_path).convert("RGB").resize((400, 400))
        img_disp = ImageTk.PhotoImage(img_disp)
        img_label.config(image=img_disp, text="")
        img_label.image = img_disp

        result_label.config(
            text=f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}%\nLabel: {label_text}",
            fg="#2e7d32" if predicted_class.lower() == "normal" else "#d32f2f"
        )

        # 5️⃣ Buat confusion matrix dummy
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        cm[class_index, class_index] = 1
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=False, fmt="d", linewidths=0.5)
        plt.title(f"Confusion Matrix (Predicted: {predicted_class})")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        tmp_path = "confusion_matrix_temp.png"
        plt.savefig(tmp_path, bbox_inches="tight", dpi=150)
        plt.close()

        cm_img = Image.open(tmp_path).resize((600, 480))
        cm_img = ImageTk.PhotoImage(cm_img)
        cm_label.config(image=cm_img, text="")
        cm_label.image = cm_img

    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan saat prediksi:\n{str(e)}")

    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    try:
        img_cv = cv2.imread(file_path)
        if img_cv is None:
            raise ValueError("Gagal membaca gambar.")

        # Deteksi X-ray valid atau tidak
        is_valid_xray = is_xray_image(file_path)

        if not is_valid_xray:
            messagebox.showwarning("Gambar Tidak Valid", "Bukan gambar X-ray medis. Tidak dapat diproses.")
            img_disp = Image.open(file_path).convert("RGB").resize((400, 400))
            img_disp = ImageTk.PhotoImage(img_disp)
            img_label.config(image=img_disp, text="")
            img_label.image = img_disp
            result_label.config(text="⚠️ Bukan gambar X-ray medis.\nTidak dapat diproses.", fg="orange")
            cm_label.config(image="", text="Masukan gambar X-ray untuk melihat matrix")
            return

        # 1️⃣ Ubah ke grayscale → RGB
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # 2️⃣ Resize & normalisasi
        img_resized = cv2.resize(img_rgb, (256, 256))
        img_array = np.expand_dims(img_resized / 255.0, axis=0)

        # 3️⃣ Prediksi
        preds = model.predict(img_array, verbose=0)[0]
        class_index = int(np.argmax(preds))
        predicted_class = class_names[class_index]
        confidence = float(preds[class_index]) * 100.0
        label_text = "Normal (Tidak Patah)" if predicted_class.lower() == "normal" else "Fracture (Patah Tulang)"

        # 4️⃣ Tampilkan hasil
        img_disp = Image.open(file_path).convert("RGB").resize((400, 400))
        img_disp = ImageTk.PhotoImage(img_disp)
        img_label.config(image=img_disp, text="")
        img_label.image = img_disp

        result_label.config(
            text=f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}%\nLabel: {label_text}",
            fg="#2e7d32" if predicted_class.lower() == "normal" else "#d32f2f"
        )

        # 5️⃣ Confusion matrix dummy (1x1)
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        cm[class_index, class_index] = 1
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=False, fmt="d", linewidths=0.5)
        plt.title(f"Confusion Matrix (Predicted: {predicted_class})")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        tmp_path = "confusion_matrix_temp.png"
        plt.savefig(tmp_path, bbox_inches="tight", dpi=150)
        plt.close()

        cm_img = Image.open(tmp_path).resize((600, 480))
        cm_img = ImageTk.PhotoImage(cm_img)
        cm_label.config(image=cm_img, text="")
        cm_label.image = cm_img

    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan saat prediksi:\n{str(e)}")

# ======================================================
# 🎨 GUI LAYOUT
# ======================================================
root.deiconify()
root.title("🦴 Bone Fracture Classification (CNN)")
root.geometry("1400x850")
root.configure(bg="#f5f5f5")

# ==================== HEADER TITLE ====================
header = Frame(root, bg="#4CAF50", height=60)
header.pack(fill="x")

Label(
    header,
    text="🩻 Sistem Deteksi Jenis Patah Tulang Berdasarkan Citra X-Ray Menggunakan CNN",
    bg="#4CAF50",
    fg="white",
    font=("Arial", 18, "bold")
).pack(pady=10)

# ==================== MAIN CONTENT ====================
main_frame = Frame(root, bg="#f5f5f5")
main_frame.pack(fill="both", expand=True)

frame_left = Frame(main_frame, bg="#f5f5f5")
frame_left.grid(row=0, column=0, padx=40, pady=30, sticky="nsew")

Label(frame_left, text="📤 Masukan Gambar X-ray", bg="#f5f5f5", font=("Arial", 15, "bold")).pack(pady=10)
Button(frame_left, text="Masukan Gambar", command=upload_and_predict,
       bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
       padx=20, pady=8, cursor="hand2").pack(pady=15)

img_label = Label(frame_left, bg="#e0e0e0", text="Gambar Belum Dimasukan",
                  relief="sunken", bd=3, width=60, height=25)
img_label.pack(pady=10, expand=True, fill="both")

frame_right = Frame(main_frame, bg="#f8f9fa", padx=20, pady=20, relief="groove", bd=2)
frame_right.grid(row=0, column=1, sticky="nsew")

main_frame.rowconfigure(0, weight=1)
main_frame.columnconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=2)

Label(frame_right, text="🧠 Hasil Prediksi", font=("Arial", 14, "bold"), bg="#f8f9fa").pack(anchor="w")
result_label = Label(frame_right, text="", font=("Arial", 11), bg="#f8f9fa", justify="left")
result_label.pack(anchor="w", pady=(5, 15))

Label(frame_right, text="📊 Confusion Matrix", font=("Arial", 13, "bold"), bg="#f8f9fa").pack(anchor="w")
cm_label = Label(frame_right, bg="#f8f9fa", text="Masukan Gambar Untuk Melihat Matrix")
cm_label.pack(pady=10)

# ==================== METRIK MODEL ====================
try:
    with open(METRICS_JSON, "r") as f:
        metrics = json.load(f)

    Label(frame_right, text="📈 Model Evaluation Metrics",
          font=("Arial", 13, "bold"), bg="#f8f9fa").pack(anchor="w", pady=(15, 0))
    Label(frame_right, text=f"Akurasi       : {metrics['accuracy']:.4f}",
          bg="#f8f9fa", font=("Arial", 11)).pack(anchor="w")
    Label(frame_right, text=f"Presisi       : {metrics['precision']:.4f}",
          bg="#f8f9fa", font=("Arial", 11)).pack(anchor="w")
    Label(frame_right, text=f"Recall        : {metrics['recall']:.4f}",
          bg="#f8f9fa", font=("Arial", 11)).pack(anchor="w")
    Label(frame_right, text=f"Average Recall: {metrics['average_recall']:.4f}",
          bg="#f8f9fa", font=("Arial", 11)).pack(anchor="w")

except Exception as e:
    Label(frame_right, text="⚠️ Gagal memuat evaluation_metrics.json",
          bg="#f8f9fa", fg="red", font=("Arial", 10, "italic")).pack(anchor="w")
    print("Error loading metrics:", e)

root.mainloop()
