# 🩻 Fracture Detection System using Dual CNN Models with PyQt5 GUI

This project is a **deep learning-based desktop application** for detecting bone fractures from X-ray images using **two CNN models**.  
It provides a graphical interface built with **PyQt5**, enabling users to load images, perform automatic classification, and visualize confusion matrices dynamically.

---

## 🚀 Features

- 🧠 **Two-Stage CNN Detection**
  1. **Model 1:** Detects whether the input is a valid X-ray image.  
  2. **Model 2:** Classifies the type of bone fracture (or detects if it’s normal).  
- 🖼️ **Interactive PyQt5 GUI** for selecting, viewing, and classifying images.  
- 📊 **Dynamic metrics display** (precision, recall, F1-score, etc.) per image.  
- 🔥 **Top-5 prediction results** list for Model 2.  
- 🎨 **Automatic confusion matrix** generation and visualization for each prediction.  

---

## 🧰 Tech Stack

| Component | Description |
|------------|-------------|
| **Language** | Python 3.x |
| **GUI Framework** | PyQt5 |
| **Deep Learning** | TensorFlow / Keras |
| **Image Handling** | Pillow (PIL) |
| **Visualization** | Matplotlib, Seaborn |
| **Data Format** | `.keras`, `.json` |

---

## 📂 Project Structure

```
detection_fractureTkinter/
│
├── main_dual_cnn.py           # Main GUI file (PyQt5 interface)
├── model1_xray_filterCNN.keras # Model 1 - X-ray filter (binary classifier)
├── TfractureCNNRev_model.keras # Model 2 - Fracture type classifier
├── class_names_model1.json     # Class labels for Model 1
├── class_names_model2.json     # Class labels for Model 2
├── evaluation_metrics.json     # Model performance metrics
├── confusion_matrix_final.png  # Static confusion matrix example
├── Model1_CNNConv2D.ipynb      # Training notebook
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation
```

---

## 🧠 How It Works

### 🩺 **Model 1 — X-ray Filter (Binary CNN)**
This model detects whether the uploaded image is a real X-ray or a non-X-ray image.  
If the image is not an X-ray, the system will warn the user and stop further classification.

### 🦴 **Model 2 — Fracture Classification (Multi-class CNN)**
If the input is a valid X-ray, Model 2 classifies it into one of several fracture types, such as:

- Avulsion Fracture  
- Comminuted Fracture  
- Fracture Dislocation  
- Greenstick Fracture  
- Hairline Fracture  
- Impacted Fracture  
- Longitudinal Fracture  
- Normal  
- Oblique Fracture  
- Pathological Fracture  
- Spiral Fracture  

It also displays top-5 probabilities and dynamic confusion matrix visualizations.

---

## 🧪 Model Training

Training was conducted in `Model1_CNNConv2D.ipynb` using **Keras Sequential API**.

### 🔹 General Workflow
1. **Data Preparation**
   - X-ray dataset split into train, validation, and test sets.
   - Images normalized and resized to `(224, 224)` for Model 1 and `(256, 256)` for Model 2.

2. **Model Architecture**
   ```python
   model = Sequential([
       Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
       MaxPooling2D(2,2),
       Conv2D(64, (3,3), activation='relu'),
       MaxPooling2D(2,2),
       Flatten(),
       Dense(128, activation='relu'),
       Dense(num_classes, activation='softmax')
   ])
   ```

3. **Compilation & Training**
   ```python
   model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
   history = model.fit(train_data, validation_data=val_data, epochs=20)
   ```

4. **Model Export**
   ```python
   model.save('TfractureCNNRev_model.keras')
   ```

---

## 🖥️ Running the Application

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/meganfbr/detection_fractureTkinter.git
cd detection_fractureTkinter
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
python main_dual_cnn.py
```

A PyQt5 window will appear, allowing you to load images and view predictions interactively.

---

## ⚠️ Notes

- Files like `TfractureCNNRev_model.keras` and `model1_xray_filterCNN.keras` are **large (>50 MB)**.  
  Consider using [Git LFS](https://git-lfs.github.com/) for better handling.

- If you get errors related to **TensorFlow GPU**, you can disable GPU by adding:
  ```python
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  ```

---

## 👩‍💻 Author

**Megan Febriana**  
💼 [GitHub Profile](https://github.com/meganfbr)  
📧 _Add your contact info here if desired_

---

## 🏷️ License

This project is licensed under the **MIT License** — free to use and modify for educational or research purposes.

---

> 💡 *AI-powered bone fracture detection — smarter diagnostics through deep learning!*
