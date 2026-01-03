import tensorflow as tf
from tensorflow.keras.models import load_model, save_model

old_path = "model1_xray_filter.h5"
new_path = "model1_xray_filter_fixed.h5"

print("Loading old model...")
model = tf.keras.models.load_model(old_path, compile=False)
print("✅ Loaded successfully!")

print("Re-saving model in new format...")
tf.keras.models.save_model(model, new_path, include_optimizer=False)
print(f"✅ Model converted and saved as: {new_path}")
