import h5py
import tensorflow as tf
from tensorflow.keras.models import model_from_json

old_path = "model1_xray_filter.h5"
new_path = "model1_xray_filter_fixed.h5"

print("🔍 Membuka file H5 lama secara manual...")
with h5py.File(old_path, "r") as f:
    # ambil arsitektur model (biasanya disimpan dalam format JSON di HDF5)
    model_config = f.attrs.get("model_config")
    if model_config is None:
        raise ValueError("Tidak ditemukan konfigurasi model di file H5 ini.")
    model_config = model_config.decode("utf-8")

# perbaiki field lama
model_config = model_config.replace('"batch_shape":', '"batch_input_shape":')

print("🛠️ Menghapus argumen 'batch_shape' lawas...")
model = model_from_json(model_config)

print("✅ Struktur model berhasil dibaca ulang!")

# load weights
print("📦 Memuat bobot model...")
with h5py.File(old_path, "r") as f:
    weights_group = f["model_weights"]
    weight_names = list(weights_group.keys())
    for layer in model.layers:
        if layer.name in weights_group:
            try:
                layer_weights = [
                    weights_group[layer.name][wn][()]
                    for wn in weights_group[layer.name]
                ]
                layer.set_weights(layer_weights)
            except Exception as e:
                print(f"⚠️ Gagal set weights untuk {layer.name}: {e}")

# simpan ulang model dengan format baru
print("💾 Menyimpan model baru...")
model.save(new_path)
print(f"✅ Konversi selesai! Model disimpan sebagai: {new_path}")
