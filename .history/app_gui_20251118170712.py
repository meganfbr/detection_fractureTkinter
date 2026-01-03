"""Load H5 model with compatibility fallback."""
```)  
sampai sebelum `class XrayClassifierApp`, jadi tinggal **satu fungsi** yang berisi blok dengan `safe_mode=False`.

hasil akhirnya seperti ini 👇

```python
def safe_load_model_with_compat(path):
    """Load H5 model with broader compatibility for TF version mismatch."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        return load_model(path, compile=False)
    except Exception as e1:
        try:
            from tensorflow.keras.layers import InputLayer, BatchNormalization, Conv2D, Dense
            # tambahkan custom_objects & skip unknown keys
            return load_model(
                path,
                compile=False,
                custom_objects={
                    "InputLayer": InputLayer,
                    "BatchNormalization": BatchNormalization,
                    "Conv2D": Conv2D,
                    "Dense": Dense,
                },
                safe_mode=False  # penting! biar batch_shape diabaikan
            )
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load model '{path}'.\nFirst error: {e1}\nSecond attempt: {e2}"
            )
