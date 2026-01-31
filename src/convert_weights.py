import tensorflow as tf
import h5py
import os

# 1) Construye el modelo EXACTO
from keras_models.attention_unet_3d import Attention_ResUNet_3D  # ajusta import
import config

SRC = "/mnt/WS2/Current Segovia Lab/Jorge Vergara/2026_09_01_3D-Nuclei-segmentation-main/models/resunet_combined_scaled_1000/modelSave.weights.h5"
DST = "/mnt/WS2/Current Segovia Lab/Jorge Vergara/2026_09_01_3D-Nuclei-segmentation-main/models/resunet_combined_scaled_1000/modelSave.fixed.weights.h5"

model = Attention_ResUNet_3D(
    config.INPUT_SHAPE, config.DROPOUT_RATE, config.BATCH_NORM, config.FILTER_NUM
)

def weight_basename(tf_weight_name: str, layer_name: str) -> str:
    # Ej: "conv3d/kernel:0" -> "kernel:0"
    # Ej: "batch_normalization/moving_mean:0" -> "moving_mean:0"
    if tf_weight_name.startswith(layer_name + "/"):
        return tf_weight_name[len(layer_name)+1:]
    return tf_weight_name.split("/")[-1]

loaded, skipped = 0, 0

with h5py.File(SRC, "r") as f:
    g = f["model_weights"]

    for layer in model.layers:
        if layer.name not in g:
            continue

        layer_group = g[layer.name]

        # arma lista de pesos en el orden que Keras espera para esa capa
        new_weights = []
        ok = True
        for w in layer.weights:
            key = weight_basename(w.name, layer.name)

            if key not in layer_group:
                ok = False
                break

            arr = np.array(layer_group[key])
            new_weights.append(arr)

        if ok and len(new_weights) > 0:
            try:
                layer.set_weights(new_weights)
                loaded += 1
            except Exception:
                skipped += 1

print(f"Layers loaded: {loaded}, skipped: {skipped}")

# guarda en formato weights-only estándar
model.save_weights(DST)
print("Saved converted weights to:", DST)