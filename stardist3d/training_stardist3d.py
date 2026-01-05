#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI for StarDist3D Training - 3D Nuclei Segmentation
===================================================

Created on Thu Jan 16 15:01:10 2025
@author: ws4
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from glob import glob
from tifffile import imread
from sklearn.model_selection import train_test_split
from stardist.models import Config3D, StarDist3D
from stardist import Rays_GoldenSpiral
from cc3d import connected_components
from preprocessing.image_preprocessing import preprocessing
from preprocessing.data_augmentation import aug_stardist
import h5py
import time
import pandas as pd
import tensorflow as tf
from csbdeep.utils.tf import limit_gpu_memory
import typer
import config

# Create typer app
app = typer.Typer(
    name="stardist3d-train",
    help="Train StarDist3D models for 3D nuclei segmentation",
    no_args_is_help=True,
)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ============================
# Funciones Auxiliares
# ============================


def normalize(img):
    """
    Normaliza una imagen entre 0 y 1.
    """
    img = (img - img.min()) / (img.max() - img.min())
    return img.astype(np.float16)


def verificar_archivos(path, tipo):
    archivos = sorted(glob(os.path.join(path, f"{tipo}/*.[tT][iI][fF]*")))
    if archivos:
        print(f"Se encontraron {len(archivos)} archivos en '{path}/{tipo}'.")
    else:
        raise FileNotFoundError(f"No se encontraron archivos en '{path}/{tipo}'.")
    return archivos


def load_images_labels(path, tag_images, tag_labels):
    images = verificar_archivos(path, tag_images)
    images = [normalize(imread(x)) for x in images]
    labels = verificar_archivos(path, tag_labels)
    labels = [connected_components(imread(x)) for x in labels]
    return images, labels


def load_from_hdf5(path):
    """
    Carga imágenes y etiquetas desde un archivo HDF5.
    """
    with h5py.File(path, "r") as hf:
        images = np.array(hf["images"])  # Cargar imágenes
        labels = np.array(hf["labels"])  # Cargar etiquetas
    return images, labels


import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from functools import lru_cache


class HDF5SingleItemGenerator(Sequence):
    """Generador sincronizado para imágenes y etiquetas en HDF5."""

    def __init__(self, hdf5_path, dataset_name="images"):
        self.hdf5_path = hdf5_path
        self.dataset_name = dataset_name

        with h5py.File(self.hdf5_path, "r") as hf:
            self.n_samples = hf[dataset_name].shape[0]  # Número total de muestras

    def __len__(self):
        return self.n_samples  # Cada elemento corresponde a un índice en el dataset

    def __getitem__(self, index):

        with h5py.File(self.hdf5_path, "r") as hf:
            item_data = hf[self.dataset_name][index]  # Cargar solo un ítem

        # Normalizar si es el dataset de imágenes
        if self.dataset_name == "images":
            item_data = item_data.astype(np.float32)
        elif self.dataset_name == "labels":
            item_data = connected_components(item_data)
        return item_data


# ============================
# CLI Command
# ============================
@app.command()
def train(
    train_path: str = typer.Argument(..., help="Path to training HDF5 file"),
    val_path: str = typer.Argument(..., help="Path to validation HDF5 file"),
    model_name: str = typer.Argument(..., help="Name for the trained model"),
    basedir: str = typer.Option(
        config.MODELS_ROOT, "--basedir", "-b", help="Base directory to save the model"
    ),
):
    """
    Train StarDist3D model using HDF5 datasets.

    Examples:
        stardist3d-train train data/train.hdf5 data/val.hdf5 --model-name my_model
        stardist3d-train train data/train.hdf5 data/val.hdf5 -m liver_nuclei -b ./models
    """

    # Validate inputs
    if not os.path.exists(train_path):
        typer.echo(f"❌ Training file not found: {train_path}", err=True)
        raise typer.Exit(1)
    if not os.path.exists(val_path):
        typer.echo(f"❌ Validation file not found: {val_path}", err=True)
        raise typer.Exit(1)

    # Show configuration
    typer.echo("🚀 StarDist3D Training:")
    typer.echo(f"   📁 Train: {train_path}")
    typer.echo(f"   📁 Val: {val_path}")
    typer.echo(f"   🏷️  Model: {model_name}")
    typer.echo(f"   📁 Output: {basedir}")
    typer.echo()

    print("Train path: ", train_path)
    print("Val path: ", val_path)

    X_train = HDF5SingleItemGenerator(train_path, dataset_name="images")
    y_train = HDF5SingleItemGenerator(train_path, dataset_name="labels")

    X_val = HDF5SingleItemGenerator(val_path, dataset_name="images")
    y_val = HDF5SingleItemGenerator(val_path, dataset_name="labels")

    conf = Config3D(
        rays=Rays_GoldenSpiral(96),
        use_gpu=True,
        n_channel_in=1,
        train_patch_size=(64, 96, 96),
        train_batch_size=2,
        train_epochs=200,
        train_learning_rate=1e-3,
        train_n_val_patches=1,
    )

    main(
        X_train,
        y_train,
        X_val,
        y_val,
        conf,
        basedir,
        model_name,
        augmenter=aug_stardist,
    )


# ============================
# Función Principal
# ============================
def main(X_train, y_train, X_val, y_val, conf, basedir, model_name, augmenter=None):
    # Crear el modelo
    model = StarDist3D(conf, name=model_name, basedir=basedir)

    # Entrenamiento del modelo
    print("Iniciando el entrenamiento del modelo...")
    # Registrar tiempo de inicio
    start_time = time.time()
    history = model.train(
        X_train, y_train, validation_data=(X_val, y_val), augmenter=augmenter
    )
    # Registrar tiempo de finalización
    end_time = time.time()
    print("Entrenamiento completado.")

    # Calcular tiempo total
    total_training_time = end_time - start_time
    print(f"Tiempo total de entrenamiento: {total_training_time:.2f} segundos.")

    # Guardar el tiempo en un archivo
    time_log_path = os.path.join(basedir, model_name, "training_time.txt")
    with open(time_log_path, "w") as f:
        f.write(f"Tiempo total de entrenamiento: {total_training_time:.2f} segundos.\n")

    # Guardar historial del modelo
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(
        os.path.join(basedir, model_name, "training_history.csv"), index=False
    )
    print("Historial de entrenamiento guardado.")


if __name__ == "__main__":
    typer.run(train)
