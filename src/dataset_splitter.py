#!/usr/bin/env python3

"""
CLI for Dataset Splitter - 3D Nuclei Segmentation
=================================================

This module provides a command-line interface for splitting datasets into training and validation sets.
It generates patches from 3D images and saves them in HDF5 format for efficient loading during training.
"""
import sys
import os
from pathlib import Path
from typing import Optional, Tuple
import typer
from utils.image import pad_img, normalize, create_path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from patchify import patchify
from tifffile import imread
from glob import glob
import numpy as np
from preprocessing.image_preprocessing import preprocessing, deconvolution_task, preprocessing_simple
from tqdm import tqdm
import h5py
from cc3d import connected_components
import config

# Create typer app
app = typer.Typer(
    name="dataset-splitter",
    help="Generate training and validation patches from 3D microscopy images",
    no_args_is_help=True,
)


def generate_patches(images, step, patch_size):
    """
    Genera parches de imágenes con un tamaño y un paso específicos.
    """
    patches = [
        patchify(x, step=step, patch_size=patch_size).reshape((-1,) + patch_size)
        for x in images
    ]
    patches = np.concatenate(patches, axis=0)
    return patches


def save_to_hdf5(path, images, labels):
    """
    Guarda imágenes y etiquetas en un archivo HDF5.
    """
    with h5py.File(path, "w") as hf:
        hf.create_dataset("images", data=images, dtype=np.float16)
        hf.create_dataset("labels", data=labels, dtype=np.uint16)


def verificar_archivos(path, tipo):
    """
    Verifica si existen archivos en una ruta específica.

    Parameters:
        path (str): Ruta base donde se buscarán los archivos.
        tipo (str): Tipo de archivos a buscar (images o labels).

    Raises:
        FileNotFoundError: Si no se encuentran archivos en la ruta especificada.
    """
    archivos = sorted(glob(os.path.join(path, f"{tipo}/*.[tT][iI][fF]*")))
    if archivos:
        print(f"Se encontraron {len(archivos)} archivos en '{path}/{tipo}'.")
    else:
        raise FileNotFoundError(f"No se encontraron archivos en '{path}/{tipo}'.")
    return archivos


def load_images_labels(path, images_tag, labels_tag):
    """
    Carga las imágenes y etiquetas en un arreglo de numpy.

    Parameters:
        path (str): Ruta base donde se encuentran las imágenes y etiquetas.
    """
    images = verificar_archivos(path, images_tag)
    images = [normalize(imread(x)) for x in images]
    labels = verificar_archivos(path, labels_tag)
    labels = [connected_components(imread(x)) for x in labels]
    return images, labels


def filter_patches(images_patches, labels_patches, max_percent):
    """
    Filtra parches según un porcentaje máximo de señal en las etiquetas.

    Parameters:
        images_patches (np.ndarray): Parches de imágenes (N, z, y, x).
        labels_patches (np.ndarray): Parches de etiquetas (N, z, y, x).
        max_percent (float): Porcentaje máximo del valor máximo de señal para filtrar.

    Returns:
        tuple: Parches de imágenes y etiquetas filtrados.
    """
    # Calcular la suma binaria de cada parche de etiquetas
    binary_sum_labels = np.array([(x > 0).sum() for x in labels_patches])

    # Determinar el umbral de filtrado
    threshold = binary_sum_labels.max() * max_percent / 100

    # Filtrar los parches de imágenes y etiquetas
    filtered_images = images_patches[binary_sum_labels > threshold]
    filtered_labels = labels_patches[binary_sum_labels > threshold]

    return filtered_images, filtered_labels


def generate_training_val_set(
    train_path,
    val_path=None,
    images_tag="images",
    labels_tag="masks",
    dir_output="./training_set",
    patch_size=config.PATCH_SIZE,
    step=config.STEP_SIZE_TRAINING,
    max_percent=config.MAX_PERCENT_SIGNAL,
    test_size=config.TEST_SIZE,
    preprocessing_enabled=config.PREPROCESSING_ENABLED,
):

    # Cargar las imágenes y etiquetas de entrenamiento y validación
    train_images, train_labels = load_images_labels(train_path, images_tag, labels_tag)

    # Preprocesar las imágenes
    if preprocessing_enabled:
        train_preprocess = [preprocessing(x) for x in tqdm(train_images)]
    else:
        train_preprocess = [preprocessing_simple(x) for x in tqdm(train_images)]

    # Agregar padding y generar parches
    images_patches_train = generate_patches(
        [pad_img(x, patch_size, step, "constant") for x in train_preprocess],
        step,
        patch_size,
    )

    labels_patches_train = generate_patches(
        [pad_img(x, patch_size, step, "constant") for x in train_labels],
        step,
        patch_size,
    )

    # Filtrar parches con un porcentaje máximo de señal en las etiquetas
    images_patches_train_filt, labels_patches_train_filt = filter_patches(
        images_patches_train, labels_patches_train, max_percent
    )

    if not val_path:
        # Dividir en conjuntos de entrenamiento y validación
        (
            images_patches_train_filt,
            images_patches_val_filt,
            labels_patches_train_filt,
            labels_patches_val_filt,
        ) = train_test_split(
            images_patches_train_filt,
            labels_patches_train_filt,
            test_size=test_size,
            random_state=42,
        )
    else:
        val_images, val_labels = load_images_labels(val_path, images_tag, labels_tag)
        if preprocessing_enabled:
            val_preprocess = [preprocessing(x) for x in tqdm(val_images)]
        else:
            val_preprocess = [preprocessing_simple(x) for x in tqdm(val_images)]


        images_patches_val = generate_patches(
            [pad_img(x, patch_size, step, "constant") for x in val_preprocess],
            patch_size,
            patch_size,
        )
        labels_patches_val = generate_patches(
            [pad_img(x, patch_size, step, "constant") for x in val_labels],
            patch_size,
            patch_size,
        )
        # Filtrar parches con un porcentaje máximo de señal en las etiquetas
        images_patches_val_filt, labels_patches_val_filt = filter_patches(
            images_patches_val, labels_patches_val, max_percent
        )

    # Guardar datos en HDF5
    create_path(dir_output)
    save_to_hdf5(
        os.path.join(dir_output, "data_train.hdf5"),
        images_patches_train_filt,
        labels_patches_train_filt,
    )
    save_to_hdf5(
        os.path.join(dir_output, "data_val.hdf5"),
        images_patches_val_filt,
        labels_patches_val_filt,
    )


@app.command()
def split_dataset(
    train_path: str = typer.Argument(
        ...,
        help="Path to training dataset directory containing 'images' and 'labels' folders",
    ),
    dataset_name: str = typer.Option(
        ...,
        "--dataset-name",
        "-d",
        help="Name for the dataset directory (will be created in data/processed/)",
    ),
    val_path: Optional[str] = typer.Option(
        None,
        "--val-path",
        "-v",
        help="Path to validation dataset directory (if not provided, will split from training data)",
    ),
    images_tag: str = typer.Option(
        "images",
        "--images-tag",
        "-i",
        help="Name of the images folder within the dataset directory",
    ),
    labels_tag: str = typer.Option(
        "masks",
        "--labels-tag",
        "-l",
        help="Name of the labels/masks folder within the dataset directory",
    ),
):
    """
    Split a 3D microscopy dataset into training and validation patches.

    This command processes 3D images using the configured parameters from config.py.
    The processing pipeline:
    1. Loads images and labels from the specified directories
    2. Applies preprocessing (deconvolution, normalization)
    3. Generates overlapping patches using configured patch size and step size
    4. Filters patches based on configured signal content threshold
    5. Saves the results in HDF5 format for efficient training

    Examples:
        # Basic usage with automatic train/val split
        dataset-splitter /path/to/dataset --dataset-name my_experiment

        # With custom validation set
        dataset-splitter /path/to/train --val-path /path/to/val --dataset-name liver_nuclei

        # With custom folder names
        dataset-splitter /path/to/dataset --dataset-name my_data --images-tag raw --labels-tag segmentation
    """

    # Validate inputs
    if not os.path.exists(train_path):
        typer.echo(f"❌ Training path does not exist: {train_path}", err=True)
        raise typer.Exit(1)

    # Construct output directory path
    output_dir = os.path.join(str(config.PROCESSED_DATA_ROOT), dataset_name)

    # Display configuration
    typer.echo("🔧 Dataset Splitter Configuration:")
    typer.echo(f"   📁 Training path: {train_path}")
    typer.echo(f"   📁 Validation path: {val_path or 'Auto-split from training'}")
    typer.echo(f"   📁 Output directory: {output_dir}")
    typer.echo(f"   🏷️  Images folder: {images_tag}")
    typer.echo(f"   🏷️  Labels folder: {labels_tag}")
    typer.echo()
    typer.echo("📐 Using configured processing parameters:")
    typer.echo(f"   📏 Patch size: {config.PATCH_SIZE}")
    typer.echo(f"   👣 Step size: {config.STEP_SIZE_TRAINING}")
    typer.echo(f"   🎯 Max signal %: {config.MAX_PERCENT_SIGNAL}")
    typer.echo(f"   🔀 Test size: {config.TEST_SIZE}")
    typer.echo()

    try:
        # Call the main processing function with config defaults
        generate_training_val_set(
            train_path=train_path,
            val_path=val_path,
            images_tag=images_tag,
            labels_tag=labels_tag,
            dir_output=output_dir,
            patch_size=config.PATCH_SIZE,
            step=config.STEP_SIZE_TRAINING,
            max_percent=config.MAX_PERCENT_SIGNAL,
            test_size=config.TEST_SIZE,
            preprocessing_enabled=config.PREPROCESSING_ENABLED
        )

        typer.echo(f"✅ Dataset processing completed successfully!")
        typer.echo(f"   📊 Training data saved to: {output_dir}/data_train.hdf5")
        typer.echo(f"   📊 Validation data saved to: {output_dir}/data_val.hdf5")

    except Exception as e:
        typer.echo(f"❌ Error processing dataset: {str(e)}", err=True)
        raise typer.Exit(1)


@app.command()
def info():
    """Show information about current configuration defaults."""
    typer.echo("📋 Current Configuration Defaults:")
    typer.echo(f"   📐 Patch size: {config.PATCH_SIZE}")
    typer.echo(f"   👣 Step size (training): {config.STEP_SIZE_TRAINING}")
    typer.echo(f"   🎯 Max signal %: {config.MAX_PERCENT_SIGNAL}")
    typer.echo(f"   🔀 Test size: {config.TEST_SIZE}")
    typer.echo(f"   🏠 Project root: {config.PROJECT_ROOT}")
    typer.echo(f"   📁 Raw data root: {config.RAW_DATA_ROOT}")
    typer.echo(f"   📁 Processed data root: {config.PROCESSED_DATA_ROOT}")
    typer.echo()
    typer.echo("💡 Usage: Datasets will be saved to processed_data_root/dataset_name/")


if __name__ == "__main__":
    typer.run(split_dataset)
