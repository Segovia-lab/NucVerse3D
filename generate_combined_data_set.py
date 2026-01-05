#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI for Dataset Splitter - 3D Nuclei Segmentation (Multi-dataset, Random Patches)
=================================================================================
- Input: a single root folder containing multiple dataset subfolders.
  Each dataset subfolder should have:
      dataset_X/
        +-- train/
        |   +-- <images_tag>/   (e.g., images/)
        |   +-- <labels_tag>/   (e.g., masks/)
        +-- val/   (optional; same structure)

- For each dataset subfolder:
  * Generate a fixed total number of random 3D patches (given by --patches-per-subfolder).
  * Split per-subfolder into train/val using test_size from config.py.
    - If val/ is missing, auto-split from train/ images.
  * Distribute the per-subfolder patch budget across TIFF images
    proportionally to their voxel counts to keep the dataset balanced.

- Output:
  * All training patches merged into a single HDF5:  data_train.hdf5
  * All validation patches merged into a single HDF5: data_val.hdf5
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import math
import typer
from glob import glob
import numpy as np
import h5py
from tqdm import tqdm
from tifffile import imread
from cc3d import connected_components
from skimage.transform import resize

from sklearn.model_selection import train_test_split  # kept for parity; not used now
from utils.image import pad_img, normalize, create_path
from preprocessing.image_preprocessing import preprocessing

import numpy as np
import tifffile as tiff
from skimage.measure import regionprops



import config

# -----------------------------------------------------------------------------
# Typer app
# -----------------------------------------------------------------------------
app = typer.Typer(
    name="dataset-splitter",
    help="Generate random training/validation patches from multiple 3D microscopy datasets",
    no_args_is_help=True,
)

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def verificar_archivos(path: str, tipo: str) -> List[str]:
    """Verifica si existen archivos TIFF en path/tipo y retorna la lista ordenada."""
    archivos = sorted(glob(os.path.join(path, f"{tipo}/*.[tT][iI][fF]*")))
    if archivos:
        print(f" {len(archivos)} archivos en '{path}/{tipo}'.")
    else:
        raise FileNotFoundError(f"No se encontraron archivos en '{path}/{tipo}'.")
    return archivos


def load_images_labels_old(path: str, images_tag: str, labels_tag: str):
    """	Carga y normaliza imagenes; convierte mascaras a instancias con CC3D."""

    images_files = verificar_archivos(path, images_tag)
    labels_files = verificar_archivos(path, labels_tag)

    images = [normalize(imread(f)) for f in images_files]
    labels = [connected_components(imread(f)) for f in labels_files]

    # Invert if "Mouse" in path
    toInvert = 	"Mouse"
    images_corrected = [
        np.amax(img) - img if toInvert in f else img
        for f, img in zip(images_files, images)
    ]
    return images_corrected, labels

def load_images_labels(path: str, images_tag: str, labels_tag: str, scale_images: bool):
    """	Carga y normaliza imagenes; convierte mascaras a instancias con CC3D."""

    print("Scaling images : ", + scale_images)

    images_files = verificar_archivos(path, images_tag)
    labels_files = verificar_archivos(path, labels_tag)

    images = [imread(f) for f in images_files]
    labels = [connected_components(imread(f)) for f in labels_files]

    # Invert if "Mouse" in path
    toInvert = 	"Mouse"
    images_corrected = [
        np.amax(img) - img if toInvert in f else img
        for f, img in zip(images_files, images)
    ]

	
    scaled_images, scaled_labels, scale_factors = [], [], []
    for img, lbl in zip(images_corrected, labels):
        if scale_images:
        	print("scaling ....")
        	img_scaled, lbl_scaled, sf = rescale_image_and_labels(img, lbl, target_radius=config.AVG_RADIUS)
        	scale_factors.append(sf)
        else:
        	img_scaled, lbl_scaled, sf = img, lbl, 1.0

        scaled_images.append(normalize(img_scaled))
        scaled_labels.append(lbl_scaled)

    return scaled_images, scaled_labels


def get_nucleus_radii(labeled_mask):
    """
    Calculate the radius of each labeled nucleus in a 2D or 3D labeled mask image.

    Parameters
    ----------
    image_path : iamge
        Labeled image.

    Returns
    -------
    radii : list of float
        List of radii for each nucleus in the image.
    avg_radius : float
        Average radius across all nuclei in the image (NaN if no nuclei found).
    """

    props = regionprops(labeled_mask)

    radii = []
    for region in props:
        if labeled_mask.ndim == 2:
            # 2D case: area = pi * r^2  -> r = sqrt(area/pi)
            r = np.sqrt(region.area / np.pi)
        elif labeled_mask.ndim == 3:
            # 3D case: volume = 4/3 * pi * r^3 -> r = (3V / (4pi))^(1/3)
            r = ((3 * region.area) / (4 * np.pi)) ** (1 / 3)
        else:
            raise ValueError("Mask must be 2D or 3D.")
        radii.append(r)

    avg_radius = np.mean(radii) if radii else np.nan
    return avg_radius


def rescale_image_and_labels(image, label, target_radius=config.AVG_RADIUS):
    """
    Rescale 3D image and label to make avg nucleus radius = target_radius.
    Uses skimage.transform.resize (linear for image, nearest for labels).
    """
    # Get scale factor from average radius
    avg_radius = get_nucleus_radii(label)
    scale_factor = target_radius / avg_radius

    # Compute new shape
    new_shape = tuple(int(s * scale_factor) for s in image.shape)

    # Resize image (linear interpolation + anti-aliasing)
    scaled_image = resize(
        image,
        new_shape,
        order=1,                # linear interpolation
        anti_aliasing=True,
        preserve_range=True
    ).astype(image.dtype)

    # Resize label (nearest neighbor, no anti-aliasing)
    scaled_labels = resize(
        label,
        new_shape,
        order=0,                # nearest neighbor
        anti_aliasing=False,
        preserve_range=True
    ).astype(label.dtype)
    # Compute new average radius (after scaling)
    new_avg_radius = get_nucleus_radii(scaled_labels)

    # Print summary
    print(
        f"Rescaling summary:\n"
        f"  Original shape: {image.shape}, New shape: {scaled_image.shape}\n"
        f"  Original avg radius: {avg_radius:.3f}, "
        f"New avg radius: {new_avg_radius:.3f}\n"
        f"  Applied scale factor: {scale_factor:.3f}\n"
    )
    return scaled_image, scaled_labels, scale_factor
# -----------------------------------------------------------------------------
# Random patching
# -----------------------------------------------------------------------------

def _pad_to_minimum(img: np.ndarray, patch_size: Tuple[int, int, int]) -> np.ndarray:
    """
    Ensure that the image has at least 1.5 x patch_size in each axis.
    Uses numpy.pad with 'reflect' mode when possible,
    and falls back to 'constant' (zero padding) if reflection is not valid.
    """
    #print("img to pad : ", img.shape)
    #print("patch_size : ", patch_size)

    pad_width = []
    for dim, p0 in zip(img.shape, patch_size):
        # Required minimum dimension: 1.25 × patch size
        p = int(np.ceil(1.25 * p0))
        if dim >= p:
            # No padding needed for this axis
            pad_width.append((0, 0))
        else:
            # Compute how much padding is required
            total_pad = p - dim
            left = total_pad // 2
            right = total_pad - left
            pad_width.append((left, right))

    # Try reflect mode first, but fall back to constant if it fails
    try:
        padded_img = np.pad(img, pad_width, mode="reflect")
    except ValueError:
        padded_img = np.pad(img, pad_width, mode="constant")

    #print("img padded : ", padded_img.shape)
    return padded_img

def _rand_start(max_start: int, rng: np.random.Generator) -> int:
    return int(rng.integers(0, max_start + 1)) if max_start > 0 else 0


def _sample_random_patches_from_one_image(
    img: np.ndarray,
    lab: np.ndarray,
    n_patches: int,
    patch_size: Tuple[int, int, int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae n parches aleatorios 3D de una sola imagen/etiqueta (ambas ya padded)."""
    z, y, x = img.shape
    pz, py, px = patch_size
    max_z = max(z - pz, 0)
    max_y = max(y - py, 0)
    max_x = max(x - px, 0)

    ipatches = []
    lpatches = []
    for _ in range(n_patches):
        sz = _rand_start(max_z, rng)
        sy = _rand_start(max_y, rng)
        sx = _rand_start(max_x, rng)
        ip = img[sz:sz+pz, sy:sy+py, sx:sx+px]
        lp = lab[sz:sz+pz, sy:sy+py, sx:sx+px]
        ipatches.append(ip)
        lpatches.append(lp)

    print(" - ", len(ipatches), " patches generated")
    return np.asarray(ipatches), np.asarray(lpatches)


def _distribute_counts_by_voxels(
    images: List[np.ndarray],
    total_patches: int
) -> List[int]:
    """Distribuye total_patches entre imagenes proporcionalmente al numero de voxeles."""
    if total_patches <= 0 or len(images) == 0:
        return [0] * len(images)

    vols = np.array([int(np.prod(img.shape)) for img in images], dtype=np.float64)
    if vols.sum() == 0:
        # fallback uniforme
        base = total_patches // len(images)
        counts = [base] * len(images)
        for i in range(total_patches - base * len(images)):
            counts[i] += 1
        return counts

    raw = vols / vols.sum() * total_patches
    floored = np.floor(raw).astype(int)
    remainder = total_patches - floored.sum()
    # Asignar los restantes a las imagenes con mayor parte fraccional
    frac_idx = np.argsort(-(raw - floored))
    for i in range(remainder):
        floored[frac_idx[i]] += 1
    return floored.tolist()


def _filter_by_signal_threshold(
    images_patches: np.ndarray,
    labels_patches: np.ndarray,
    max_percent: float,
    rng: np.random.Generator,
    desired: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filtra parches segun el % del maximo conteo de voxeles etiquetados (>0) observado.
    Mantiene los parches con suma_binaria > max * (max_percent/100).

    Si 'desired' se da y hay mas que 'desired', hace una seleccion aleatoria de 'desired'.
    """
    if labels_patches.size == 0:
        return images_patches, labels_patches

    bin_sums = np.array([(lp > 0).sum() for lp in labels_patches], dtype=np.int64)
    max_sum = int(bin_sums.max()) if bin_sums.size > 0 else 0
    threshold = max_sum * (max_percent / 100.0)

    keep = bin_sums > threshold
    I = images_patches[keep]
    L = labels_patches[keep]

    if desired is not None and len(I) > desired:
        idx = rng.choice(len(I), size=desired, replace=False)
        I = I[idx]
        L = L[idx]

    return I, L


def _build_patches_for_split(
    images: List[np.ndarray],
    labels: List[np.ndarray],
    patch_size: Tuple[int, int, int],
    total_desired: int,
    max_percent: float,
    rng: np.random.Generator,
    oversample_factor: float = 2.0,
    max_tries: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera 'total_desired' parches aleatorios (con filtro por senal) de un conjunto de imagenes.
    Oversamplea y filtra; reintenta aumentando el oversample si no alcanza la cantidad deseada.
    """
    if total_desired <= 0:
        return np.empty((0,) + patch_size, dtype=np.float32), np.empty((0,) + patch_size, dtype=np.uint16)

    # Pad to minimu 
    images_padded = [preprocessing(_pad_to_minimum(img, patch_size)) for img in images]
    labels_padded = [_pad_to_minimum(lab, patch_size) for lab in labels]
    
    desired = total_desired
    tries = 0
    current_factor = oversample_factor

    while tries < max_tries:
        per_image_counts = _distribute_counts_by_voxels(
            images_padded,
            max(1, int(math.ceil(desired * current_factor)))
        )

        ip_all = []
        lp_all = []
        
        for img, lab, nps in zip(images_padded, labels_padded, per_image_counts):
            if nps <= 0:
                continue
            ip, lp = _sample_random_patches_from_one_image(
                img=img, lab=lab, n_patches=nps, patch_size=patch_size, rng=rng
            )
            ip_all.append(ip)
            lp_all.append(lp)

        if len(ip_all) == 0:
            return np.empty((0,) + patch_size, dtype=np.float32), np.empty((0,) + patch_size, dtype=np.uint16)

        Ibig = np.concatenate(ip_all, axis=0).astype(np.float32, copy=False)
        Lbig = np.concatenate(lp_all, axis=0).astype(np.uint16, copy=False)

        # Filtro por senal (relativo al maximo observado)
        If, Lf = _filter_by_signal_threshold(Ibig, Lbig, max_percent=max_percent, rng=rng, desired=None)

        if len(If) >= desired:
            # Seleccion final aleatoria a la cantidad exacta deseada
            idx = rng.choice(len(If), size=desired, replace=False)
            return If[idx], Lf[idx]

        # Si no alcanza, incrementa el oversampling y reintenta
        tries += 1
        current_factor *= 2.0

    # ultimo retorno: lo que haya (mejor parcial que nada)
    return If, Lf


# -----------------------------------------------------------------------------
# HDF5 appendable writers
# -----------------------------------------------------------------------------
def _init_or_require_ds(hf: h5py.File, name: str, patch_size: Tuple[int, int, int], dtype) -> h5py.Dataset:
    if name in hf:
        ds = hf[name]
        # sanity: same last dims
        if tuple(ds.shape[1:]) != tuple(patch_size):
            raise ValueError(f"HDF5 dataset '{name}' has shape {ds.shape}, expected (_, {patch_size}).")
        return ds
    else:
        maxshape = (None,) + tuple(patch_size)
        chunks = (min(64,  max(1, maxshape[0] or 64)),) + tuple(patch_size)  # reasonable chunking
        return hf.create_dataset(name, shape=(0,) + tuple(patch_size), maxshape=maxshape, chunks=chunks, dtype=dtype)


def append_to_hdf5(path: str, images: np.ndarray, labels: np.ndarray):
    """Append (N, z, y, x) images/labels to HDF5 file at 'path' (creates file/datasets if needed)."""
    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels must have same first dimension")

    if images.size == 0:
        return

    patch_size = images.shape[1:]
    with h5py.File(path, "a") as hf:
        dsi = _init_or_require_ds(hf, "images", patch_size, np.float16)
        dsl = _init_or_require_ds(hf, "labels", patch_size, np.uint16)

        # convert dtype on write to save space
        I = images.astype(np.float16, copy=False)
        L = labels.astype(np.uint16, copy=False)

        n_old = dsi.shape[0]
        n_new = n_old + I.shape[0]
        dsi.resize((n_new,) + patch_size)
        dsl.resize((n_new,) + patch_size)
        dsi[n_old:n_new] = I
        dsl[n_old:n_new] = L


def reset_hdf5_files(output_dir: str):
    """Start fresh train/val HDF5 files."""
    create_path(output_dir)
    for fname in ("data_train.hdf5", "data_val.hdf5"):
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)



# -----------------------------------------------------------------------------
# Main multi-dataset routine
# -----------------------------------------------------------------------------
def process_all_subfolders(
    input_root: str,
    output_dir: str,
    patches_per_subfolder: int,
    images_tag: str,
    labels_tag: str,
    patch_size: Tuple[int, int, int],
    max_percent: float,
    test_size: float,
    seed: int = 42,
    scale_images: bool=True, 
):
    rng = np.random.default_rng(seed)

    # Prepare output files
    reset_hdf5_files(output_dir)
    train_out = os.path.join(output_dir, "data_train.hdf5")
    val_out   = os.path.join(output_dir, "data_val.hdf5")

    # Enumerate dataset subfolders
    subfolders = [p for p in sorted(Path(input_root).iterdir()) if p.is_dir()]
    if not subfolders:
        raise FileNotFoundError(f"No dataset subfolders found in: {input_root}")

    print("\n === Dataset discovery ===")
    for p in subfolders:
        print(f"  - {p.name}")

    print("\n=== Processing ===")
    for ds_path in subfolders:
        train_path = ds_path / "train"
        val_path   = ds_path / "val"

        if not train_path.exists():
            print(f"[SKIP] '{ds_path.name}' does not contain 'train/'.")
            continue

        # Train/Val targets per subfolder
        n_val   = int(round(patches_per_subfolder * test_size))
        n_train = int(patches_per_subfolder - n_val)

        # Load train images/labels
        print(f"\n[{ds_path.name}] Loading TRAIN images...")
        tr_images, tr_labels = load_images_labels(str(train_path), images_tag, labels_tag, scale_images)


        # If val exists, load and draw val patches from there; otherwise, auto-split from train
        if val_path.exists():
            print(f"[{ds_path.name}] Loading VAL images...")
            va_images, va_labels = load_images_labels(str(val_path), images_tag, labels_tag, scale_images)

            # Build patches
            	
            train_I, train_L = _build_patches_for_split(
                tr_images, tr_labels, patch_size, n_train, max_percent, rng
            )

            val_I, val_L = _build_patches_for_split(
                va_images, va_labels, patch_size, n_val, max_percent, rng
            )
            val_source = "Validation folder"
        else:
            # Auto-split: both train and val patches come from train images
            print(f"[{ds_path.name}] No 'val/' found . Auto-split from training.")
            train_I, train_L = _build_patches_for_split(
                tr_images, tr_labels, patch_size, n_train, max_percent, rng
            )
            val_I, val_L = _build_patches_for_split(
                tr_images, tr_labels, patch_size, n_val, max_percent, rng
            )
            val_source = "Auto-split from training"

        # Append to global HDF5s
        append_to_hdf5(train_out, train_I, train_L)
        append_to_hdf5(val_out, val_I, val_L)

        print(f"[{ds_path.name}] -> train: {len(train_I)}  | val: {len(val_I)}  ({val_source})")

    print("\n=== Done ===")
    print(f"Training data merged at: {train_out}")
    print(f"Validation data merged at: {val_out}")


# -----------------------------------------------------------------------------
# Typer commands
# -----------------------------------------------------------------------------
@app.command()
def split_dataset(
    input_root: str = typer.Argument(
        ...,
        help="Root folder containing dataset subfolders. Each subfolder should have 'train/' and optional 'val/'."
    ),
    dataset_name: str = typer.Option(
        ...,
        "--dataset-name", "-d",
        help="Name for the merged dataset directory (created in data/processed/).",
    ),
    patches_per_subfolder: int = typer.Option(
        1000,
        "--patches-per-subfolder", "-n",
        help="Fixed number of random patches to generate per dataset subfolder (train+val combined).",
    ),
    images_tag: str = typer.Option(
        "images", "--images-tag", "-i",
        help="Name of the images folder inside train/ and val/."
    ),
    labels_tag: str = typer.Option(
        "masks", "--labels-tag", "-l",
        help="Name of the labels/masks folder inside train/ and val/."
    ),
    seed: int = typer.Option(
        42, "--seed",
        help="Random seed for reproducibility."
    ),
    scale_images: bool = typer.Option(
        True,
        "--scale/--no-scale",
        help="Scale images to radius of liver nuclei.",
    )   

):
    """
    Process ALL dataset subfolders under INPUT_ROOT:

    - For each dataset subfolder:
      * Generate <patches_per_subfolder> random patches (3D) in total.
      * Split per-subfolder into train/val by config.TEST_SIZE.
      * If val/ is absent, validation patches are auto-sampled from train/.

    - Merge all subfolders' patches into:
        processed/<dataset_name>/data_train.hdf5
        processed/<dataset_name>/data_val.hdf5
    """
    if not os.path.exists(input_root):
        typer.echo(f"Input root does not exist: {input_root}", err=True)
        raise typer.Exit(1)

    output_dir = os.path.join(str(config.PROCESSED_DATA_ROOT), dataset_name)

    typer.echo("=== Dataset Splitter Configuration ===")
    typer.echo(f"Input root:        {input_root}")
    typer.echo(f"Output directory:  {output_dir}")
    typer.echo(f"Images folder:     {images_tag}")
    typer.echo(f"Labels folder:     {labels_tag}")
    typer.echo()
    typer.echo("Using config defaults:")
    typer.echo(f"Patch size:        {config.PATCH_SIZE}")
    typer.echo(f"Test size:         {config.TEST_SIZE}")
    typer.echo(f"Max signal %:      {config.MAX_PERCENT_SIGNAL}")
    typer.echo(f"Patches/subfolder: {patches_per_subfolder}")
    typer.echo(f"Scaling images:    {scale_images}")
    typer.echo()

    try:
        process_all_subfolders(
            input_root=input_root,
            output_dir=output_dir,
            patches_per_subfolder=patches_per_subfolder,
            images_tag=images_tag,
            labels_tag=labels_tag,
            patch_size=config.PATCH_SIZE,
            max_percent=config.MAX_PERCENT_SIGNAL,
            test_size=config.TEST_SIZE,
            seed=seed,
            scale_images=scale_images,
        )

        typer.echo(" Merged dataset processing completed successfully!")
        typer.echo(f"  Training HDF5:   {output_dir}/data_train.hdf5")
        typer.echo(f"  Validation HDF5: {output_dir}/data_val.hdf5")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def info():
    """Show information about current configuration defaults."""
    typer.echo("Current Configuration Defaults:")
    typer.echo(f"  Patch size:           {config.PATCH_SIZE}")
    typer.echo(f"  Step size (training): {config.STEP_SIZE_TRAINING}")
    typer.echo(f"  Max signal %:         {config.MAX_PERCENT_SIGNAL}")
    typer.echo(f"  Test size:            {config.TEST_SIZE}")
    typer.echo(f"  Project root:         {config.PROJECT_ROOT}")
    typer.echo(f"  Raw data root:        {config.RAW_DATA_ROOT}")
    typer.echo(f"  Processed data root:  {config.PROCESSED_DATA_ROOT}")
    typer.echo(f"  Scaling images:       {scale_images}")
    typer.echo()
    typer.echo("Usage: Results are saved to processed_data_root/dataset_name/")


if __name__ == "__main__":
    app()

#Use example:
#python src/generate_combined_data_set.py split-dataset data/interim/ -d all_data_1000 -n 1000 --seed 42

#nohup python src/generate_combined_data_set.py split-dataset data/interim/  -d combined_1000        -n 1000 --seed 42 --no-scale  > output_data_combined_1000.log  2>&1 &
#nohup python src/generate_combined_data_set.py split-dataset data/interim/  -d combined_scaled_1000 -n 1000 --seed 42 --scale     > output_data_combined_scaled_1000.log  2>&1 &


