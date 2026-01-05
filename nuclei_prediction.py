#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:19:56 2024

@author: ws2
"""

import os
import tensorflow as tf
import config
from pathlib import Path
from skimage.transform import resize

os.environ["CUDA_VISIBLE_DEVICES"]  = "1"  # Only the second GPU will be visible


gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, config.MEMORY_GROWTH_ENABLED)


import numpy as np
from tifffile import imread, imwrite
from utils.image import create_path, normalize
from preprocessing.image_preprocessing import preprocessing, aligned_image_simple, preprocessing_simple
from utils.image import pad_img, linear_interp_vol, unpad_img
from patchify import patchify
import gc
import cc3d
from scipy import ndimage as ndi
import scipy.spatial
from skimage.morphology import remove_small_objects
import typer

from generate_combined_data_set import rescale_image_and_labels, get_nucleus_radii

app = typer.Typer(
    name="nuclei-prediction",
    help="Prediction Attention ResUNet 3D models for 3D nuclei segmentation",
    no_args_is_help=True,
)


def prediction(
    img,
    model,
    dir_prediction=None,
    name=None,
    batch_size=config.PREDICTION_BATCH_SIZE,
    num_div=config.PREDICTION_NUM_DIV,
    patch_size=config.PATCH_SIZE,
    step=config.STEP_SIZE_PREDICTION,
):
    # Patches prediction
    from patchify import patchify
    import gc

    from utils.image import pad_img
    img_val_pad = pad_img(img, patch_size, step, "reflect")
    img_patchy_val = patchify(img_val_pad, patch_size, step=step)
    img_patches_reshaped = img_patchy_val.reshape((-1,) + patch_size)
    print("img_patches_reshaped: ", len(img_patches_reshaped))	

    img_val_pad_shape = img_val_pad.shape
    img_patchy_val_shape = img_patchy_val.shape

    del img_val_pad, img_patchy_val
    gc.collect()

    # Prediction
    from tensorflow.keras.backend import clear_session
    from tqdm import tqdm

    y_pred_bin = np.zeros((img_patches_reshaped.shape) + (2,), dtype=np.float16)
    y_pred_map = np.zeros((img_patches_reshaped.shape) + (3,), dtype=np.float16)

    n = len(img_patches_reshaped)
    if n == 0:
        raise ValueError("No patches to predict.") 

    img_patches_reshaped = [i for i in img_patches_reshaped]
    tiles = max(1,len(img_patches_reshaped) // num_div)

    print("tiles:", tiles)

    for j in tqdm(range(0, len(img_patches_reshaped), tiles)):
        curr_block = np.array(img_patches_reshaped[:tiles])
        print(
            f" Processed elements from {j} to {(j + len(curr_block))}. Remaining elements: {len(img_patches_reshaped)}"
        )
        y_pred_bin_batch, y_pred_map_batch = model.predict(
            curr_block, batch_size=batch_size, verbose=0
        )
        y_pred_bin[j : j + len(curr_block)] = y_pred_bin_batch.astype("float16")
        y_pred_map[j : j + len(curr_block)] = y_pred_map_batch.astype("float16")
        del img_patches_reshaped[:tiles]
        clear_session()
        gc.collect()

    del img_patches_reshaped
    gc.collect()
    # Reshape
    y_pred_bin = np.array(y_pred_bin).reshape(img_patchy_val_shape + (2,))
    y_pred_map = np.array(y_pred_map).reshape(img_patchy_val_shape + (3,))

    # Linear interpolation
    from utils.image import linear_interp_vol

    y_pred_bin = linear_interp_vol(y_pred_bin, img_val_pad_shape, patch_size, step, 2)
    y_pred_map = linear_interp_vol(y_pred_map, img_val_pad_shape, patch_size, step, 3)

    # Argmax
    y_pred_bin = np.argmax(y_pred_bin, axis=-1)

    # Unpad
    from utils.image import unpad_img

    y_pred_bin = unpad_img(y_pred_bin, img.shape)
    y_pred_map = unpad_img(y_pred_map, img.shape + (3,))

    
#    imwrite(os.path.join(dir_prediction, f"{name}_grad.tif"), y_pred_map.astype("float16"))
    gc.collect()
    return y_pred_bin, y_pred_map


from scipy import ndimage as ndi
from tqdm import tqdm


def grad_point(gradient_vector, point):
    # Correctly reshape the coordinate array
    coords = point
    # Separate the gradient components
    grad_z_component = gradient_vector[..., 0]
    grad_y_component = gradient_vector[..., 1]
    grad_x_component = gradient_vector[..., 2]
    # Map the coordinates for each component
    grad_z = ndi.map_coordinates(grad_z_component, coords, order=1)
    grad_y = ndi.map_coordinates(grad_y_component, coords, order=1)
    grad_x = ndi.map_coordinates(grad_x_component, coords, order=1)
    return grad_z, grad_y, grad_x


def gradient_descent_momentum(
    gradient,
    start,
    learn_rate=config.GRADIENT_DESCENT_LEARNING_RATE,
    n_iter=config.GRADIENT_DESCENT_N_ITER,
    tolerance=config.GRADIENT_DESCENT_TOLERANCE,
    momentum=config.GRADIENT_DESCENT_MOMENTUM,
):
    vector = np.array(start, dtype=gradient.dtype)
    history = []
    diff = 0
    for _ in tqdm(range(n_iter)):
        grad_z, grad_y, grad_x = grad_point(gradient, vector)
        diff = learn_rate * np.array([grad_z, grad_y, grad_x]) + momentum * diff
        vector = vector + diff
        history.append(vector.astype("int16"))
    return vector.astype("int16"), history


def filt_history(history, h, w, d):
    logical_and = np.logical_and.reduce(
        (
            history[0] >= 0,
            history[0] <= h - 1,
            history[1] >= 0,
            history[1] <= w - 1,
            history[2] >= 0,
            history[2] <= d - 1,
        )
    )
    history_filt = history[:, logical_and]
    return history_filt, logical_and


def centroid_estimation(vol_seg, gradient_vector, dir_prediction):
    gradient_vector = gradient_vector.astype(np.float64)
    points = np.array(np.where(vol_seg > 0)).astype("int16")

    # Descenso de gradiente
    gd, history = gradient_descent_momentum(gradient_vector, points, n_iter=config.GRADIENT_DESCENT_N_ITER)

    # FIltro de limites
    h, w, d = vol_seg.shape

    zeros_ = np.zeros_like(vol_seg, dtype=np.int32)
    history_filter, and_filt = filt_history(np.floor(history[-1]), h, w, d)

    # Generacion de centroides
    history_filter_points, history_filter_counts = np.unique(
        history_filter, axis=1, return_counts=True
    )
    a, b, c = history_filter_points.astype(np.int32)
    zeros_[a, b, c] += history_filter_counts.astype(np.int32)

#    imwrite(os.path.join(dir_prediction, f'{dir_prediction.split("/")[-1]}_zeros.tif'), zeros_)

    # Filtro gaussiano y threshold de centroides
    gaus_zeros = ndi.gaussian_filter(zeros_, config.CENTROID_GAUSSIAN_SIGMA)
#    imwrite(os.path.join(dir_prediction, f'{dir_prediction.split("/")[-1]}_gauss_zeros.tif'), gaus_zeros)

    # th = gaus_zeros[gaus_zeros.astype("int") != 0].mean() + 0*gaus_zeros[gaus_zeros.astype("int") != 0].std()
    th = np.mean(gaus_zeros) + config.CENTROID_THRESHOLD_STD_MULTIPLIER * np.std(gaus_zeros)
    # Gerneracion de semillas
    import cc3d

    seeds = gaus_zeros > th
    seeds = cc3d.connected_components(seeds)
#    imwrite(os.path.join(dir_prediction, f'{dir_prediction.split("/")[-1]}_seeds.tif'), seeds)
    return seeds, points[:, and_filt], history_filter


def calculate_labels(seeds, final_points, radius=config.LABEL_RADIUS):
    import scipy.spatial
    import cc3d

    seeds_labeled = cc3d.connected_components(seeds)
    # Convertir seeds_labeled en una lista de coordenadas con sus etiquetas
    indices = np.argwhere(seeds_labeled > 0)
    labels = seeds_labeled[seeds_labeled > 0]

    # Crear un KDTree para las coordenadas de los componentes conectados
    tree = scipy.spatial.KDTree(indices)

    # Convertir history_filter a una forma adecuada para la consulta
    final_points = final_points.T

    # Encontrar los puntos en history_filter que están cerca de algún componente en seeds
    distances, nearest_indices = tree.query(final_points, distance_upper_bound=radius)

    # Crear una máscara para los puntos dentro del radio de proximidad
    mask = distances < radius

    # Asignar valores de etiquetas a los puntos de history_filter dentro del radio de proximidad
    labels_id = np.zeros(final_points.shape[0], dtype=int)
    labels_id[mask] = labels[nearest_indices[mask]]

    return labels_id


from skimage.morphology import remove_small_objects


def nuclei_segmentation(
    vol_seg,
    initial_points,
    labels_id,
    dir_prediction,
    name,
    original_shape,
    vox_threhsold=config.VOX_THRESHOLD,
):
    a, b, c = initial_points
    labels = np.zeros_like(vol_seg, dtype=np.int16)
    labels[a, b, c] = labels_id
    labels = remove_small_objects(labels, min_size=20)

    if labels.shape != original_shape: 
    	labels = rescale_labels(labels, original_shape)  
    	vol_seg = rescale_labels(vol_seg, original_shape)   

    imwrite(os.path.join(dir_prediction, f"{name}_labels.tif"), labels.astype(np.uint16))
    imwrite(os.path.join(dir_prediction, f"{name}_binary.tif"), vol_seg.astype("uint16"))
    return labels

def rescale_image(image, avg_radius, target_radius=config.AVG_RADIUS):
    """
    Rescale 3D image  to make avg nucleus radius = target_radius.
    Uses skimage.transform.resize (linear for image).
    """
    # Get scale factor from average radius
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

    
    # Print summary
    print(
        f"Rescaling summary:\n"
        f"  Original shape: {image.shape}, New shape: {scaled_image.shape}\n"
    )
    return scaled_image

def rescale_labels(label, new_shape):
    """
    Rescale 3D label to origibal size.
    Uses skimage.transform.resize (nearest for labels).
    """

    # Resize label (nearest neighbor, no anti-aliasing)
    scaled_labels = resize(
        label,
        new_shape,
        order=0,                # nearest neighbor
        anti_aliasing=False,
        preserve_range=True
    ).astype(label.dtype)

    
    # Print summary
    print(
        f"Rescaling summary:\n"
        f"  Original shape: {label.shape}, New shape: {scaled_labels.shape}\n"
    )
    return scaled_labels


# %%
@app.command()
def main(
    model_name: str = typer.Argument(..., help="Name of the model"),
    image_dir: str = typer.Argument(..., help="Path to image dir"),
    avg_radius: float = typer.Option(
        config.AVG_RADIUS, "--avg_radius", "-r", help="Average radius for nuclei in input images"
    ),
    output_dir: str = typer.Option(
        config.RESULTS_ROOT, "--output-dir", "-o", help="Output dir for predictions"
    ),
):

    print("HERE: ", config.RESULTS_ROOT, " ->", output_dir)
	
    # Variables definidas directamente en el codigo
    model_folder = config.get_model_folder(model_name)  # Carpeta del modelo
    num_div = config.PREDICTION_NUM_DIV   # Cuantos batches se le entrega a la imagen (aumentar si no cabe en memoria).Por ej. 16,32,64,96,128, etc
    step = config.STEP_SIZE_PREDICTION    # A menor step,mayor numero de parches a predecir (no cambiar Z = 16, puede variar XY entre 64-128) Por ej. (16,96,96)
    batch_size = config.PREDICTION_BATCH_SIZE  # No cambiar
    alineacion = False  # Si la imagen ya esta alineada False
    preprocessing_enabled = config.PREPROCESSING_ENABLED  # Si se quiere preprocesar la imagen antes de la prediccion

    from utils.loss_functions import FocalTverskyLoss, MSE_loss, dice_coef
    from keras_models.attention_unet_3d import Attention_ResUNet_3D

    folder_path = model_folder
    model = Attention_ResUNet_3D(
        config.INPUT_SHAPE, config.DROPOUT_RATE, config.BATCH_NORM, config.FILTER_NUM
    )
    best_model = config.BEST_MODEL_FILENAME
    best_model_path = os.path.join(folder_path, best_model)
    model.load_weights(best_model_path)

    # Procesar todas las imagenes en el directorio
    for image_file in os.listdir(image_dir):
        if any(
            image_file.endswith(ext) for ext in config.IMAGE_EXTENSIONS
        ):  # Filtrar archivos de imagen
            image_path = os.path.join(image_dir, image_file)

            # Cargar imagen
            def load_data_prediction(path):
                name = os.path.basename(path).split(".")[0]
                image = imread(path)
                original_shape = image.shape
                # Invert all images if "Mouse" is in the folder path
                if "Mouse" in path:
                        image = np.amax(image) - image

		# new scaling
                if avg_radius != config.AVG_RADIUS:
                        image = rescale_image(image, avg_radius, target_radius=config.AVG_RADIUS)

                image = normalize(image)

                return image.astype(np.float16), name, original_shape

            img_val, name, original_shape = load_data_prediction(image_path)
            print("HERE NOW: ", config.RESULTS_ROOT, " ->", output_dir, " ,", model_name, " ,", name, batch_size )

            # n-levels up
            _image_dir = Path(image_dir)
            n = 2  # change this to go 2 levels up, 3 levels up, etc.
            nth_parent = _image_dir.parents[n-1].name

            dir_prediction = os.path.join(output_dir, model_name, nth_parent, name)
            print("Saving data prediction  at: ", dir_prediction)
            create_path(dir_prediction)
#            imwrite(os.path.join(dir_prediction, f"{name}_orig.tif"), img_val)

            # Align (solo si necesita alineacion)
            if alineacion:
                img_val = aligned_image_simple(img_val).astype(np.float16)
#                imwrite(os.path.join(dir_prediction, f"{name}_aligned.tif"), img_val)

            # Preprocesamiento
            if preprocessing_enabled:
                img_preprocess = preprocessing(img_val).astype(np.float16)
            else:
                img_preprocess = preprocessing_simple(img_val).astype(np.float16)
#            imwrite(os.path.join(dir_prediction, f"{name}_preprocessed.tif"), img_preprocess)

            # Prediccion
            binary, grad_map = prediction(
                img_preprocess,
                model,
                dir_prediction,
                name,
                num_div=num_div,
                step=step,
                batch_size=batch_size,
            )

            # Estimacion de centroides
            seeds, initial_points, final_points = centroid_estimation(
                binary, grad_map, dir_prediction
            )

            # Calculo de etiquetas
            labels_id = calculate_labels(seeds, final_points)

            # Proximidad espacial
            labels = nuclei_segmentation(binary, initial_points, labels_id, dir_prediction, name, original_shape)


if __name__ == "__main__":
    typer.run(main)
