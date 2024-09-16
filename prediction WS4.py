#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:19:56 2024

@author: ws2
"""

#%%
import os
import tensorflow as tf
import sys
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current_path)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


import numpy as np
from tifffile import imread, imwrite
from utils.utils import create_path, normalize
from utils.preprocessing import preprocessing, aligned_image_simple
from utils.utils import pad_img, linear_interp_vol, unpad_img
from patchify import patchify
import gc
import cc3d
from scipy import ndimage as ndi
import scipy.spatial
from skimage.morphology import remove_small_objects

def prediction(img, model, dir_prediction = None, name = None, batch_size = 2, num_div = 16, patch_size = (64,128,128), step = (16,64,64)):
        
    # Patches prediction
    from patchify import patchify
    import gc
    
    from utils.utils import pad_img
    img_val_pad = pad_img(img, patch_size, step, 'constant')
    img_patchy_val = patchify(img_val_pad, patch_size, step=step)
    img_patches_reshaped = img_patchy_val.reshape((-1,)+patch_size)
    
    img_val_pad_shape = img_val_pad.shape
    img_patchy_val_shape = img_patchy_val.shape
    
    del img_val_pad, img_patchy_val
    gc.collect()
    
    # Prediction
    from tensorflow.keras.backend import clear_session
    from tqdm import tqdm
    
    y_pred_bin = np.zeros((img_patches_reshaped.shape) + (2,), dtype = np.float16)
    y_pred_map = np.zeros((img_patches_reshaped.shape) + (3,), dtype = np.float16)
    img_patches_reshaped = [i for i in img_patches_reshaped]
    tiles = len(img_patches_reshaped)//num_div
    print(tiles)
    for j in tqdm(range(0, len(img_patches_reshaped), tiles)): 
        curr_block = np.array(img_patches_reshaped[:tiles])
        print(f' Processed elements from {j} to {(j+len(curr_block))}. Remaining elements: {len(img_patches_reshaped)}')
        y_pred_bin_batch, y_pred_map_batch = model.predict(curr_block, batch_size = batch_size, verbose = 0)
        y_pred_bin[j:j+len(curr_block)] = y_pred_bin_batch.astype('float16')
        y_pred_map[j:j+len(curr_block)] = y_pred_map_batch.astype('float16')
        del img_patches_reshaped[:tiles]
        clear_session()
        gc.collect()
        
    del img_patches_reshaped
    gc.collect()    
    # Reshape
    y_pred_bin = np.array(y_pred_bin).reshape(img_patchy_val_shape+(2,))
    y_pred_map = np.array(y_pred_map).reshape(img_patchy_val_shape+(3,))
    
    # Linear interpolation
    from utils.utils import linear_interp_vol
    y_pred_bin = linear_interp_vol(y_pred_bin, img_val_pad_shape, patch_size, step, 2) 
    y_pred_map = linear_interp_vol(y_pred_map, img_val_pad_shape, patch_size, step, 3)

    # Argmax
    y_pred_bin = np.argmax(y_pred_bin, axis=-1)

    # Unpad
    from utils.utils import unpad_img
    y_pred_bin = unpad_img(y_pred_bin, img.shape)
    y_pred_map = unpad_img(y_pred_map, img.shape + (3,))
    
    imwrite(os.path.join(dir_prediction, f'{name}_binary.tif'), y_pred_bin.astype('uint16'))
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

def gradient_descent_momentum(gradient, start, learn_rate=0.05, n_iter=200, tolerance=1e-6, momentum = 0.8):
    vector = np.array(start, dtype=gradient.dtype)
    history = []
    diff = 0
    for _ in tqdm(range(n_iter)):
        grad_z, grad_y, grad_x = grad_point(gradient, vector)
        diff = learn_rate*np.array([grad_z, grad_y, grad_x]) + momentum*diff      
        vector = vector + diff
        history.append(vector.astype('int16'))
    return vector.astype('int16'), history

def filt_history(history, h, w, d):
    logical_and = np.logical_and.reduce((history[0]>=0,
                                         history[0]<=h-1, 
                                         history[1]>=0,
                                         history[1]<=w-1,
                                         history[2]>=0,
                                         history[2]<=d-1))
    history_filt = history[:,logical_and]
    return history_filt, logical_and


def centroid_estimation(vol_seg, gradient_vector, dir_prediction):

    gradient_vector = gradient_vector.astype(np.float64)
    points = np.array(np.where(vol_seg>0)).astype('int16')
    
    # Descenso de gradiente
    gd, history = gradient_descent_momentum(gradient_vector, points, n_iter=50)
    
    # FIltro de limites
    h, w, d = vol_seg.shape

    zeros_ = np.zeros_like(vol_seg, dtype=np.int32)
    history_filter, and_filt = filt_history(np.floor(history[-1]), h, w, d)
    
    
    # Generacion de centroides
    history_filter_points, history_filter_counts = np.unique(history_filter, axis=1, return_counts=True)
    a,b,c = history_filter_points.astype(np.int32)
    zeros_[a,b,c] += history_filter_counts.astype(np.int32)
    
    # imwrite(os.path.join(dir_prediction, f'{dir_prediction.split("/")[-1]}_zeros.tif'), zeros_)
    
    # Filtro gaussiano y threshold de centroides
    gaus_zeros = ndi.gaussian_filter(zeros_, 1)
    # imwrite(os.path.join(dir_prediction, f'{dir_prediction.split("/")[-1]}_gauss_zeros.tif'), gaus_zeros)
    
    # th = gaus_zeros[gaus_zeros.astype("int") != 0].mean() + 0*gaus_zeros[gaus_zeros.astype("int") != 0].std()
    th = np.mean(gaus_zeros) + 5*np.std(gaus_zeros)
    # Gerneracion de semillas
    import cc3d
    seeds = gaus_zeros>th
    seeds = cc3d.connected_components(seeds)
    # imwrite(os.path.join(dir_prediction, f'{dir_prediction.split("/")[-1]}_seeds.tif'), seeds)
    return seeds, points[:, and_filt], history_filter

def calculate_labels(seeds, final_points, radius = 5):
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
def nuclei_segmentation(vol_seg, initial_points, labels_id, dir_prediction, name, vox_threhsold = 20):
    a, b, c = initial_points
    labels = np.zeros_like(vol_seg, dtype=np.int16)
    labels[a, b, c] = labels_id
    labels = remove_small_objects(labels, min_size=20)
    imwrite(os.path.join(dir_prediction, f'{name}_labels.tif'), labels.astype(np.uint16))
    return labels

#%%
def main():
    # Variables definidas directamente en el código
    model_folder = '8. 3d raunet overlape 32 dataet total 64_128 dropout 0.4 filters 32' # Carpeta del modelo
    image_path = './images/DAPI_16w_Control_6/DAPI_16w_Control_6.tif' # Path de la imagen a predecir
    output_dir = './prediction/' # Carpeta output
    num_div = 32 # Cuantos batches se le entrega a la imagen (aumentar si no cabe en memoria).Por ej. 16,32,64,96,128, etc
    step = (16, 80, 80) # A menor step,mayor numero de parches a predecir (no cambiar Z = 16, puede variar XY entre 64-128) Por ej. (16,96,96)
    batch_size = 2 # No cambiar
    alineacion = True # Si la imagen ya esta alineada False
    
    from utils.loss_functions import FocalTverskyLoss, MSE_loss, dice_coef
    from utils.models_3d_attention import Attention_ResUNet_3D
    
    folder_path = os.path.join('./models_training', model_folder)
    model = tf.keras.models.load_model(os.path.join(folder_path, 'modelSave.h5'), compile=False)
    best_model = 'best_model.hdf5'
    best_model_path = os.path.join(folder_path, best_model)
    model.load_weights(best_model_path)

    # Cargar imagen
    def load_data_prediction(path):
        name = os.path.basename(path).split(".")[0]
        images = normalize(imread(path))
        return images.astype(np.float16), name

    img_val, name = load_data_prediction(image_path)
    dir_prediction = os.path.join(output_dir, model_folder, name)
    create_path(dir_prediction)
    imwrite(os.path.join(dir_prediction, f'{name}_orig.tif'), img_val)
    
    # Align (solo si necesita alineacion)
    if(alineacion):
        img_val = aligned_image_simple(img_val).astype(np.float16) 
        imwrite(os.path.join(dir_prediction, f'{name}_aligned.tif'), img_val)
    
    # Preprocesamiento
    img_preprocess = preprocessing(img_val).astype(np.float16)
    imwrite(os.path.join(dir_prediction, f'{name}_preprocessed.tif'), img_preprocess)

    # Predicción
    binary, grad_map = prediction(img_preprocess, model, dir_prediction, name, num_div=num_div, step=step, batch_size=batch_size)

    # Estimación de centroides
    seeds, initial_points, final_points = centroid_estimation(binary, grad_map, dir_prediction)

    # Cálculo de etiquetas
    labels_id = calculate_labels(seeds, final_points)

    # Proximidad espacial
    labels = nuclei_segmentation(binary, initial_points, labels_id, dir_prediction, name)

if __name__ == '__main__':
    main()


