# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:52:58 2024

@author: ws2
"""
#%%
import sys
import os
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current_path)
from utils.utils import pad_img, normalize, create_path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from patchify import patchify
from tifffile import imread, imwrite
from glob import glob
import numpy as np
from utils.preprocessing import preprocessing, deconvolution_task
from tqdm import tqdm
import h5py

def load_data_training(path):
    images = sorted(glob(os.path.join(path, "imagenes/*.[tT][iI][fF]*")))
    images = np.array([imread(x) for x in images])
    labels = sorted(glob(os.path.join(path, "labels/*.[tT][iI][fF]*")))
    labels = np.array([imread(x) for x in labels])
    return images, labels

def generate_patches(images, step, patch_size, keep=None):
    patches = np.array([patchify(x, step=step, patch_size=patch_size) for x in images])
    patches = patches.reshape((-1,)+patch_size)
    return patches

def generate_training_val_set(dir_input = "./dataset",
                              dir_output = "./training",
                              step = (128,128,128),
                              patch_size = (128,128,128),
                              max_percent = 10, 
                              oversampling = False):

    # Lee las imagenes
    images_train, labels_train = load_data_training(dir_input)
    images_train_preprocess = np.array([preprocessing(x) for x in tqdm(images_train)])
    
    # Agrega un pad para poder dividir las imagenes del tamaño del parche
    images_train_pad = np.array([pad_img(x, patch_size, step, 'constant') for x in images_train_preprocess])
    labels_train_pad = np.array([pad_img(x, patch_size, step, 'constant') for x in labels_train])
    
    # Se generan los parches sin solapamiento
    images_patches_train = generate_patches(images_train_pad, step, patch_size)
    labels_patches_train = generate_patches(labels_train_pad, step, patch_size)
    
    # Se filtran los parches de acuerdo al porcentaje del valor maximo establecido (max_percent = 5, quiere decir que filtrara todos
    # los que tengan señal menor o igual al 5% del maximo de señal en los parches)
    binary_sum_labels = np.array([(x>0).sum() for x in labels_patches_train])
    images_patches_train_filt = images_patches_train[binary_sum_labels > (binary_sum_labels.max()*max_percent/100)]
    labels_patches_train_filt = labels_patches_train[binary_sum_labels > (binary_sum_labels.max()*max_percent/100)]
    
    images_patches_train_filt_norm = np.array([normalize(x) for x in tqdm(images_patches_train_filt)])
    
    # Generacion de un training y val set
    X_train, X_val, y_train, y_val = train_test_split(images_patches_train_filt_norm, labels_patches_train_filt, test_size=0.1)
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

    # Balancear datos
    binary_patch = y_train != 0
    vol_patch = np.array([x.sum() for x in binary_patch])
    hist, bins = np.histogram(vol_patch, bins=5)

    # Iterar sobre cada bin del histograma
    X_train_i = []
    y_train_i = []
    for i in range(len(bins) - 1):
        if(i<len(bins)-2):
            bin_images = np.where((vol_patch >= bins[i]) & (vol_patch < bins[i+1]))
        else:
            bin_images = np.where((vol_patch >= bins[i]) & (vol_patch <= bins[i+1]))
        X_train_i.append(X_train[bin_images])
        y_train_i.append(y_train[bin_images])
    
    max_samples = max(len(category) for category in X_train_i)
    X_train_oversampled = []
    y_train_oversampled = []
    # Hacer oversampling en cada categoría para que tengan el mismo número de muestras
    for i in range(len(X_train_i)):
        p = np.random.choice(range(len(X_train_i[i])), max_samples-len(X_train_i[i]))
        X_train_oversampled.append(np.concatenate([X_train_i[i], X_train_i[i][p]], axis = 0))
        y_train_oversampled.append(np.concatenate([y_train_i[i], y_train_i[i][p]], axis = 0))
    
    X_train_oversampled = np.reshape(X_train_oversampled, ((-1,) + np.shape(X_train_oversampled)[-3:]))
    y_train_oversampled = np.reshape(y_train_oversampled, ((-1,) + np.shape(y_train_oversampled)[-3:]))
    
    if(oversampling == True):
        X_train = X_train_oversampled
        y_train = y_train_oversampled
        
       
    create_path(os.path.join(dir_output))
    # Ruta del archivo HDF5
    hdf5_file_path_training = os.path.join(dir_output, 'data_train.hdf5')
    # Crear el archivo HDF5 y los datasets de imágenes y etiquetas
    with h5py.File(hdf5_file_path_training, 'w') as hf:
        # Crear datasets vacíos con el tamaño adecuado
        hf.create_dataset('images', data=X_train, dtype=np.float16)
        hf.create_dataset('labels', data=y_train, dtype=np.uint16)
    
    hdf5_file_path_validation = os.path.join(dir_output, 'data_val.hdf5')
    # Crear el archivo HDF5 y los datasets de imágenes y etiquetas
    with h5py.File(hdf5_file_path_validation, 'w') as hf:
        # Crear datasets vacíos con el tamaño adecuado
        hf.create_dataset('images', data=X_val, dtype=np.float16)
        hf.create_dataset('labels', data=y_val, dtype=np.uint16)

#%%
def main():
    generate_training_val_set(dir_input= "./dataset",
                              dir_output = "./training sets/training_overlape_32",
                              step = (32,96,96),
                              patch_size = (64,128,128),
                              oversampling=False)
if __name__ == '__main__':
    main()
    