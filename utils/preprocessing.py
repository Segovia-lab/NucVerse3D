#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 17:34:45 2024

@author: ws2
"""


import numpy as np
from skimage import exposure, restoration
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from skimage.filters import gaussian
from scipy.ndimage import gaussian_filter

from skimage.transform import resize
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1
from joblib import Parallel, delayed
from joblib_progress import joblib_progress

from skimage.transform import resize
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1
import multiprocessing
def create_psf(size=15, sigma=1):
    """
    Crea una función de dispersión de punto (PSF) inicial como un núcleo gaussiano normalizado.
    """
    psf = np.zeros((size, size))
    center = size // 2
    psf[center, center] = 1
    psf = gaussian_filter(psf, sigma=sigma)
    psf /= psf.sum()  # Normalizar la PSF para que la suma sea 1
    return psf

def deconvolution_blind(image, psf):
    """
    Realiza la deconvolución ciega de una imagen usando el enfoque de Wiener-Hunt.
    """
    deconvolved, _ = restoration.unsupervised_wiener(image, psf)
    return deconvolved

def deconvolution_task(args):
    z, image_slice, psf = args
    return z, deconvolution_blind(image_slice, psf)

def parallel_deconvolution(corrected_image, psf):
    deconvolved_image_3d = np.empty_like(corrected_image)

    # Preparar las tareas para la paralelización
    tasks = [(z, corrected_image[z], psf) for z in range(len(corrected_image))]
    
    # Obtener el número de CPUs y calcular la mitad
    num_cores = multiprocessing.cpu_count()
    num_jobs = num_cores // 2
    
    # Usar Joblib con barra de progreso
    with joblib_progress("Deconvolucion:", total=len(tasks)):
        results = Parallel(n_jobs=num_jobs)(delayed(deconvolution_task)(task) for task in tasks)

    # Guardar los resultados en el orden correcto
    for z, deconvolved_slice in results:
        deconvolved_image_3d[z] = deconvolved_slice

    return deconvolved_image_3d


def correct_intensity_per_slice_uneven(image_3d, sigma = 50, epsilon = 1e-10):
    gaussian_slice = np.array([gaussian_filter(i, sigma = sigma) for i in image_3d.astype(np.float32)])
    mean_slice = np.mean(image_3d, axis=(1, 2))
    corrected = image_3d*(mean_slice[:, np.newaxis, np.newaxis]/(gaussian_slice + epsilon))
    return corrected

def normalize_image(image, p_lower=2, p_upper=99.8):
    # Calcular los percentiles
    lower_bound = np.percentile(image, p_lower)
    upper_bound = np.percentile(image, p_upper)
    # Normalizar la imagen utilizando los percentiles
    normalized_image = exposure.rescale_intensity(image, in_range=(lower_bound, upper_bound), out_range=(0, 1))
    return normalized_image

def preprocessing(img):
    corrected_image = correct_intensity_per_slice_uneven(img)
    psf = create_psf()
    deconvolved_image_3d = parallel_deconvolution(corrected_image, psf)
    norm_img = normalize_image(deconvolved_image_3d)
    return norm_img.astype(np.float16)


def optical_flow_task_middle(image0, image1, order=3):
    v, u = optical_flow_tvl1(image0, image1)
    nr, nc = image0.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    image1_warp = warp(image1, np.array([row_coords + v/2, col_coords + u/2]), mode='edge', order=order)
    image0_warp = warp(image0, np.array([row_coords - v/2, col_coords - u/2]), mode='edge', order=order)
    return image0_warp, image1_warp

def parallel_optical_flow(args):
    img_par_rescale, img_impar_rescale, i, order = args
    return optical_flow_task_middle(img_par_rescale[i], img_impar_rescale[i], order)

def aligned_image(img, order=3):
    if img.shape[1] % 2 != 0:
        img = img[:, :-1]
    
    img_par = np.stack([img[:, i, :] for i in range(len(img[0])) if i % 2 == 0], axis=1)
    img_impar = np.stack([img[:, i, :] for i in range(len(img[0])) if i % 2 != 0], axis=1)
    
    img_par_rescale = resize(img_par, (img_par.shape[0], img_par.shape[1] * 2, img_par.shape[2]), order=3)
    img_impar_rescale = resize(img_impar, (img_impar.shape[0], img_impar.shape[1] * 2, img_impar.shape[2]), order=3)
    
    # Obtener el número de CPUs y calcular la mitad
    num_cores = multiprocessing.cpu_count()
    num_jobs = num_cores // 2
    
    # Paralelización usando Joblib
    with joblib_progress("Alineacion:", total=len(img_impar_rescale)):
        results = Parallel(n_jobs=num_jobs)(delayed(optical_flow_task_middle)(img_par_rescale[i], img_impar_rescale[i], order) for i in range(len(img_impar_rescale)))
    
    # Separar los resultados en image0_warp e image1_warp
    image0_warp, image1_warp = zip(*results)
    image0_warp, image1_warp = np.array(image0_warp), np.array(image1_warp)
    
    final = np.zeros_like(img_impar_rescale)
    for i in range(len(final[0])):
        if i % 2 == 0:
            final[:, i] = image0_warp[:, i]
        else:
            final[:, i] = image1_warp[:, i]
    
    return final


def optical_flow_task_middle(image0, image1, order=3):
    v, u = optical_flow_tvl1(image0, image1)
    nr, nc = image0.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    image1_warp = warp(image1, np.array([row_coords + v/2, col_coords + u/2]), mode='edge', order=order)
    image0_warp = warp(image0, np.array([row_coords - v/2, col_coords - u/2]), mode='edge', order=order)
    return image0_warp, image1_warp


def aligned_image_simple(img, order=3):
    if img.shape[1] % 2 != 0:
        img = img[:, :-1]
    
    img_par = np.stack([img[:, i, :] for i in range(len(img[0])) if i % 2 == 0], axis=1)
    img_impar = np.stack([img[:, i, :] for i in range(len(img[0])) if i % 2 != 0], axis=1)
    
    img_par_rescale = resize(img_par, (img_par.shape[0], img_par.shape[1] * 2, img_par.shape[2]), order=order)
    img_impar_rescale = resize(img_impar, (img_impar.shape[0], img_impar.shape[1] * 2, img_impar.shape[2]), order=order)
    
    # Obtener el número de CPUs y calcular la mitad
    num_cores = multiprocessing.cpu_count()
    num_jobs = num_cores // 2
    
    # Paralelización usando Joblib
    with joblib_progress("Alineacion:", total=len(img_impar_rescale)):
        results = Parallel(n_jobs=num_jobs)(delayed(optical_flow_task_middle)(img_par_rescale[i], img_impar_rescale[i], order) for i in range(len(img_impar_rescale)))
    
    # Separar los resultados en image0_warp e image1_warp
    image0_warp, image1_warp = zip(*results)
    image0_warp, image1_warp = np.array(image0_warp), np.array(image1_warp)
    
    final = np.zeros_like(img_impar_rescale)
    for i in range(len(final[0])):
        if i % 2 == 0:
            final[:, i] = image0_warp[:, i]
        else:
            final[:, i] = image1_warp[:, i]
    
    return final.astype(np.float16)