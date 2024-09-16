# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:39:31 2024

@author: WORKSTATION 2
"""

from scipy import ndimage as ndi
from tqdm import tqdm
import numpy as np
from skimage.filters import threshold_otsu, threshold_multiotsu, threshold_li, threshold_local, threshold_sauvola
import multiprocessing as mp

def normalizer(img):
    return (img-img.min())/(img.max()-img.min())

def grad_point(gradient_vector, point):
    # Correctly reshape the coordinate array
    coords = point
    # Separate the gradient components
    grad_z_component = gradient_vector[..., 0]
    grad_y_component = gradient_vector[..., 1]
    grad_x_component = gradient_vector[..., 2]
    # Map the coordinates for each component
    grad_z = ndi.map_coordinates(grad_z_component, coords, order=3)
    grad_y = ndi.map_coordinates(grad_y_component, coords, order=3)
    grad_x = ndi.map_coordinates(grad_x_component, coords, order=3)
    return grad_z, grad_y, grad_x

def gradient_descent(gradient, start, learn_rate=.1, n_iter=200, tolerance=1e-6):
    vector = np.array(start, dtype=gradient.dtype)
    history = []
    for _ in tqdm(range(n_iter)):
        grad_z, grad_y, grad_x = grad_point(gradient, vector)
        diff = learn_rate * np.array([grad_z, grad_y, grad_x])  
        a,b,c = diff
        check = np.sqrt(a**2 + b**2 + c**2)
        diff_check = diff[:,check>tolerance]
        vector = vector[:,check>tolerance] + diff_check
        history.append(vector.astype('int16'))
    return vector.astype('int16'), history

def gradient_descent_real(gradient, start, learn_rate=.1, n_iter=200, tolerance=1e-6):
    vector = np.array(start, dtype=gradient.dtype)
    history = []
    for _ in tqdm(range(n_iter)):
        grad_z_1, grad_y_1, grad_x_1 = grad_point(gradient, vector)
        diff = learn_rate * np.array([grad_z_1, grad_y_1, grad_x_1])
        a,b,c = diff
        check = np.sqrt(a**2 + b**2 + c**2)
        diff_check = diff[:,check>tolerance]
        vector = vector[:,check>tolerance] + diff_check
        history.append(vector)
    return vector.astype('int16'), history


def sobel_threshold(sobel_vol_M):
    th_sqrt_sobel_dz = np.zeros_like(sobel_vol_M)
    th_sqrt_sobel_dy = np.zeros_like(sobel_vol_M)
    th_sqrt_sobel_dx = np.zeros_like(sobel_vol_M)
    for i in range(len(sobel_vol_M)):
        th = threshold_multiotsu(sobel_vol_M[i])
        th_sqrt_sobel_dz[i] = sobel_vol_M[i]>th[1]
    for i in range(len(sobel_vol_M[0])):
        th = threshold_multiotsu(sobel_vol_M[:,i])
        th_sqrt_sobel_dy[:,i] = sobel_vol_M[:,i]>th[1]
    for i in range(len(sobel_vol_M[0,0])):
        th = threshold_multiotsu(sobel_vol_M[:,:,i])
        th_sqrt_sobel_dx[:,:,i] = sobel_vol_M[:,:,i]>th[1]
    
    return th_sqrt_sobel_dz, th_sqrt_sobel_dy, th_sqrt_sobel_dx

def sobel_max_M(dz,dy,dx):
    sobel_z_z = ndi.sobel(dz, axis = 0, mode='reflect')
    sobel_z_y = ndi.sobel(dz, axis = 1, mode='reflect')
    sobel_z_x = ndi.sobel(dz, axis = 2, mode='reflect')
    sobel_z = np.sqrt(sobel_z_z**2 + sobel_z_y**2 + sobel_z_x**2)
    
    sobel_y_z = ndi.sobel(dy, axis = 0, mode='reflect')
    sobel_y_y = ndi.sobel(dy, axis = 1, mode='reflect')
    sobel_y_x = ndi.sobel(dy, axis = 2, mode='reflect')
    sobel_y = np.sqrt(sobel_y_z**2 + sobel_y_y**2 + sobel_y_x**2)
    
    sobel_x_z = ndi.sobel(dx, axis = 0, mode='reflect')
    sobel_x_y = ndi.sobel(dx, axis = 1, mode='reflect')
    sobel_x_x = ndi.sobel(dx, axis = 2, mode='reflect')
    sobel_x = np.sqrt(sobel_x_z**2 + sobel_x_y**2 + sobel_x_x**2)
    
    sobel_vol_M = np.max(np.array([sobel_z,sobel_y,sobel_x]), axis=0)
    return sobel_z, sobel_y, sobel_x, sobel_vol_M


def gradient_descent_parallel(gradient, start, learn_rate=.1, n_iter=200, tolerance=1e-6, num_processes = 4):
    # Obtener el numero de punto para calcular de acuerdo al tamaño de las unidades de computo el tamaño del chunk
    m = start.shape[1]
    chunk_size = m // num_processes
    pool = mp.Pool(processes=num_processes)

    # Generación de los chnk y el inpunt
    chunks = [start[:, i*chunk_size:(i+1)*chunk_size] for i in range(num_processes)]
    for i in chunks:
        print(i.shape)
    input_ = [(gradient, chunk) for chunk in chunks]

    # Resultados en paralelo
    results = pool.starmap(gradient_descent, input_)
    pool.close()
    pool.join()

    # Concatenar los resultados para que sean igual si es que no hubiera sido en paralelo
    gd, history = zip(*results)
    history_complete = [list(col) for col in zip(*history)]
    history_complete = [np.concatenate(i, axis = -1) for i in history_complete] 
    
    return gd, history_complete

def filt_history(history, h, w, d):
    history_filt = [history[i][:,np.logical_and.reduce((history[i][0]>=0,
                                                       history[i][0]<=h-1, 
                                                       history[i][1]>=0,
                                                       history[i][1]<=w-1,
                                                       history[i][2]>=0,
                                                       history[i][2]<=d-1))] for i in range(len(history))]
    return history_filt