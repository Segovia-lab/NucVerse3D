from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
import cc3d
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
from skimage.transform import resize
from joblib import Parallel, delayed, cpu_count

def create_path(path):
    if not os.path.exists(path):
        # Create the directory
        os.makedirs(path)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
        
def generate_objects(distance_map, binary_seg):
    th_otsu = threshold_otsu(distance_map)
    print(th_otsu)
    im_otsu = (distance_map > th_otsu)
    th_otsu_components = cc3d.connected_components(im_otsu, connectivity = 6)
    labels_distances = watershed(-distance_map, th_otsu_components, mask = binary_seg)
    return labels_distances

def convert_labels_with_borders(img, border_size = 1):
    import numpy as np
    from scipy.ndimage import binary_dilation, binary_erosion
    import pandas as pd
    import swifter 
    import os
    os.environ['SWIFTER_PROGRESS_BAR'] = 'False'
    
    # Definition of erotion
    n_erosions = 1
    erosion_kernel = np.ones((3,3,3), np.uint8)      ## Start by eroding edge pixels
    
    # Get each object for independent eriotion     
    image_per_object = [img==x for x in np.unique(img)[1:]]
    
    dF_dict = {"image": image_per_object}
    dF = pd.DataFrame(dF_dict)
    
    # Funciton to apply in parallel
    def apply_erosion(image):
        image = np.reshape(image, img.shape)
        return binary_erosion(image, erosion_kernel, iterations = n_erosions)
    
    # Apply function
    dF['eroded_image'] = dF['image'].swifter.progress_bar(False).apply(apply_erosion)
    
    # Getting the real image in 3d 
    eroded_image = np.sum(dF['eroded_image'], axis = 0)
    
    # Apply dilatation to gerate the border   
    kernel_size = 2*border_size + 1 
    dilation_kernel = np.ones((kernel_size, kernel_size, kernel_size), np.uint8)   #Kernel to be used for dilation
    dilated  = binary_dilation(eroded_image, dilation_kernel, iterations = n_erosions)
    
    ## Replace 255 values to 127 for all pixels. Eventually we will only define border pixels with this value
    dilated_127 = np.where(dilated, 127, 0)
    
    #In the above dilated image, convert the eroded object parts to pixel value 255
    #What's remaining with a value of 127 would be the boundary pixels. 
    original_with_border = np.where(eroded_image, 255, dilated_127)
    return original_with_border
       
def sigmoid(x, k):
    return 1 / (1 + np.exp(-k * (x - 0.5)))     

def interp_X(img, output_shape, patch_size, step, overlap, z, j, k):
    merged = np.zeros((patch_size[0], patch_size[1], output_shape[-1], k), dtype=img.dtype)
    for i in range(img.shape[2]):
        img1 = merged.copy()
        img2 = img[z, j, i]
        
        if np.logical_and(img1[:, :, step[2]*i:step[2]*i+patch_size[2]], img2).sum():
            overlap1 = img1[:, :, step[2]*i:step[2]*i+overlap]
            overlap2 = img2[:, :, :overlap]
            mask = np.linspace(0, 1, overlap)
            mask = sigmoid(mask, overlap)
            mask = np.ones((patch_size[0], patch_size[1], overlap)) * mask
            mask = np.expand_dims(mask, -1)
            mask = np.repeat(mask, k, axis=-1)
            merged[:, :, step[2]*i:step[2]*i+overlap] = (1 - mask) * overlap1 + mask * overlap2
            merged[:, :, step[2]*i+overlap:step[2]*i+patch_size[2]] = img2[:, :, overlap:]
        else:
            merged[:, :, step[2]*i:step[2]*i+patch_size[2]] = img2
    return merged

def interp_Y(img, output_shape, patch_size, step, overlap, z, k, merged_x):
    merged_xy = np.zeros((patch_size[0], output_shape[1], output_shape[2], k), dtype=img.dtype)
    for i in range(img.shape[1]):
        img1 = merged_xy.copy()
        img2 = merged_x[z, i]
        
        if np.logical_and(img1[:, step[1]*i:step[1]*i+patch_size[1], :], img2).sum():
            overlap1 = img1[:, step[1]*i:step[1]*i+overlap, :]
            overlap2 = img2[:, :overlap, :]
            mask = np.linspace(0, 1, overlap)
            mask = sigmoid(mask, overlap)
            mask = mask.reshape(overlap, 1)
            mask = np.ones((patch_size[0], overlap, output_shape[2])) * mask
            mask = np.expand_dims(mask, -1)
            mask = np.repeat(mask, k, axis=-1)
            merged_xy[:, step[1]*i:step[1]*i+overlap, :] = (1 - mask) * overlap1 + mask * overlap2
            merged_xy[:, step[1]*i+overlap:step[1]*i+patch_size[1], :] = img2[:, overlap:, :]
        else:
            merged_xy[:, step[1]*i:step[1]*i+patch_size[1], :] = img2
    return merged_xy

import numpy as np
from joblib import Parallel, delayed

def raised_cosine(n):
    if n <= 0:
        return np.ones(1, dtype=np.float32)
    t = np.linspace(0, np.pi, n, dtype=np.float32)
    return 0.5 - 0.5*np.cos(t)

def _mask_1d(L, ovl, pos, pos_max, ramp):
    w = np.ones(L, dtype=np.float32)
    if ovl > 0:
        if pos != 0:
            w[:ovl] = ramp
        if pos != pos_max:
            w[-ovl:] = ramp[::-1]
    return w

def stitch_joblib_3d(img_patchy_val, patch_size, step, padded_shape, n_jobs=-1):
    """
    img_patchy_val: indexable como img_patchy_val[i, j, k] -> (px, py, pz)
                    (sirve tanto lista/ndarray de objetos como ndarray 6D)
    patch_size: (px, py, pz)
    step: (sx, sy, sz)
    padded_shape: (X, Y, Z) del volumen destino
    """
    Gx, Gy, Gz = img_patchy_val.shape[:3]
    max_x, max_y, max_z = Gx-1, Gy-1, Gz-1

    overlap = tuple(np.array(patch_size) - np.array(step))
    if any(o < 0 for o in overlap):
        raise ValueError("Step size must be <= patch size en las 3 dims.")

    rec = np.zeros(padded_shape, dtype=np.float16)
    rec_m = np.zeros(padded_shape, dtype=np.float16)

    rx = np.linspace(0, 1, overlap[0], dtype=np.float16)
    ry = np.linspace(0, 1, overlap[1], dtype=np.float16)
    rz = np.linspace(0, 1, overlap[2], dtype=np.float16)

    def process_one(i, j, k):
        patch = img_patchy_val[i, j, k]  # (px, py, pz)
        wx = _mask_1d(patch_size[0], overlap[0], i, max_x, rx)
        wy = _mask_1d(patch_size[1], overlap[1], j, max_y, ry)
        wz = _mask_1d(patch_size[2], overlap[2], k, max_z, rz)
        mask = (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).astype(np.float16)

        xs, ys, zs = i*step[0], j*step[1], k*step[2]
        return xs, ys, zs, patch*mask, mask
    
    from tqdm import tqdm
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_one)(i, j, k)
        for i in tqdm(range(Gx)) for j in range(Gy) for k in range(Gz)
    )
    
    for xs, ys, zs, wpatch, wmask in tqdm(results):
        rec[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += wpatch
        rec_m[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += wmask
    return rec/rec_m

def linear_interp_vol(img, output_shape, patch_size, step, k):
    #suponiendo que img es 3D con k canales
    reconstructed = np.stack([stitch_joblib_3d(img[...,c], patch_size, step, output_shape, n_jobs=-1) for c in range(k)], axis = -1)
    return reconstructed.astype(np.float16)


def linear_interp_vol_old(img, output_shape, patch_size, step, k):
    overlap = tuple(x - y for x, y in zip(patch_size, step))   
    
    # Interpolacion en X
    print("Interpolacion en X")
    merged_x = []
    for z in tqdm(range(img.shape[0])):
        for j in range(img.shape[1]):           
            merged_x.append(interp_X(img, output_shape, patch_size, step, overlap[2], z, j, k))
    merged_x = np.reshape(merged_x, (img.shape[0], img.shape[1], patch_size[0], patch_size[1], output_shape[-1], k))
    
    # Interpolacion en Y
    print("Interpolacion en Y")
    merged_y = []
    for z in tqdm(range(img.shape[0])):
        merged_y.append(interp_Y(img, output_shape, patch_size, step, overlap[1], z, k, merged_x))
        
    # Interpolacion en Z
    overlap = overlap[0]
    print("Interpolacion en Z")
    merged_xyz = np.zeros(output_shape + (k,), dtype=img.dtype)
    for i in tqdm(range(img.shape[0])):
        img1 = merged_xyz.copy()
        img2 = merged_y[i]
        if np.logical_and(img1[step[0]*i:step[0]*i+patch_size[0], :, :], img2).sum():
            overlap1 = img1[step[0]*i:step[0]*i+overlap, :, :]
            overlap2 = img2[:overlap, :, :]
            mask = np.linspace(0, 1, overlap)
            mask = sigmoid(mask, overlap)
            mask = mask.reshape(overlap, 1, 1)
            mask = np.ones((overlap, output_shape[1], output_shape[2])) * mask
            mask = np.expand_dims(mask, -1)
            mask = np.repeat(mask, k, axis=-1)
            merged_xyz[step[0]*i:step[0]*i+overlap, :, :] = (1 - mask) * overlap1 + mask * overlap2
            merged_xyz[step[0]*i+overlap:step[0]*i+patch_size[0], :, :] = img2[overlap:, :, :]
        else:
            merged_xyz[step[0]*i:step[0]*i+patch_size[0], :, :] = img2
    return merged_xyz

def pad_img(img, patch_size, step, mode="wrap"):
    import sympy as sym
    from math import ceil
    import numpy as np

	
    def LS(patch_size, step, size):
        # returns the final length for one dimension
        k = max(0, ceil((size - patch_size) / step))
        return int(patch_size + k * step)

    LS_img = [LS(patch_size[i], step[i], img.shape[i]) for i in range(img.ndim)]

    # Aplicar el padding calculado a la imagen
    img = np.pad(
        img,
        pad_width=[(0, max(0, LS_img[i] - img.shape[i])) for i in range(img.ndim)],
        mode=mode,
    )

    return img


def unpad_img(img, original_shape):
    return img[0:original_shape[0], 0:original_shape[1], 0:original_shape[2]]

def double_unpad_img(img_padded, pad_size):
    return img_padded[pad_size:-pad_size, pad_size:-pad_size, pad_size:-pad_size]

def normalize(img):
    return (img-img.min())/(img.max()-img.min())

def parallel_resize_3d(img, output_shape, order=3, n_jobs=cpu_count() // 2):
    input_shape = np.array(img.shape)
    output_shape = np.array(output_shape)
    factors = output_shape / input_shape
    result = np.copy(img)
    
    if factors[0] != 1:
        result = Parallel(n_jobs=n_jobs)(
            delayed(resize)(result[:, i], (output_shape[0], result.shape[2]), order=order)
            for i in range(result.shape[1])
        )
        result = np.stack(result, axis=1)
        
    if factors[1] != 1:
        result = Parallel(n_jobs=n_jobs)(
            delayed(resize)(result[i], (output_shape[1], result.shape[2]), order=order)
            for i in range(result.shape[0])
        )
        result = np.stack(result, axis=0)

    if factors[2] != 1:
        result = Parallel(n_jobs=n_jobs)(
            delayed(resize)(result[i], (result.shape[1], output_shape[2]), order=order)
            for i in range(result.shape[0])
        )
        result = np.stack(result, axis=0)

    return result