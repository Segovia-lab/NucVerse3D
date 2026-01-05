#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:53:25 2024

@author: ws2
"""

import numpy as np
from scipy.ndimage import affine_transform
import elasticdeform
import multiprocessing as mp
import skimage.io as io

# import utils.array_tool as at
from skimage.filters import gaussian
from skimage.util import random_noise
from skimage.exposure import equalize_adapthist
from skimage.transform import rotate as skimage_rotate
from scipy.ndimage import rotate as scipy_rotate
from skimage.transform import resize
import random
import config

def downsample_image(X, y):
    s = X.shape
    random_value_0_1 = random.choices([0, 1], k=3)
    random_value_1_2_4 = [2 if random_value_0_1[i] != 0 else 1 for i in range(3)]

    X_resized = resize(
        resize(
            X,
            (
                s[0] // random_value_1_2_4[0],
                s[1] // random_value_1_2_4[1],
                s[2] // random_value_1_2_4[2],
            ),
            order=0,
        ),
        s,
        order=0,
    )
    return X_resized, y


def flip3D(X, y):
    """
    Flip the 3D image respect one of the 3 axis chosen randomly
    """
    choice = np.random.randint(3)
    if choice == 0:  # flip on x
        X_flip, y_flip = X[::-1, :, :, :], y[::-1, :, :]
    if choice == 1:  # flip on y
        X_flip, y_flip = X[:, ::-1, :, :], y[:, ::-1, :]
    if choice == 2:  # flip on z
        X_flip, y_flip = X[:, :, ::-1, :], y[:, :, ::-1]
    return X_flip, y_flip


def transpose3D(X, y):
    """
    Transpose the 3D image respect one of the 3 axis chosen randomly
    """
    p = np.random.permutation(2) + 1
    X_trans, y_trans = np.transpose(X, (0, p[0], p[1], 3)), np.transpose(
        y, (0, p[0], p[1])
    )
    return X_trans, y_trans


def brightness(X, y):
    """
    Changing the brighness of a image using power-law gamma transformation.
    Gain and gamma are chosen randomly for each image channel.

    Gain chosen between [0.8 - 1.2]
    Gamma chosen between [0.8 - 1.2]

    new_im = gain * im^gamma
    """
    print("image shape: ", X.shape)
    X_new = np.zeros(X.shape)
    for c in range(X.shape[-1]):
        im = X[:, :, :, c]
        gain, gamma = (1.5 - 0.5) * np.random.random_sample(
            2,
        ) + 0.5
        im_new = np.sign(im) * gain * (np.abs(im) ** gamma)
        if np.min(im_new) < 0:
            print("something wrong")
        max_val = np.max(im_new)
        if max_val > 1.0:
            im_new = im_new / max_val
        X_new[:, :, :, c] = im_new

    return X_new, y

def brightness_new(
    X,
    y,
    gain_range=config.BRIGHTNESS_RANGE,
    gamma_range=config.BRIGHTNESS_RANGE,
    seed=config.RANDOM_SEED,
    keep_unit_range=False,
):
    """
    Apply per-channel random power-law (gamma) brightness transform on a 3D image.
    Works with (Z, Y, X, C) or (Z, Y, X) arrays. Channels are processed independently.

    new = sign(im) * gain * |im|**gamma

    Args
    ----
    X : np.ndarray
        3D/3D+channels image. Expected shape (Z, Y, X, C) or (Z, Y, X).
        Values can be any dtype; will compute in float32.
    y : mask same dimension as X
    gain_range : (float, float)
        Uniform range for multiplicative gain per channel.
    gamma_range : (float, float)
        Uniform range for gamma exponent per channel.
    seed : int or None
        Seed for reproducibility.
    keep_unit_range : bool
        If True, after transforming each channel, rescale by its max so the
        channel stays <= 1.0 when it would otherwise exceed it. This mimics the
        original behavior without assuming 8-bit. If your data are not in [0, 1],
        set this to False to keep the original numeric scale.

    Returns
    -------
    X_new : np.ndarray
        Transformed image with same shape and dtype as X (cast back, clipped if int).
    y : any
        Unchanged.
    """
    if X.ndim not in (3, 4):
        raise ValueError(f"X must be (Z,Y,X) or (Z,Y,X,C); got shape {X.shape}")

    # Ensure channels-last 4D
    has_channel = (X.ndim == 4)
    if not has_channel:
        X = X[..., np.newaxis]  # (Z,Y,X,1)

    orig_dtype = X.dtype
    Xf = X.astype(np.float32, copy=False)
    Z, Y, W, C = Xf.shape

    rng = np.random.default_rng(None if seed <= 0 else seed)
    X_new = np.empty_like(Xf)

    # Precompute dtype bounds if integer
    lo, hi = 0.0, 1.0

    for c in range(C):
        im = Xf[..., c]

        gain = rng.uniform(gain_range[0], gain_range[1])
        gamma = rng.uniform(gamma_range[0], gamma_range[1])

        # Support for signed data: apply gamma to magnitude, then reapply sign.
        im_new = np.sign(im) * gain * (np.abs(im) ** gamma)

        # Optional unit-range safeguard (does NOT assume 8-bit; it only avoids overshoot)
        if keep_unit_range:
            max_val = np.max(np.abs(im_new))  # use abs to be safe with signed data
            if max_val > 1.0 and np.isfinite(max_val) and max_val > 0:
                im_new = im_new / max_val

        X_new[..., c] = im_new

    # Clip to dtype range before casting back
    X_new = np.clip(X_new, lo, hi).astype(orig_dtype, copy=False)
 

    if not has_channel:
        X_new = X_new[..., 0]

    return X_new, y




def zoom3D(X, y):
    """
    Apply zoom to a 3D image. The zoom factors for each axis are chosen randomly between 0.8 and 1.2.
    """
    a, b = 0.8, 1.2
    alpha, beta, gamma = (b - a) * np.random.random_sample(3) + a
    R_scale = np.array([[alpha, 0, 0], [0, beta, 0], [0, 0, gamma]])
    X_zoom = np.empty_like(X)
    y_zoom = np.empty_like(y)

    for channel in range(X.shape[-1]):
        X_zoom[:, :, :, channel] = affine_transform(
            X[:, :, :, channel], R_scale, offset=0, order=1, mode="constant"
        )

    y_zoom = affine_transform(y, R_scale, offset=0, order=0, mode="constant")

    return X_zoom, y_zoom


def zoom3D_new(X, y):
    """
    Apply zoom to a 3D image and adjust it to match the original shape.
    If the zoomed shape is smaller, padding is added.
    If the zoomed shape is larger, a random crop is applied.

    Parameters:
        img (np.ndarray): Original 3D image.
        lbl (np.ndarray): Corresponding label 3D image.

    Returns:
        tuple: Adjusted 3D image and label after zoom and adjustment.
    """
    a, b = config.ZOOM_RANGE #0.8, 1.2
    alpha, beta, gamma = (b - a) * np.random.random_sample(3) + a

    X_zoom = np.empty_like(X)
    y_zoom = np.empty_like(y)

    # Calculate new dimensions
    depth, height, width = X.shape[:-1]
    new_depth = np.round(alpha * depth).astype(int)
    new_height = np.round(beta * height).astype(int)
    new_width = np.round(gamma * width).astype(int)

    current_shape = (new_depth, new_height, new_width)
    original_shape = (depth, height, width)

    def adjust_shape(volume, current_shape, original_shape):
        adjusted_volume = volume.copy()
        for axis, (current_dim, original_dim) in enumerate(
            zip(current_shape, original_shape)
        ):
            if current_dim < original_dim:
                # Padding if the dimension is smaller
                pad_before = (original_dim - current_dim) // 2
                pad_after = original_dim - current_dim - pad_before
                pad_width = [(0, 0)] * len(current_shape)
                pad_width[axis] = (pad_before, pad_after)
                adjusted_volume = np.pad(
                    adjusted_volume, pad_width, mode="constant", constant_values=0
                )
            elif current_dim > original_dim:
                # Crop if the dimension is larger
                start = np.random.randint(0, current_dim - original_dim + 1)
                end = start + original_dim
                slices = [slice(None)] * len(current_shape)
                slices[axis] = slice(start, end)
                adjusted_volume = adjusted_volume[tuple(slices)]
        return adjusted_volume

    for channel in range(X.shape[-1]):
        X_zoom_channel = resize(
            X[..., channel], current_shape, order=1, mode="constant"
        )
        X_zoom[..., channel] = adjust_shape(
            X_zoom_channel, current_shape, original_shape
        )
    y_zoom = resize(y, current_shape, order=0, mode="constant")
    y_zoom = adjust_shape(y_zoom, current_shape, original_shape)
    return X_zoom, y_zoom


def rotate3D(X, y):
    """
    Rotate a 3D image with a random angle between 0 and 45 degrees around the Z-axis.
    The image has shape [Z, Y, X]. The rotation is performed around the center of the image.
    """

    max_angle = 45
    alpha = np.random.random_sample() * max_angle
    X_rot = np.array([skimage_rotate(x, alpha) for x in X])
    y_rot = np.array([skimage_rotate(x, alpha, order=0) for x in y])
    return X_rot, y_rot


def rotate3D_new(X, y, max_angle=45):
    """
    Rotate a 3D image and label by a random angle.

    Parameters:
        X (np.ndarray): Input 3D image (depth, height, width, channels).
        y (np.ndarray): Input 3D label (depth, height, width).
        max_angle (float): Maximum rotation angle in degrees.

    Returns:
        tuple: Rotated 3D image and label.
    """
    alpha = np.random.random_sample() * max_angle
    axes = (1, 2)  # Rotate around the depth axis (height, width plane)

    # Rotate the entire 3D volume at once
    X_rot = np.array(
        [
            scipy_rotate(
                X[..., channel],
                alpha,
                axes=axes,
                reshape=False,
                order=1,
                mode="constant",
            )
            for channel in range(X.shape[-1])
        ]
    )
    X_rot = np.moveaxis(X_rot, 0, -1)  # Move channels back to the last axis
    y_rot = scipy_rotate(y, alpha, axes=axes, reshape=False, order=0, mode="constant")

    return X_rot, y_rot


def elastic(X, y):
    """
    Elastic deformation on a image and its target
    """
    [Xel, yel] = elasticdeform.deform_random_grid(
        [X, y],
        points=4,
        sigma=2,
        axis=[(0, 1, 2), (0, 1, 2)],
        order=[1, 0],
        mode="constant",
    )

    return Xel, yel


def blur(X, y):
    """
    Gaussian blur
    """
    sigma = np.random.random()  # 0-1
    X = gaussian(X, preserve_range=True, sigma=sigma)
    return X, y


def add_noise(X, y):
    newX = random_noise(X, mode="poisson", clip=True)
    return newX, y


def random_decisions(N):
    """
    Generate N random decisions for augmentation
    N should be equal to the batch size
    """
    aug_num = 9
    decisions = np.zeros(
        (N, aug_num)
    )  # 9 is number of aug techniques to combine (patch extraction is not used here)
    for n in range(N):
        decisions[n] = np.random.randint(2, size=aug_num)

    return decisions

import tifffile as tiff

def combine_aug(X, y, do):
    """
    Combine randomly the different augmentation techniques written above
    """
    Xnew, ynew = X, y
    # do = [0,1,0,0,0,0,0,0]
    #print("do: ", do)
    # make sure to use at least 25% of original images
    if np.random.random_sample() > config.AUGMENTATION_PROBABILITY:
        return Xnew, ynew
    else:
        Xnew = np.float32(Xnew)
        if do[0] == 1:
            Xnew, ynew = downsample_image(Xnew, ynew)

        if do[1] == 1:
            Xnew, ynew = flip3D(Xnew, ynew)

        if do[2] == 1:
            Xnew, ynew = transpose3D(Xnew, ynew)  # make sure your input is a cube

        if do[3] == 1:
            Xnew, ynew = brightness_new(Xnew, ynew)

        if do[4] == 1:
            Xnew, ynew = rotate3D_new(Xnew, ynew)

        if do[5] == 1:
            Xnew, ynew = zoom3D_new(Xnew, ynew)

        if do[6] == 1:
            Xnew, ynew = elastic(Xnew, ynew)

        if do[7] == 1:
            Xnew, ynew = add_noise(Xnew, ynew)

        if do[8] == 1:
            Xnew, ynew = blur(Xnew, ynew)

        return Xnew, ynew


def aug_batch(Xb, Yb):
    """
    Generate an augmented image batch
    """
    batch_size = len(Xb)
    newXb, newYb = np.empty_like(Xb), np.empty_like(Yb)

    decisions = random_decisions(batch_size)
    for i in range(batch_size):

        newXb[i], newYb[i] = combine_aug(Xb[i], Yb[i], decisions[i])
        #print("image", i)
        #tiff.imwrite(f"input_{i}.tif", Xb[i])
        #tiff.imwrite(f"output_{i}_{sum(decisions[i])}.tif", newXb[i])

    return newXb, newYb


def aug_stardist(Xb, Yb):
    decisions = random_decisions(1)[0]
    newXb, newYb = combine_aug(Xb[..., np.newaxis], Yb, decisions)
    return newXb[..., 0], newYb
