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
from skimage.transform import rotate

from skimage.transform import resize
import random
def downsample_image(X, y):
    s = X.shape    
    random_value_0_1 = random.choices([0, 1], k=3)
    random_value_1_2_4 = [2 if random_value_0_1[i]!=0 else 1 for i in range(3)]
   
    X_resized = resize(resize(X, (s[0]//random_value_1_2_4[0],s[1]//random_value_1_2_4[1],s[2]//random_value_1_2_4[2]), order=0), s, order=0)
    return X_resized, y

def flip3D(X, y):
    """
    Flip the 3D image respect one of the 3 axis chosen randomly
    """
    choice = np.random.randint(3)
    if choice == 0: # flip on x
        X_flip, y_flip = X[::-1, :, :, :], y[::-1, :, :]
    if choice == 1: # flip on y
        X_flip, y_flip = X[:, ::-1, :, :], y[:, ::-1, :]
    if choice == 2: # flip on z
        X_flip, y_flip = X[:, :, ::-1, :], y[:, :, ::-1]
    return X_flip, y_flip

def transpose3D(X, y):
    """
    Transpose the 3D image respect one of the 3 axis chosen randomly
    """
    p = np.random.permutation(2) + 1
    X_trans, y_trans = np.transpose(X, (0, p[0], p[1], 3)), np.transpose(y, (0, p[0], p[1]))
    return X_trans, y_trans

def brightness(X, y):
    """
    Changing the brighness of a image using power-law gamma transformation.
    Gain and gamma are chosen randomly for each image channel.
    
    Gain chosen between [0.8 - 1.2]
    Gamma chosen between [0.8 - 1.2]
    
    new_im = gain * im^gamma
    """
    
    X_new = np.zeros(X.shape)
    for c in range(X.shape[-1]):
        im = X[:,:,:,c]        
        gain, gamma = (1.5 - 0.5) * np.random.random_sample(2,) + 0.5
        im_new = np.sign(im)*gain*(np.abs(im)**gamma)
        if np.min(im_new)<0:
            print("something wrong")
        max_val = np.max(im_new)
        if max_val>1.0:
            im_new = im_new/max_val
        X_new[:,:,:,c] = im_new 
    
    return X_new, y


# def rotation_zoom3D(X, y):
#     """
#     Rotate a 3D image with alfa, beta and gamma degree respect the axis x, y and z respectively.
#     The three angles are chosen randomly between 0-30 degrees
#     """
#     alpha, beta, gamma = np.random.random_sample(3)*np.pi/6
#     Rx = np.array([[1, 0, 0],
#                     [0, np.cos(alpha), -np.sin(alpha)],
#                     [0, np.sin(alpha), np.cos(alpha)]])
   
#     Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
#                     [0, 1, 0],
#                     [-np.sin(beta), 0, np.cos(beta)]])
  
#     Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
#                     [np.sin(gamma), np.cos(gamma), 0],
#                     [0, 0, 1]])
    
#     R_rot = np.dot(np.dot(Rx, Ry), Rz)
    
#     a, b = 0.6, 1.4
#     alpha, beta, gamma = (b-a)*np.random.random_sample(3) + a
#     R_scale = np.array([[alpha, 0, 0],
#                     [0, beta, 0],
#                     [0, 0, gamma]])
    
#     R = np.dot(R_rot, R_scale)
#     X_rot = np.empty_like(X)
#     y_rot = np.empty_like(y)

#     for channel in range(X.shape[-1]):
#         X_rot[:,:,:,channel] = affine_transform(X[:,:,:,channel], R, offset=0, order=1, mode='constant')
#     y_rot = affine_transform(y, R, offset=0, order=0, mode='constant')
#     return X_rot, y_rot

def zoom3D(X, y):
    """
    Apply zoom to a 3D image. The zoom factors for each axis are chosen randomly between 0.8 and 1.2.
    """
    a, b = 0.8, 1.2
    alpha, beta, gamma = (b-a)*np.random.random_sample(3) + a
    R_scale = np.array([[alpha, 0, 0],
                    [0, beta, 0],
                    [0, 0, gamma]])
    X_zoom = np.empty_like(X)
    y_zoom = np.empty_like(y)
    
    for channel in range(X.shape[-1]):
        X_zoom[:,:,:,channel] = affine_transform(X[:,:,:,channel], R_scale, offset=0, order=1, mode='constant')
    
    y_zoom = affine_transform(y, R_scale, offset=0, order=0, mode='constant')
    
    return X_zoom, y_zoom


def rotate3D(X, y):
    """
    Rotate a 3D image with a random angle between 0 and 45 degrees around the Z-axis.
    The image has shape [Z, Y, X]. The rotation is performed around the center of the image.
    """
    max_angle = 45
    alpha = np.random.random_sample()*max_angle
    X_rot = np.array([rotate(x, alpha) for x in X])
    y_rot = np.array([rotate(x, alpha, order=0) for x in y])
    return X_rot, y_rot

def elastic(X, y):
    """
    Elastic deformation on a image and its target
    """
    [Xel, yel] = elasticdeform.deform_random_grid([X, y], points=4, sigma=2, axis=[(0, 1, 2), (0, 1, 2)], order=[1, 0], mode='constant')
    
    return Xel, yel

def blur(X, y):
    """
    Gaussian blur
    """
    sigma = np.random.random()    # 0-1
    X = gaussian(X, preserve_range = True, sigma=sigma)
    return X, y

def add_noise(X,y):
    newX = random_noise(X, mode="poisson", clip=True)
    return newX, y

def random_decisions(N):
    """
    Generate N random decisions for augmentation
    N should be equal to the batch size
    """
    aug_num = 9
    decisions = np.zeros((N, aug_num)) # 5 is number of aug techniques to combine (patch extraction is not used here)
    for n in range(N):
        decisions[n] = np.random.randint(2, size=aug_num)
        
    return decisions

def combine_aug(X, y, do):
    """
    Combine randomly the different augmentation techniques written above
    """
    Xnew, ynew = X, y
    # do = [0,1,0,0,0,0,0,0]
    # make sure to use at least 25% of original images
    # print(do)
    if np.random.random_sample()>0.75:
        return Xnew, ynew
    else:
        Xnew = np.float32(Xnew)
        if do[0] == 1:
            Xnew, ynew = downsample_image(Xnew, ynew)
        
        if do[1] == 1:
            Xnew, ynew = flip3D(Xnew, ynew)

        if do[2] == 1:
            Xnew, ynew = transpose3D(Xnew, ynew)    # make sure your input is a cube

        if do[3] == 1:
            Xnew, ynew = brightness(Xnew, ynew)   

        if do[4] == 1:
            Xnew, ynew = rotate3D(Xnew, ynew)
        
        if do[5] == 1:
            Xnew, ynew = zoom3D(Xnew, ynew)
            
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
        
    return newXb, newYb