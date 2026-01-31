#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:01:13 2025

@author: ws4
"""

import tensorflow as tf
import h5py
import numpy as np
from tifffile import imread
import config
from keras_models.nisnet import encode, get_object_code
from preprocessing.data_augmentation import aug_batch


class DataLoader:
    def __init__(
        self,
        batch_size,
        image_shape=config.INPUT_SHAPE[:-1],
        num_classes=config.NUM_CLASSES,
    ):
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.num_classes = num_classes

    def normalize(self, img):
        return (img - img.min()) / (img.max() - img.min())

    def process_image(self, img):
        img = self.normalize(img)
        img = np.expand_dims(img, axis=-1)
        return img

    def read_img(self, path):
        file = imread(path.decode("utf-8"))
        file = self.process_image(file)
        return tf.cast(file, tf.float32)

    def read_lbl(self, path):
        file = imread(path.decode("utf-8"))
        return tf.cast(file, tf.float32)

    def target_vector_binary(self, lbls, small_obj_vox=config.SMALL_OBJECT_VOX):
        batch_size = lbls.shape[0]
        target_binary_list = []
        target_vector_list = []

        for i in range(batch_size):
            lbl = lbls[i]
            img_ctrs = get_object_code(lbl, vox_threshold=small_obj_vox)
            target = encode(
                img_ctrs, lbl, *lbl.shape, None, self.num_classes, small_obj_vox
            )
            target_vector, target_binary = target[:3], target[3:-1]
            target_vector = np.transpose(target_vector, (1, 2, 3, 0))

            target_vector_norm = np.linalg.norm(target_vector, axis=-1, keepdims=True)
            target_vector_mask = (target_vector_norm>0)[...,0]
            target_vector[target_vector_mask] = target_vector[target_vector_mask]/target_vector_norm[target_vector_mask]
            
            target_binary = np.transpose(target_binary, (1, 2, 3, 0))
            target_binary_list.append(target_binary)
            target_vector_list.append(target_vector)

        target_binary_batch = tf.cast(np.stack(target_binary_list, axis=0), tf.float32)
        target_vector_batch = tf.cast(np.stack(target_vector_list, axis=0), tf.float32)

        return target_binary_batch, target_vector_batch

    def process_image_tf(self, img, lbl):
        img = tf.numpy_function(func=self.process_image, inp=[img], Tout=tf.float32)
        lbl = tf.cast(lbl, tf.float32)
        return img, lbl

    def apply_label_vector(self, img, lbl):
        lbl, vec = tf.numpy_function(
            func=self.target_vector_binary, inp=[lbl], Tout=[tf.float32, tf.float32]
        )
        return img, lbl, vec

    @tf.autograph.experimental.do_not_convert
    def aug_data_nisnet(self, img, lbl):
        img, lbl = tf.numpy_function(
            func=aug_batch, inp=[img, lbl], Tout=[tf.float32, tf.float32]
        )
        return img, lbl

    @tf.autograph.experimental.do_not_convert
    def _fixup_shape(self, img, lbl, grad):
        # Ajustar la forma de las imágenes dinámicamente según image_shape
        img.set_shape([None] + list(self.image_shape) + [1])
        # Ajustar la forma de las etiquetas dinámicamente según image_shape y num_classes
        lbl.set_shape([None] + list(self.image_shape) + [self.num_classes])
        # Ajustar la forma de los pesos dinámicamente según image_shape
        grad.set_shape([None] + list(self.image_shape) + [3])
        return img, (lbl, grad)

    def hdf5_data_generator(self, hdf5_file_path):
        with h5py.File(hdf5_file_path, "r") as hf:
            images, labels = hf["images"], hf["labels"]
            for img, lbl in zip(images, labels):
                yield img, lbl

    def create_hdf5_dataset(self, hdf5_file_path):
        dataset = tf.data.Dataset.from_generator(
            lambda: self.hdf5_data_generator(hdf5_file_path),
            output_signature=(
                tf.TensorSpec(shape=self.image_shape, dtype=tf.float32),
                tf.TensorSpec(shape=self.image_shape, dtype=tf.float32),
            ),
        )
        return dataset

    def get_dataset(self, hdf5_file_path, augment=True, step_train_size=None):
        dataset = self.create_hdf5_dataset(hdf5_file_path)
        dataset = dataset.map(
            self.process_image_tf, num_parallel_calls=tf.data.AUTOTUNE
        )
        if augment:
            dataset = dataset.shuffle(buffer_size=1000)
        if step_train_size:
            dataset = dataset.take(step_train_size * self.batch_size)
        dataset = dataset.cache()
        dataset = dataset.batch(self.batch_size)
        if augment:
            dataset = dataset.map(
                self.aug_data_nisnet, num_parallel_calls=tf.data.AUTOTUNE
            )
        dataset = dataset.map(
            self.apply_label_vector, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(self._fixup_shape, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
