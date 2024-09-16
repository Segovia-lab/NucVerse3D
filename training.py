# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:45:41 2024

@author: ws2
"""

#%%
import os
import tensorflow as tf

os.chdir(os.path.dirname(os.path.abspath(__file__)))

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#%%
import sys
import numpy as np
from glob import glob
from tifffile import imread
from utils.nisnet import encode, get_object_code
from utils.data_augmentation import aug_batch
import matplotlib.pyplot as plt
#%%
def load_data(path):
    images = sorted(glob(os.path.join(path, "images/*.[tT][iI][fF]*")))
    labels = sorted(glob(os.path.join(path, "labels/*.[tT][iI][fF]*")))
    return images, labels

def target_vector_binary(lbls, small_obj_vox=100, num_classes=2):
    batch_size = lbls.shape[0]
    target_binary_list = []
    target_vector_list = []
    
    for i in range(batch_size):
        lbl = lbls[i]
        img_ctrs = get_object_code(lbl, vox_threshold=small_obj_vox)
        target = encode(img_ctrs, lbl, lbl.shape[0], lbl.shape[1], lbl.shape[2], None, num_classes, small_obj_vox)
        target_vector, target_binary = target[:3], target[3:-1]
        target_vector = np.transpose(target_vector, (1, 2, 3, 0))
        target_binary = np.transpose(target_binary, (1, 2, 3, 0))
        target_binary_list.append(target_binary)
        target_vector_list.append(target_vector)

    target_binary_batch = np.stack(target_binary_list, axis=0)
    target_binary_batch = tf.cast(target_binary_batch, tf.float32)
    
    target_vector_batch = np.stack(target_vector_list, axis=0)
    target_vector_batch = tf.cast(target_vector_batch, tf.float32)
    
    return target_binary_batch, target_vector_batch


def apply_label_vector(img, lbl):
    lbl, vec = tf.numpy_function(func=target_vector_binary, inp=[lbl], Tout=[tf.float32, tf.float32])  # Corregir el inp
    return img, lbl, vec

def normalize(img):
    return (img-img.min())/(img.max()-img.min())


def read_img(path):
    file_path = path.decode('utf-8')
    file = imread(file_path)
    file = normalize(file)
    file = np.expand_dims(file, axis=-1)
    file = tf.cast(file, tf.float32)
    return file


def read_lbl(path):
    file_path = path.decode('utf-8')
    file = imread(file_path)
    file = tf.cast(file, tf.float32)
    return file

# Función que normaliza y expande las dimensiones de la imagen
def process_image(image):
    image = normalize(image)
    image = np.expand_dims(image, axis=-1)
    return image

def process_image_tf(img, lbl):
    img = tf.numpy_function(func=process_image, inp=[img], Tout=tf.float32)
    lbl = tf.cast(lbl, tf.float32)
    return img, lbl

def read_data_train(img, lbl):
    img = tf.numpy_function(func=read_img, inp=[img], Tout=tf.float32)
    lbl = tf.numpy_function(func=read_lbl, inp=[lbl], Tout=tf.float32)
    return img, lbl


def read_data_val(img, lbl):
    img = tf.numpy_function(func=read_img, inp=[img], Tout=tf.float32)
    lbl = tf.numpy_function(func=read_lbl, inp=[lbl], Tout=tf.float32)
    return img, lbl

def aug_data_nisnet(img, lbl):
    img, lbl = tf.numpy_function(func=aug_batch, inp=[img, lbl], Tout=[tf.float32, tf.float32])
    return img, lbl


def _fixup_shape(img, lbl, grad):
    # Ajustar la forma de las imágenes
    img.set_shape([None, 64, 128, 128, 1])
    # Ajustar la forma de las etiquetas
    lbl.set_shape([None, 64, 128, 128, 2])
    # Ajustar la forma de los pesos
    grad.set_shape([None, 64, 128, 128, 3])
    return img, (lbl, grad)

# %%
import h5py
# Función para crear un generador que lee los datos por lotes desde el archivo HDF5
def hdf5_data_generator(hdf5_file_path, batch_size):
    with h5py.File(hdf5_file_path, 'r') as hf:
        num_samples = hf['images'].shape[0]
        
        # Leer el batch directamente sin barajado

        images = hf['images']
        labels = hf['labels']
        
        for i in range(len(images)):
           yield images[i], labels[i]

# Función para crear el dataset desde un generador
def create_hdf5_dataset(hdf5_file_path):
    dataset = tf.data.Dataset.from_generator(
        lambda: hdf5_data_generator(hdf5_file_path, batch_size),
        output_signature=(
            tf.TensorSpec(shape=(64, 128, 128), dtype=tf.float32),  # Imagen
            tf.TensorSpec(shape=(64, 128, 128), dtype=tf.float32)   # Etiqueta
        )
    )
    
    return dataset

batch_size = 2
training_path = "./training sets/training_overlape_32/data_train.hdf5"
train_dataset = create_hdf5_dataset(training_path)
train_dataset = train_dataset.map(process_image_tf, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(buffer_size=1000)  # Buffer para barajar
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.map(aug_data_nisnet, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(apply_label_vector, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(_fixup_shape, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

#%%
# Obtener un lote del dataset
sample_batch = train_dataset.take(1)
for i in sample_batch:
    print(i[0].numpy().shape, i[1][0].numpy().shape, i[1][1].numpy().shape)
    plt.imshow(i[0].numpy()[0,...,0].max(axis = 0))

# %%

test_path = "./training sets/training_overlape_32/data_val.hdf5"
test_dataset = create_hdf5_dataset(test_path)
test_dataset = test_dataset.map(process_image_tf, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.cache()
test_dataset = test_dataset.map(apply_label_vector, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.map(_fixup_shape, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

#%%
sample_batch = test_dataset.take(1)
for i in sample_batch:
    print(i[0].numpy().shape, i[1][0].numpy().shape, i[1][1].numpy().shape)


#%% Training
from utils.get_model_memory_usage import get_model_memory_usage
from utils.loss_functions import FocalTverskyLoss, MSE_loss, dice_coef
from utils.models_3d_attention import Attention_ResUNet_3D
from tensorflow.keras.optimizers import Adam
metrics = ["mae", dice_coef]
model = Attention_ResUNet_3D((64, 128, 128, 1), 2, 1, dropout_rate=0.4, FILTER_NUM=32)
model.compile(optimizer=Adam(0.0001), loss=[FocalTverskyLoss(), MSE_loss], metrics=metrics)
print(model.summary())
print(get_model_memory_usage(batch_size, model))
#%%
from utils.get_model_memory_usage import get_model_memory_usage
from utils.loss_functions import FocalTverskyLoss, MSE_loss, dice_coef
from utils.models_3d_attention import Attention_ResUNet_3D
from tensorflow.keras.optimizers import Adam
folder = '4. 3d raunet 64_128 dropout 0.4 filters 32 oversampling'
folder_path = os.path.join('./models_training', folder)
model = tf.keras.models.load_model(os.path.join(folder_path, 'modelSave.h5'), compile = False)

metrics = ["mae", dice_coef]
model.compile(optimizer=Adam(0.0001), loss=[FocalTverskyLoss(), MSE_loss], metrics=metrics)
best_model = 'best_model.hdf5'
best_model_path = os.path.join(folder_path,best_model)
model.load_weights(best_model_path)
# %%
def create_path(path):
    if not os.path.exists(path):
        # Create the directory
        os.makedirs(path)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
        
base_training_models = './models_training'
create_path(base_training_models)
#%%
from tensorflow.keras.callbacks import ModelCheckpoint
import os 
folder = '8. 3d raunet overlape 32 dataet total 64_128 dropout 0.4 filters 32'
folder_path = os.path.join(base_training_models, folder)
create_path(folder_path)
best_model = 'best_model.hdf5'
best_model_path = os.path.join(folder_path,best_model)

checkpoint = ModelCheckpoint(
    filepath=best_model_path,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

# %%
history = model.fit(
    train_dataset,
    epochs=200,
    validation_data=test_dataset,
    callbacks=checkpoint)
# %%
import pandas as pd
df = pd.DataFrame(history.history)
history_path = 'history.csv'
df.to_csv(os.path.join(folder_path, history_path))

# %%
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save(os.path.join(folder_path,'modelSave.h5'))