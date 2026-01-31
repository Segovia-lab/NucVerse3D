import numpy as np


def interp_X(img, output_shape, patch_size, step, overlap, z, j):
    merged = np.zeros((patch_size, patch_size, output_shape[-1]), img.dtype)
    for i in range(len(img[0, 0])):
        img1 = merged.copy()
        img2 = img[z, j, i]

        if np.logical_and(img1[:, :, step * i : step * i + patch_size], img2).sum():
            overlap1 = img1[:, :, step * i : step * i + overlap]
            overlap2 = img2[:, :, :overlap]
            mask = np.linspace(0, 1, overlap)
            mask = sigmoid(mask, overlap)
            mask = np.ones((patch_size, patch_size, overlap)) * mask
            merged[:, :, step * i : step * i + overlap] = (
                1 - mask
            ) * overlap1 + mask * overlap2
            merged[:, :, step * i + overlap : step * i + patch_size] = img2[
                :, :, overlap:
            ]
        else:
            merged[:, :, step * i : step * i + patch_size] = img2
    return merged


def interp_Y(img, output_shape, patch_size, step, overlap, z, merged_x):
    merged_xy = np.zeros((patch_size, output_shape[-2], output_shape[-1]), img.dtype)
    for i in range(len(img[0])):
        img1 = merged_xy.copy()
        img2 = merged_x[z, i]
        if np.logical_and(img1[:, step * i : step * i + patch_size, :], img2).sum():
            overlap1 = img1[:, step * i : step * i + overlap, :]
            overlap2 = img2[:, :overlap, :]
            mask = np.linspace(0, 1, overlap)
            mask = sigmoid(mask, overlap)
            mask = mask.reshape(overlap, 1)
            mask = np.ones((patch_size, overlap, output_shape[-1])) * mask
            merged_xy[:, step * i : step * i + overlap, :] = (
                1 - mask
            ) * overlap1 + mask * overlap2
            merged_xy[:, step * i + overlap : step * i + patch_size, :] = img2[
                :, overlap:, :
            ]
        else:
            merged_xy[:, step * i : step * i + patch_size, :] = img2
    return merged_xy


def sigmoid(x, k):
    return 1 / (1 + np.exp(-k // 2 * (x - 0.5)))


def linear_interp_vol(img, output_shape, patch_size, step):
    # La general del algoritmo es mediante interolaciones lineales en un cada eje ir uniendo los parches
    # para lograr una reconstruccion mucho mas limpia. Para esto se agregan:
    # img: la imagen 3D en formato (Z,Y,X, patch_size, patch_size, patch_size)
    # output_shape: la forma de la imagen en la salida (para saber de que tamaño seran los parches cuando se unen)
    # patch_size: el tamaño del parche (igual en todos los ejes)
    # step: son los pasos entre cada  parche (idealmente menor al parche)

    overlap = patch_size - step

    # Interpolacion en X
    merged_x = []
    for z in range(len(img)):
        for j in range(len(img[0])):
            merged_x.append(
                interp_X(img, output_shape, patch_size, step, overlap, z, j)
            )
    merged_x = np.reshape(
        merged_x, (img.shape[0], img.shape[1], patch_size, patch_size, output_shape[-1])
    )

    # Interpolacion en Y
    merged_y = []
    for z in range(len(img)):
        merged_y.append(
            interp_Y(img, output_shape, patch_size, step, overlap, z, merged_x)
        )

    # Interpolacion en Z
    merged_xyz = np.zeros(output_shape, img.dtype)
    for i in range(len(img)):
        img1 = merged_xyz.copy()
        img2 = merged_y[i]
        if np.logical_and(img1[step * i : step * i + patch_size, :, :], img2).sum():
            overlap1 = img1[step * i : step * i + overlap, :, :]
            overlap2 = img2[:overlap, :, :]
            mask = np.linspace(0, 1, overlap)
            mask = sigmoid(mask, overlap)
            mask = mask.reshape(overlap, 1, 1)
            mask = np.ones((overlap, output_shape[-2], output_shape[-1])) * mask
            merged_xyz[step * i : step * i + overlap, :, :] = (
                1 - mask
            ) * overlap1 + mask * overlap2
            merged_xyz[step * i + overlap : step * i + patch_size, :, :] = img2[
                overlap:, :, :
            ]
        else:
            merged_xyz[step * i : step * i + patch_size, :, :] = img2
    return merged_xyz
