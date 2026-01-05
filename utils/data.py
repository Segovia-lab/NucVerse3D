import numpy as np
from scipy.ndimage.interpolation import affine_transform
import elasticdeform
import multiprocessing as mp
import skimage.io as io

# import utils.array_tool as at  # Module not found
from skimage.filters import gaussian
from skimage.util import random_noise
from skimage.exposure import equalize_adapthist


############# Utility Functions ##############
def normalize(vol):
    # stretch contrast to 0-255
    max_val = np.amax(vol)
    min_val = np.amin(vol)
    vol = (vol - min_val) / (max_val - min_val + 1e-9) * 255
    return np.uint8(vol)


def prepare(vol_syn, vol_gt, mode="constant"):
    # check dimension, prepare for training
    h, w, d = vol_syn.shape
    assert d <= h and h == w
    if d < h:
        pad_d1 = int((h - d) / 2)
        pad_d2 = h - d - pad_d1
        vol_syn = np.pad(vol_syn, ((0, 0), (0, 0), (pad_d1, pad_d2)), mode)
        vol_gt = np.pad(vol_gt, ((0, 0), (0, 0), (pad_d1, pad_d2)), mode)

    return vol_syn, vol_gt


def reshape(inputs, outputs, targets, orig_img):
    # orig_img = at.tonumpy(orig_img)  # array_tool module not found
    orig_img = np.array(orig_img)  # Assuming orig_img can be converted to numpy array
    b, c, h, w, d = orig_img.shape
    start_d = int((inputs.shape[-1] - d) / 2)
    end_d = start_d + d
    start_w = int((inputs.shape[-2] - w) / 2)
    end_w = start_w + w
    start_h = int((inputs.shape[-3] - h) / 2)
    end_h = start_h + h
    return (
        inputs[:, :, start_h:end_h, start_w:end_w, start_d:end_d],
        outputs[:, :, start_h:end_h, start_w:end_w, start_d:end_d],
        targets[:, :, start_h:end_h, start_w:end_w, start_d:end_d],
    )


########### Data Augmentation Methods #############


def patch_extraction(Xb, yb, sizePatches=128, Npatches=1):
    """
    3D patch extraction
    """

    batch_size, rows, columns, slices, channels = Xb.shape
    X_patches = np.empty(
        (batch_size * Npatches, sizePatches, sizePatches, sizePatches, channels)
    )
    y_patches = np.empty((batch_size * Npatches, sizePatches, sizePatches, sizePatches))
    i = 0
    for b in range(batch_size):
        for p in range(Npatches):
            x = np.random.randint(rows - sizePatches + 1)
            y = np.random.randint(columns - sizePatches + 1)
            z = np.random.randint(slices - sizePatches + 1)

            X_patches[i] = Xb[
                b, x : x + sizePatches, y : y + sizePatches, z : z + sizePatches, :
            ]
            y_patches[i] = yb[
                b, x : x + sizePatches, y : y + sizePatches, z : z + sizePatches
            ]
            i += 1

    return X_patches, y_patches


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
    p = np.random.permutation(3)
    X_trans, y_trans = np.transpose(X, (p[0], p[1], p[2], 3)), np.transpose(
        y, (p[0], p[1], p[2])
    )
    return X_trans, y_trans


def rotation_zoom3D(X, y):
    """
    Rotate a 3D image with alfa, beta and gamma degree respect the axis x, y and z respectively.
    The three angles are chosen randomly between 0-30 degrees
    """
    alpha, beta, gamma = np.random.random_sample(3) * np.pi / 2
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ]
    )

    Ry = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )

    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )

    R_rot = np.dot(np.dot(Rx, Ry), Rz)

    a, b = 0.8, 1.2
    alpha, beta, gamma = (b - a) * np.random.random_sample(3) + a
    R_scale = np.array([[alpha, 0, 0], [0, beta, 0], [0, 0, gamma]])

    R = np.dot(R_rot, R_scale)
    X_rot = np.empty_like(X)
    for channel in range(X.shape[-1]):
        X_rot[:, :, :, channel] = affine_transform(
            X[:, :, :, channel], R, offset=0, order=1, mode="constant"
        )
    y_rot = affine_transform(y, R, offset=0, order=0, mode="constant")

    return X_rot, y_rot


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
        im = X[:, :, :, c]
        gain, gamma = (1.2 - 0.8) * np.random.random_sample(
            2,
        ) + 0.8
        im_new = np.sign(im) * gain * (np.abs(im) ** gamma)
        if np.min(im_new) < 0:
            print("something wrong")
        max_val = np.max(im_new)
        if max_val > 255:
            im_new = im_new / max_val * 255
        X_new[:, :, :, c] = im_new

    return X_new, y


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
    newX = random_noise(X / 255, mode="poisson", clip=True) * 255
    return newX, y


def random_decisions(N):
    """
    Generate N random decisions for augmentation
    N should be equal to the batch size
    """

    decisions = np.zeros(
        (N, 8)
    )  # 5 is number of aug techniques to combine (patch extraction is not used here)
    for n in range(N):
        decisions[n] = np.random.randint(2, size=8)

    return decisions


def combine_aug(X, y, do):
    """
    Combine randomly the different augmentation techniques written above
    """
    Xnew, ynew = X, y
    # do = [0,1,0,0,0,0,0,0]
    # make sure to use at least 25% of original images
    if np.random.random_sample() > 0.75:
        return Xnew, ynew
    else:
        Xnew = np.float32(Xnew)
        if do[0] == 1:
            Xnew, ynew = flip3D(Xnew, ynew)

        if do[1] == 1:
            Xnew, ynew = transpose3D(Xnew, ynew)  # make sure your input is a cube

        if do[2] == 1:
            Xnew, ynew = brightness(Xnew, ynew)

        if do[3] == 1:
            Xnew, ynew = rotation_zoom3D(Xnew, ynew)

        if do[4] == 1:
            Xnew, ynew = elastic(Xnew, ynew)

        if do[5] == 1:
            Xnew, ynew = add_noise(Xnew, ynew)

        if do[6] == 1:
            Xnew, ynew = blur(Xnew, ynew)

        if do[7] == 1:
            Xnew = normalize(Xnew)

        Xnew = np.uint8(Xnew)
        return Xnew, ynew


def aug_batch(Xb, Yb):
    """
    Generate a augmented image batch
    """
    batch_size = len(Xb)
    newXb, newYb = np.empty_like(Xb), np.empty_like(Yb)

    decisions = random_decisions(batch_size)

    inputs = [(X, y, do) for X, y, do in zip(Xb, Yb, decisions)]
    pool = mp.Pool(processes=8)
    multi_result = pool.starmap(combine_aug, inputs)
    pool.close()

    for i in range(batch_size):
        newXb[i], newYb[i] = multi_result[i][0], multi_result[i][1]

    return newXb, newYb


def aug_one_sample(Xb, Yb):
    """
    Generate a augmented image batch
    """
    batch_size = 1
    Xb = Xb[np.newaxis, :, :, :, np.newaxis]
    Yb = Yb[np.newaxis, :, :, :]

    newXb, newYb = np.empty_like(Xb, dtype=Xb.dtype), np.empty_like(Yb, dtype=Yb.dtype)

    decisions = random_decisions(batch_size)

    for i in range(batch_size):
        newXb[i], newYb[i] = combine_aug(Xb[i], Yb[i], decisions[i])

    return np.squeeze(newXb), np.squeeze(newYb)


def test_aug_batch():
    import os

    gt_path = "/pub2/wu1114/Documents/dataset/NISNet/titan/train_3Dtif/gt/"
    img_path = "/pub2/wu1114/Documents/dataset/NISNet/titan/train_3Dtif/syn/"
    gt_files = sorted(os.listdir(gt_path))
    img_files = sorted(os.listdir(img_path))
    for i, (gt_file, img_file) in enumerate(zip(gt_files, img_files)):
        if i <= 600:
            continue
        print("reading files:", gt_file, img_file)
        gt = io.imread(os.path.join(gt_path, gt_file)).astype(np.uint16)
        img = io.imread(os.path.join(img_path, img_file)).astype(np.uint8)

        img = img[np.newaxis, :, :, :, np.newaxis]
        gt = gt[np.newaxis, :, :, :]
        Xnew, ynew = aug_batch(img, gt)
        print(Xnew.shape, ynew.shape)
        Xnew, ynew = np.squeeze(Xnew), np.squeeze(ynew)
        # io.imsave(os.path.join('aug', 'syn', img_file),np.uint8(np.transpose(Xnew, (2,0,1))))
        # io.imsave(os.path.join('aug', 'gt', gt_file),np.uint8(np.transpose(ynew, (2,0,1))))


if __name__ == "__main__":
    test_aug_batch()
