from typing import List
import numpy as np
from skimage import measure
from skimage.feature import peak_local_max
from skimage.transform import rescale


def get_object_code(vol_gt, vox_threshold):
    # object volume smaller than vol_threshold will be ignored, this function returns all nuclei centroid
    nuclei_intensity = list(np.unique(vol_gt))
    nuclei_intensity.remove(0)
    object_code = []
    for idx in nuclei_intensity:
        # for ith nuclei
        # print(i, idx)
        h, w, z = vol_gt.shape
        (xs, ys, zs) = np.where(vol_gt == idx)
        y1, x1, z1, y2, x2, z2 = (
            np.min(xs),
            np.min(ys),
            np.min(zs),
            np.max(xs),
            np.max(ys),
            np.max(zs),
        )
        y, x, z = (
            round((y2 - y1) / 2 + y1),
            round((x2 - x1) / 2 + x1),
            round((z2 - z1) / 2 + z1),
        )
        nucleus_vox = len(vol_gt[vol_gt == idx].flatten())
        if nucleus_vox >= vox_threshold:
            object_code.append([y, x, z, y1, y2, x1, x2, z1, z2, 0])

    return np.array(object_code, np.int16)


def calc_vector_distance_by_object(
    gt_vol,
    y_coords: List[int],
    x_coords: List[int],
    z_coords: List[int],
    image_height: int,
    image_width: int,
    image_depth: int,
    max_dist,
    vox_threshold: int,
):
    assert len(y_coords) == len(x_coords), "list of coordinates should be the same"
    assert len(y_coords) > 0, "No centroids in source image"

    shape = [image_height, image_width, image_depth]
    image_coords = np.indices(shape)
    image_coords_planar = np.transpose(image_coords, [1, 2, 3, 0])
    vec_ctr = np.zeros([image_height, image_width, image_depth, 3], dtype=np.float32)
    # for (i, (y, x, z)) in enumerate(zip(y_coords, x_coords, z_coords)):
    nuclei_intensity = list(np.unique(gt_vol))
    nuclei_intensity.remove(0)
    for i in nuclei_intensity:
        (xs, ys, zs) = np.where(gt_vol == i)
        y1, x1, z1, y2, x2, z2 = (
            np.min(xs),
            np.min(ys),
            np.min(zs),
            np.max(xs),
            np.max(ys),
            np.max(zs),
        )
        y, x, z = (
            round((y2 - y1) / 2 + y1),
            round((x2 - x1) / 2 + x1),
            round((z2 - z1) / 2 + z1),
        )  # get centroid
        nucleus_vox = len(gt_vol[gt_vol == i].flatten())
        if nucleus_vox >= vox_threshold:
            vec = np.array([y, x, z]) - image_coords_planar
            # only keep the vector in an object
            obj_mask = np.uint8(gt_vol == i)
            vec = np.float32(vec) * obj_mask[:, :, :, np.newaxis]
            # normalize vec to (-1,1)
            # vec_min, vec_max = np.amin(vec), np.amax(vec)
            # if vec_min < 0:
            #     vec[vec<0] = vec[vec<0]/np.abs(vec_min)
            # if vec_max > 0:
            #     vec[vec>0] = vec[vec>0]/np.abs(vec_max)
            vec_ctr = vec_ctr + vec  # add object distance map to vec_cube

    # Clip vectors
    if not max_dist is None:
        active = np.sqrt(np.sum(vec_ctr ** 2, axis=3)) > max_dist
        vec_ctr[active, :] = 0

    return vec_ctr


def encode(
    coords,
    vol_gt,
    image_height: int,
    image_width: int,
    image_depth: int,
    max_dist: int,
    num_classes: int,
    vox_threshold: int,
):
    if len(coords) != 0:
        (
            y_coords,
            x_coords,
            z_coords,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = np.transpose(coords)
        # Encode vectors
        target_vectors = calc_vector_distance_by_object(
            vol_gt,
            y_coords,
            x_coords,
            z_coords,
            image_height,
            image_width,
            image_depth,
            max_dist,
            vox_threshold,
        )
    else:
        target_vectors = np.zeros(
            [image_height, image_width, image_depth, 3], dtype=np.float32
        )

    if not max_dist is None:
        target_vectors /= max_dist
    target_vectors = np.transpose(target_vectors, [3, 0, 1, 2])

    # Encode logits (bounding box is drawn as ellipses)
    target_logits = np.zeros((num_classes, image_height, image_width, image_depth))
    target_logits[0] = 1
    target_logits[0][vol_gt > 0] = 0
    target_logits[1][vol_gt > 0] = 1

    target = np.concatenate((target_vectors, target_logits, vol_gt[np.newaxis, :]))
    return target


def encode_norm(vol_gt, vox_threshold=100):
    import cc3d

    # Re-etiquetar la entrada
    vol_gt = cc3d.dust(cc3d.connected_components(vol_gt), vox_threshold)

    from skimage import measure

    # Obtener propiedades de cada etiqueta (bounding box, coordenadas de los centroides, valores de las etiquetas)
    props = measure.regionprops(vol_gt)
    bbox = np.array([x["bbox"] for x in props])
    centroids = get_object_code(vol_gt, vox_threshold)[:, :3]
    labels = np.array([x["label"] for x in props]).astype(np.int16)
    if len(labels) != 0:
        # Separar cada coordenada de la bounding box de cada etiqueta
        z1, y1, x1, z2, y2, x2 = (
            bbox[:, 0],
            bbox[:, 1],
            bbox[:, 2],
            bbox[:, 3],
            bbox[:, 4],
            bbox[:, 5],
        )

        # Obtener cada máscara de cada núcleo
        mask_i = [
            vol_gt[z1[i] : z2[i], y1[i] : y2[i], x1[i] : x2[i]] == labels[i]
            for i in range(len(z1))
        ]

        # Obtener dimensiones de la bonding box de cada objeto
        h, w, d = (
            bbox[:, 3] - bbox[:, 0],
            bbox[:, 4] - bbox[:, 1],
            bbox[:, 5] - bbox[:, 2],
        )

        # Máscara de unos para multiplicar la gradiente en cada bounding box
        img_ones = [np.ones((h[i], w[i], d[i])) for i in range(len(z1))]

        # Generación de gradiente normalizada de acuerdo al valor máximo absoluto de la gradiente en la bounding box
        concat_z = []
        concat_y = []
        concat_x = []
        for i in range(len(z1)):
            grad_z_concat = np.concatenate(
                [
                    np.linspace(
                        centroids[i, 0] - z1[i], 0, centroids[i, 0] - z1[i] + 1
                    ),
                    np.linspace(
                        0, -(z2[i] - centroids[i, 0] - 1), z2[i] - centroids[i, 0]
                    )[1:],
                ]
            )
            max_grad_z = np.max(np.abs(grad_z_concat)) + 1e-10
            grad_z_concat_norm = (grad_z_concat / max_grad_z).reshape(-1, 1, 1)

            grad_y_concat = np.concatenate(
                [
                    np.linspace(
                        centroids[i, 1] - y1[i], 0, centroids[i, 1] - y1[i] + 1
                    ),
                    np.linspace(
                        0, -(y2[i] - centroids[i, 1] - 1), y2[i] - centroids[i, 1]
                    )[1:],
                ]
            )
            max_grad_y = np.max(np.abs(grad_y_concat)) + 1e-10
            grad_y_concat_norm = (grad_y_concat / max_grad_y).reshape(1, -1, 1)

            grad_x_concat = np.concatenate(
                [
                    np.linspace(
                        centroids[i, 2] - x1[i], 0, centroids[i, 2] - x1[i] + 1
                    ),
                    np.linspace(
                        0, -(x2[i] - centroids[i, 2] - 1), x2[i] - centroids[i, 2]
                    )[1:],
                ]
            )
            max_grad_x = np.max(np.abs(grad_x_concat)) + 1e-10
            grad_x_concat_norm = (grad_x_concat / max_grad_x).reshape(1, 1, -1)

            concat_z.append(grad_z_concat_norm.astype(np.float16))
            concat_y.append(grad_y_concat_norm.astype(np.float16))
            concat_x.append(grad_x_concat_norm.astype(np.float16))

        # Multiplicación de la gradiente con el objeto para obtener un objeto con gradiente
        grad_z_i = [img_ones[i] * concat_z[i] * mask_i[i] for i in range(len(z1))]
        grad_y_i = [img_ones[i] * concat_y[i] * mask_i[i] for i in range(len(y1))]
        grad_x_i = [img_ones[i] * concat_x[i] * mask_i[i] for i in range(len(x1))]

        # Obtención de la imagen original con la gradiente normalizada en cada objeto
        grad_z = np.zeros(vol_gt.shape, dtype=np.float32)
        grad_y = np.zeros(vol_gt.shape, dtype=np.float32)
        grad_x = np.zeros(vol_gt.shape, dtype=np.float32)
        for i in range(len(z1)):
            grad_z[z1[i] : z2[i], y1[i] : y2[i], x1[i] : x2[i]] += grad_z_i[i]
            grad_y[z1[i] : z2[i], y1[i] : y2[i], x1[i] : x2[i]] += grad_y_i[i]
            grad_x[z1[i] : z2[i], y1[i] : y2[i], x1[i] : x2[i]] += grad_x_i[i]
        target_vector = np.stack(
            [
                grad_z.astype(np.float32),
                grad_y.astype(np.float32),
                grad_x.astype(np.float32),
            ],
            axis=-1,
        )

    else:
        target_vector = np.zeros(vol_gt.shape + (3,), dtype=np.float32)

    target_logits = np.zeros(vol_gt.shape + (2,))
    target_logits[vol_gt == 0, ..., 0] = 1
    target_logits[vol_gt != 0, ..., 1] = 1

    return target_vector, target_logits, vol_gt[..., np.newaxis]


def calc_vote_image(centroid_vectors: np.array, f):
    channels, height, width, depth = centroid_vectors.shape

    size = np.array(
        np.array((height, width, depth), dtype="float32") * ((1 / f), (1 / f), (1 / f)),
        dtype="int16",
    )
    indices = np.indices((height, width, depth), dtype=centroid_vectors.dtype)

    # Calculate absolute vectors
    vectors = ((centroid_vectors + indices) * (1 / f)).astype("int")
    nimage = np.zeros((size[0], size[1], size[2]))

    # Clip pixels
    logic = np.logical_and(
        np.logical_and(
            np.logical_and(vectors[0] >= 0, vectors[1] >= 0), vectors[2] >= 0
        ),
        np.logical_and(
            np.logical_and(vectors[0] < size[0], vectors[1] < size[1]),
            vectors[2] < size[2],
        ),
    )
    coords = vectors[:, logic]

    # Accumulate
    np.add.at(nimage, (coords[0], coords[1], coords[2]), 1)
    return np.expand_dims(nimage, axis=0).astype(np.float32)


def decode(
    input: np.ndarray,
    max_dist: int,
    binning: int,
    nm_size: int,
    centroid_threshold: int,
):
    _, image_height, image_width, image_depth = input.shape
    # centroid_vectors = input[0:3] * max_dist  # no need this line if no normalization is applied during training
    centroid_vectors = input[0:3]
    logits = input[3:]

    # Calculate class ids and class probabilities
    class_ids = np.expand_dims(np.argmax(logits, axis=0), axis=0).astype(np.int16)
    sum_logits = np.expand_dims(np.sum(logits, axis=0), axis=0)
    class_probs = np.expand_dims(np.max((logits / sum_logits), axis=0), axis=0)
    class_probs = np.clip(class_probs, 0, 1)

    # Calculate the centroid images
    votes = calc_vote_image(centroid_vectors, binning)
    votes_nm = peak_local_max(
        votes[0],
        min_distance=nm_size,
        threshold_abs=centroid_threshold,
        indices=False,
        exclude_border=False,
        num_peaks_per_label=1,
    )
    votes_nm = np.expand_dims(votes_nm, axis=0)
    votes_nm = np.uint8(votes_nm) * 255
    # Calculate list of centroid statistics
    coords = np.transpose(np.where(votes_nm[0] > 0))
    centroids = [
        [
            y * binning,
            x * binning,
            z * binning,
            class_ids[0, y * binning, x * binning, z * binning] - 1,
            class_probs[0, y * binning, x * binning, z * binning],
        ]
        for (y, x, z) in coords
    ]
    # convert the votes and votes_nm to the same dimension as input image
    votes_nm = np.zeros((1, image_height, image_width, image_depth), np.uint8)
    for y, x, z in coords:
        votes_nm[0, y * binning, x * binning, z * binning] = 255
    votes_rescale = rescale(
        votes[0], scale=binning, order=0
    )  # interpolate with nearest neighbor
    votes_rescale = votes_rescale[np.newaxis, :]

    # length of centroids cannot represent the number of centroids because some peaks might have multiple pixels, if they are adjacent peaks. But it does not affect marker based watershed.
    return centroid_vectors, votes_rescale, class_ids, class_probs, votes_nm, centroids
