import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from tifffile import imread, imwrite
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist import random_label_cmap
from stardist.models import StarDist3D
from preprocessing.image_preprocessing import preprocessing
import os

np.random.seed(6)
lbl_cmap = random_label_cmap()

basedir = r"./stardist3d"
model_name = "stardist3d_model_synthetic_transposed_64_96_96_32_20250629"
print(os.listdir(os.path.join(basedir, model_name)))
model = StarDist3D(None, name=model_name, basedir=basedir)

# Path to the directory containing input images
input_dir = r"datasets/ground_truth_and_synthetic_train_test/test/images"
# Path to the directory where predictions will be saved
output_dir = os.path.join(basedir, model_name, "predictions")
n_tiles = (1, 1, 1)


# Cargar imagen
def load_data_prediction(path):
    name = os.path.basename(path).split(".")[0]
    images = normalize(imread(path))
    return images.astype(np.float16), name


def process_and_predict_images(input_dir, output_dir, n_tiles):
    """
    Process and predict instances for all images in a directory.

    Parameters:
        input_dir (str): Path to the directory containing input images.
        output_dir (str): Path to the directory where predictions will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all image files in the input directory
    image_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".tif")
    ]

    for image_path in image_files:
        try:
            # Load and preprocess the image
            img, name = load_data_prediction(image_path)
            img_preproc = preprocessing(img)

            # Predict instances
            labels, details = model.predict_instances(img_preproc, n_tiles=n_tiles)

            # Save the prediction
            output_path = os.path.join(output_dir, f"{name}_label.tif")
            imwrite(output_path, labels)
            print(f"Processed and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


process_and_predict_images(input_dir, output_dir, n_tiles=n_tiles)
