import h5py
import os
from tifffile import imwrite
import numpy as np

def print_hdf5_structure(name, obj):
    """Helper function to print groups and datasets in an HDF5 file."""
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"  Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")

# Root folder containing all subfolders
root_folder = "/medicina/hmorales/projects/NucleiSegmentation3D/code/data/processed"

# Walk through subfolders and look for 'data_val.hdf5'
for subdir, _, files in os.walk(root_folder):
    if "data_train.hdf5" in files:
        file_path = os.path.join(subdir, "data_val.hdf5")
        print(f"\n=== Exploring file: {file_path} ===")
        try:
            with h5py.File(file_path, "r") as f:
                # Print HDF5 structure
                f.visititems(print_hdf5_structure)

                # Export datasets if they exist
                for dataset_name in ["images", "labels"]:
                    if dataset_name in f:
                        data = f[dataset_name][:]
                        print(f"  -> Exporting {dataset_name}, shape={data.shape}, dtype={data.dtype}")

                        # Output path (multi-page TIFF)
                        out_path = os.path.join(subdir, f"{dataset_name}.tif")

                        # Reshape for multipage TIFF:
                        # (N, z, y, x) -> (N*z, y, x)
                        if data.ndim == 4:
                            N, z, y, x = data.shape
                            data_to_save = data.reshape(N * z, y, x)
                        elif data.ndim == 3:
                            data_to_save = data
                        else:
                            raise ValueError(f"Unexpected dataset shape: {data.shape}")

                        # Save as multipage TIFF (Fiji-compatible)
                        imwrite(out_path, (255 * data_to_save).astype(np.uint8), photometric="minisblack")
                        print(f"  -> Saved {out_path}")

        except Exception as e:
            print(f"Could not open {file_path}: {e}")
