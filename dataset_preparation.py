# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:40:13 2025

@author: WORKSTATION-PC
"""

import os
import random
import shutil
from pathlib import Path

# Ruta de la carpeta principal
base_path = r"./datasets/ground_truth_and_synthetic"


def is_synthetic(path):
    return "synthetic" in os.path.normpath(path).split(os.sep)


def get_files(base_path):
    files_names, files_names_gt = [], []
    files_synthetic, files_synthetic_gt = [], []

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue

        file_names_class, file_names_gt_class = [], []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if not (file.endswith("tif") or file.endswith("tiff")):
                    continue

                file_path = os.path.join(root, file)

                if is_synthetic(file_path) and not "gt" in file_path:
                    files_synthetic.append(file_path)
                    continue

                if is_synthetic(file_path) and "gt" in file_path:
                    files_synthetic_gt.append(file_path)
                    continue

                if "gt" in file_path:
                    file_names_gt_class.append(file_path)
                    continue
                file_names_class.append(file_path)

        files_names.append(file_names_class)
        files_names_gt.append(file_names_gt_class)

    return files_names, files_names_gt, files_synthetic, files_synthetic_gt


def split_data(files_names, files_names_gt, test_size=0.1, seed=42):
    random.seed(seed)
    test_files, train_files, test_files_gt, train_files_gt = [], [], [], []

    for file_names_class, file_names_gt_class in zip(files_names, files_names_gt):
        num_test = max(1, int(len(file_names_class) * test_size))
        test_files_class = random.sample(file_names_class, num_test)
        train_files_class = [x for x in file_names_class if x not in test_files_class]

        test_files.extend(test_files_class)
        train_files.extend(train_files_class)

        test_files_gt_class = [
            gt
            for gt in file_names_gt_class
            if any(Path(img).stem in gt for img in test_files_class)
        ]
        train_files_gt_class = [
            gt for gt in file_names_gt_class if gt not in test_files_gt_class
        ]

        test_files_gt.extend(test_files_gt_class)
        train_files_gt.extend(train_files_gt_class)

    return train_files, test_files, train_files_gt, test_files_gt


def create_dirs(base_output, images_tag="images", labels_tag="masks"):
    train_dir = os.path.join(base_output, "train", images_tag)
    test_dir = os.path.join(base_output, "test", images_tag)
    train_mask_dir = os.path.join(base_output, "train", labels_tag)
    test_mask_dir = os.path.join(base_output, "test", labels_tag)

    for d in [train_dir, test_dir, train_mask_dir, test_mask_dir]:
        os.makedirs(d, exist_ok=True)

    return train_dir, test_dir, train_mask_dir, test_mask_dir


def copy_files(file_list, dest_dir):
    for file in file_list:
        shutil.copy(file, os.path.join(dest_dir, os.path.basename(file)))


def process_dataset(base_path, output_path, test_size=0.1, seed=42):
    files_names, files_names_gt, files_synthetic, files_synthetic_gt = get_files(
        base_path
    )
    train_files, test_files, train_files_gt, test_files_gt = split_data(
        files_names, files_names_gt, test_size, seed
    )
    train_dir, test_dir, train_mask_dir, test_mask_dir = create_dirs(output_path)

    copy_files(train_files, train_dir)
    copy_files(test_files, test_dir)
    copy_files(train_files_gt, train_mask_dir)
    copy_files(test_files_gt, test_mask_dir)


def copy_file_synthetic(path_synthetic, dst_folder, is_gt=False):
    parts = list(Path(path_synthetic).parts[-4:-2])
    filename = os.path.basename(path_synthetic)
    if is_gt:
        filename = filename.replace(".tif", "_gt.tif")

    dst_file = os.path.join(dst_folder, "_".join(parts) + "_" + filename)
    shutil.copy(path_synthetic, dst_file)


def copy_synthetic(files_synthetic, files_synthetic_gt, train_dir, train_mask_dir):
    for syn_file in files_synthetic:
        copy_file_synthetic(syn_file, train_dir)
    for syn_file in files_synthetic_gt:
        copy_file_synthetic(syn_file, train_mask_dir, is_gt=True)


def validate_data(train_dir, train_mask_dir):
    train_data = sorted(os.listdir(train_dir))
    train_data = [Path(x).stem for x in train_data]
    train_data_gt = sorted(os.listdir(train_mask_dir))
    train_data_gt = [Path(x).stem[:-3] for x in train_data_gt]

    if train_data == train_data_gt:
        print("✅ Los archivos de imágenes y GT coinciden perfectamente.")
    else:
        print("⚠️ Hay diferencias entre las imágenes y sus máscaras.")
        missing_in_gt = set(train_data) - set(train_data_gt)
        missing_in_train = set(train_data_gt) - set(train_data)

        if missing_in_gt:
            print("❌ Archivos en imágenes que no tienen GT:", missing_in_gt)
        if missing_in_train:
            print(
                "❌ Archivos en GT que no tienen imagen correspondiente:",
                missing_in_train,
            )


# Uso principal
def main():
    base_path = "./datasets/ground_truth_and_synthetic"
    output_path1 = "./datasets/solo_ground_truth_train_test"
    output_path2 = "./datasets/ground_truth_and_synthetic_train_test"

    print("Procesando solo Ground Truth...")
    process_dataset(base_path, output_path1)
    train_dir, _, train_mask_dir, _ = create_dirs(output_path1)
    print("Validando datos...")
    validate_data(train_dir, train_mask_dir)

    print("Procesando Ground Truth y Synthetic...")
    process_dataset(base_path, output_path2)

    train_dir, _, train_mask_dir, _ = create_dirs(output_path2)

    _, _, files_synthetic, files_synthetic_gt = get_files(base_path)
    copy_synthetic(files_synthetic, files_synthetic_gt, train_dir, train_mask_dir)

    print("Validando datos...")
    validate_data(train_dir, train_mask_dir)


if __name__ == "__main__":
    main()
