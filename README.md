# NucVerse3D

Source code for **NucVerse3D: generalizable 3D nuclear instance segmentation across heterogeneous microscopy modalities**
(Vergara, Perez-Gallardo, Velasco et al., *Scientific Reports*, 2026 — [doi:10.1038/s41598-026-51994-x](https://doi.org/10.1038/s41598-026-51994-x)).

NucVerse3D is a 3D Residual Attention U-Net that predicts a binary nuclear mask plus a 3D
gradient field; the field is then collapsed into instance masks via gradient-descent voxel
grouping. The code in this repository covers dataset preparation, training, and prediction.

---

## 1. Requirements

- Linux with an NVIDIA GPU (CUDA-capable). Windows works via WSL2 (Ubuntu) with the
  NVIDIA Windows driver — native Windows is not supported since TensorFlow dropped GPU
  support there after 2.10.
- Python **3.10**
- Conda (recommended) or any virtualenv tool

Install:

```bash
conda create -n nucverse3d python=3.10 -y
conda activate nucverse3d
pip install -e .
```

All dependencies (TensorFlow, scikit-image, typer, …) are declared in `pyproject.toml`.
Only TensorFlow is hard-pinned (`tensorflow[and-cuda]==2.16.2`, the version used in the
paper, which drives the entire CUDA wheel stack); everything else is left unpinned so the
install keeps solving against current PyPI.

The editable install also exposes every script under `src/` as a top-level module
(`config`, `nuclei_prediction`, `training_pipeline`, …) plus sub-packages (`keras_models`,
`utils`, …), so you can run `python -m nuclei_prediction …` from the repo root.

---

## 2. Data and pretrained models

All data and trained model weights for the paper are archived on Zenodo:
**[https://doi.org/10.5281/zenodo.18517324](https://doi.org/10.5281/zenodo.18517324)**

The repository follows a Noble (2009) layout. Expand `raw.zip` into `data/raw/` and
`models.zip` into `models/`. Each raw dataset is a folder with `train/{images,masks}` and
`test/{images,masks}`; each model is a folder with `best_model.weights.h5`. Concretely,
after extraction you should see something like:

```
data/raw/
  2_livernuclei/
    train/{images,masks}/        # 6 TIFFs each
    test/{images,masks}/         # 3 TIFFs each
  6_Drosophila_denoised/
    train/
      images/Denoised-C2_M03.tif, Denoised-C2_M05.tif, Denoised-C2_M06.tif
      masks/C2_M03-label.tif, C2_M05-label.tif, C2_M06-label.tif
    test/
      images/C2_M01.tif, C2_M04.tif
      masks/C2_M01-label.tif, C2_M04-label.tif
  7_liver_hcc_dataset/
    train/{images,masks}/        # 8 TIFFs each
    test/{images,masks}/         # 4 TIFFs each

models/
  resunet_drosophila-denoised/   # ~486 MB best_model.weights.h5 + modelSave + history + time
  resunet_combined_1000/
  resunet_combined_scaled_1000/
  ...                            # additional pretrained models from models.zip

data/processed/                  # data_train.hdf5 / data_val.hdf5 produced by the splitter
external/                        # optional: third-party datasets you bring in yourself
```

Image and mask filenames don't need to share a prefix — the splitter pairs them by sorted
order within each `{images,masks}` folder (so `Denoised-C2_M0X.tif` ↔ `C2_M0X-label.tif`
works out of the box). For every input TIFF, prediction writes two files under
`results/<model_name>/<parent>/<image_stem>/`:

```
<image_stem>_binary.tif   # foreground probability mask
<image_stem>_labels.tif   # 3D instance labels (uint16)
```

`<parent>` is the second-level parent directory of the input path — e.g., predicting on
`data/raw/6_Drosophila_denoised/test/images/` puts outputs under
`results/<model_name>/6_Drosophila_denoised/<image_stem>/`.

---

## 3. Training a model

Pick a `<name>` for your model and thread it through each step — the HDF5 lands at
`data/processed/<name>/`, the trained weights at `models/resunet_<name>/`. Two builders
produce the HDF5 in the same format; the choice is just how many source datasets get
pooled:

| Goal                                                   | Builder                       | Setup                                                                                  |
| ------------------------------------------------------ | ----------------------------- | -------------------------------------------------------------------------------------- |
| **Single-dataset model** (one tissue/modality)         | `dataset_splitter`            | one dataset folder with `train/{images,masks}`                                         |
| **Combined model** (one model across N datasets)       | `generate_combined_data_set`  | a folder containing N datasets — use `group_datasets` to build it (full recipe in §6)  |

### 3.1 Build HDF5 — single dataset

```bash
python -m dataset_splitter \
    data/raw/<dataset>/train \
    --val-path data/raw/<dataset>/val \
    --dataset-name <name> \
    --images-tag images --labels-tag masks
```

`--val-path` is optional; if `val/` doesn't exist, `train/` is auto-split. Image/mask
pairing is by **sorted filename order** (so prefixes like `Denoised-…` on images and
`…-label.tif` on masks are fine).

### 3.2 Build HDF5 — combined (multiple datasets, same format)

A combined HDF5 is §3.1's format pooled from N source datasets, so one model can train
across all of them. First group the source datasets under one folder (§6 shows the full
recipe with `group_datasets`), then:

```bash
python -m generate_combined_data_set split-dataset \
    data/raw/<grouped_root> \
    -d <name> -n 1000 --seed 42 --no-scale   # or --scale
```

`-n` is the patch budget **per source dataset** (paper used 1000 = 800 train + 200 val).
`--scale` isotropically rescales each dataset so the mean nuclear radius matches a common
reference (produces a "scaled" model — see §7); `--no-scale` keeps native voxel sizes.

### 3.3 Train

Same command regardless of how the HDF5 was built:

```bash
python -m training_pipeline \
    data/processed/<name>/data_train.hdf5 \
    data/processed/<name>/data_val.hdf5 \
    --model-name resunet_<name>
```

Trained weights land in `models/resunet_<name>/best_model.weights.h5` — the file
`nuclei_prediction` loads. Hyperparameters live in `src/config.py`.

---

## 4. Prediction

```bash
python -m nuclei_prediction <model_name> <input_dir>
```

`<model_name>` is the name of a folder inside `models/`. When you run
`python -m nuclei_prediction my_model …`, the script opens exactly one file:
`models/my_model/best_model.weights.h5` (relative to the repo root). If that file isn't
there, it errors — there's no search of any other location. After extracting
`models.zip` you'll have folders like `resunet_liver_hcc`, `resunet_drosophila-denoised`,
`resunet_combined_1000`, `resunet_combined_scaled_1000` — pass any of those. A model you
trained in §3 with `--model-name resunet_<name>` is saved in `models/resunet_<name>/`, so
you pass that same name to predict.

`<input_dir>` is a folder containing 3D TIFFs to segment. Outputs land under
`results/<model_name>/<parent>/<image_stem>/` (see §2 for the `<parent>` rule).

### Dataset-specific or generalized (unscaled) model

```bash
python -m nuclei_prediction resunet_liver_hcc      data/raw/7_liver_hcc_dataset/test/images
python -m nuclei_prediction resunet_combined_1000  data/raw/2_livernuclei/test/images
```

### Generalized **scaled** model

Add `--avg_radius` (mean nuclear radius of the input, in voxels). Per-dataset values used
in the paper are in `predict_general.sh`; for new data estimate it as
`np.mean(np.cbrt(3 * volumes / (4 * np.pi)))` from a few annotated nuclei.

```bash
python -m nuclei_prediction resunet_combined_scaled_1000 \
    data/raw/7_liver_hcc_dataset/test/images \
    --avg_radius 14.0
```

---

## 5. Worked example — single-dataset train → predict (Drosophila)

End-to-end workflow on the smallest dataset we have (`6_Drosophila_denoised` — 3 training
volumes). Pick a name (`my_drosophila` below) and thread it through every step:

```bash
# Build HDF5 from the dataset's train/ folder; val auto-splits from train.
python -m dataset_splitter \
    data/raw/6_Drosophila_denoised/train \
    --dataset-name my_drosophila \
    --images-tag images --labels-tag masks

python -m training_pipeline \
    data/processed/my_drosophila/data_train.hdf5 \
    data/processed/my_drosophila/data_val.hdf5 \
    --model-name resunet_my_drosophila

python -m nuclei_prediction resunet_my_drosophila \
    data/raw/6_Drosophila_denoised/test/images
```

Outputs land in `results/resunet_my_drosophila/6_Drosophila_denoised/<image_stem>/` — open
`<image_stem>_labels.tif` alongside the raw stack in [Napari](https://napari.org/) or
Fiji.

---

## 6. Worked example — combined train → predict

Full multi-dataset workflow, demonstrated on three datasets
(`2_livernuclei`, `6_Drosophila_denoised`, `7_liver_hcc_dataset`). Add more dataset
folders to the `group_datasets` call to combine larger benchmarks.

Pick a name for this combined dataset and thread it through every step. The example
below uses `my_combined` and assumes the source datasets live under `data/raw/`; both
locations are user-chosen.

**Step 1 — Group the source datasets.** `generate_combined_data_set` reads **one root**
that contains all datasets to pool, so first bundle them under one folder using the
`group_datasets` helper:

```bash
python -m group_datasets \
    --from data/raw \
    --out  data/raw/my_combined \
    2_livernuclei 6_Drosophila_denoised 7_liver_hcc_dataset
```

`--from` is the directory your datasets live in; `--out` is where to save the grouped
root; the positional arguments are the dataset folder names.

Resulting structure (each dataset duplicated under `my_combined/`):

```
data/raw/my_combined/
├── 2_livernuclei/
│   ├── train/{images,masks}/
│   └── test/{images,masks}/
├── 6_Drosophila_denoised/
│   ├── train/{images,masks}/
│   └── test/{images,masks}/
└── 7_liver_hcc_dataset/
    ├── train/{images,masks}/
    └── test/{images,masks}/
```

The builder uses `train/` from each dataset and auto-splits validation from it (since
`val/` is absent); `test/` is ignored.

**Steps 2–4 — Build, train, predict** (same commands as §3 and §4, threading `my_combined`
through):

```bash
python -m generate_combined_data_set split-dataset \
    data/raw/my_combined -d my_combined -n 1000 --seed 42 --no-scale

python -m training_pipeline \
    data/processed/my_combined/data_train.hdf5 \
    data/processed/my_combined/data_val.hdf5 \
    --model-name resunet_my_combined

python -m nuclei_prediction resunet_my_combined \
    data/raw/6_Drosophila_denoised/test/images
```

---

## 7. Worked example — combined **scaled** train → predict

Identical to §6 except: `--scale` at build (rescales each source dataset to a common
reference nuclear radius), and `--avg_radius` at predict (the mean nuclear radius of the
input in voxels — Drosophila = 13.5; see `predict_general.sh` for other datasets).
Reuses the `data/raw/my_combined/` grouping from §6:

```bash
python -m generate_combined_data_set split-dataset \
    data/raw/my_combined -d my_combined_scaled -n 1000 --seed 42 --scale

python -m training_pipeline \
    data/processed/my_combined_scaled/data_train.hdf5 \
    data/processed/my_combined_scaled/data_val.hdf5 \
    --model-name resunet_my_combined_scaled

python -m nuclei_prediction resunet_my_combined_scaled \
    data/raw/6_Drosophila_denoised/test/images \
    --avg_radius 13.5
```

---

## 8. Repository layout (relevant entry points)

```
src/
  config.py                  # all hyperparameters & paths
  dataset_splitter.py        # raw images/masks -> HDF5 patches (single dataset)
  generate_combined_data_set.py
  generate_combined_data_set_subfolder.py
  training_pipeline.py       # train one model on HDF5 patches
  nuclei_prediction.py       # predict over a folder of 3D TIFFs
  group_datasets.py          # bundle multiple datasets under one root
  keras_models/              # Attention ResUNet 3D + other architectures
  preprocessing/             # illumination correction, deconvolution, normalization
  postprocessing/            # gradient-descent grouping into instances
  utils/                     # losses, image/tensor helpers
  stardist3d/                # StarDist3D baselines used in benchmarking

train.sh                     # training recipes used in the paper
predict_general.sh           # prediction recipes (specific + general + scaled)
```

---

## 9. Citation

If you use NucVerse3D, please cite:

> Vergara J., Perez-Gallardo C., Velasco R., Martinez D., Badilla D., Freire-Navarrete D.,
> Contreras E. G., Guevara P., Segovia-Miranda F. & Morales-Navarrete H.
> *NucVerse3D: generalizable 3D nuclear instance segmentation across heterogeneous microscopy
> modalities.* Sci Rep (2026). https://doi.org/10.1038/s41598-026-51994-x

## License

See [LICENSE](LICENSE).
