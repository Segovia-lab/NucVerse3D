# Segovia Lab 3D Nuclei Segmentation

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

3D nuclei segmentation project using deep learning models for biological image analysis.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src/                                              <- Source code for use in this project
    ├── __init__.py
    ├── config.py                                     <- Configuration settings and parameters
    ├── dataset_preparation.py                        <- Dataset preparation (was dataset_synthetic_nuclei.py)
    ├── dataset_splitter.py                           <- Train/validation split (was generate_training_val_set.py)
    ├── nuclei_prediction.py                          <- Main prediction pipeline (was prediction WS4.py)
    ├── training_pipeline.py                          <- Training pipeline (was training.py)
    ├── dataset.py                                    <- Stub file (preserved)
    ├── features.py                                   <- Stub file (preserved)
    ├── plots.py                                      <- Stub file (preserved)
    │
    ├── keras_models/                                 <- Keras model architectures
    │   ├── __init__.py
    │   ├── unet_basic.py                             <- Basic U-Net model
    │   ├── unet_3d_dual_decoder.py                   <- Dual decoder U-Net (was unet_3d_model_2_decoder.py)
    │   ├── resnet_unet_3d.py                         <- ResNet U-Net 3D (was res_unet_3D.py)
    │   ├── attention_unet_3d.py                      <- Attention U-Net 3D (was models_3d_attention.py)
    │   ├── unet_3d_filters.py                        <- U-Net with filters (was Unet3D_filts.py)
    │   └── nisnet.py                                 <- NISNet architecture
    │
    ├── modeling/                                     <- Training and prediction scripts
    │   ├── __init__.py
    │   ├── train.py                                  <- Model training scripts
    │   └── predict.py                                <- Model prediction scripts
    │
    ├── preprocessing/                                <- Data preprocessing utilities
    │   ├── __init__.py
    │   ├── dataloader.py                             <- Data loading and batching (moved from src root)
    │   ├── image_preprocessing.py                    <- Image preprocessing (was preprocessing.py)
    │   └── data_augmentation.py                      <- Data augmentation functions
    │
    ├── postprocessing/                               <- Post-processing utilities
    │   ├── __init__.py
    │   ├── segmentation_postprocessing.py            <- Segmentation post-processing (was post_procesamiento.py)
    │   └── smooth_prediction.py                      <- Smooth prediction (was smooth_predict3D.py)
    │
    ├── utils/                                        <- Utility functions
    │   ├── __init__.py
    │   ├── image.py                                  <- Image utilities (was utils.py)
    │   ├── tensor.py                                 <- Tensor utilities (was utils_2.py)
    │   ├── data.py                                   <- Data utilities (was data_utils.py)
    │   ├── parallel_interpolation.py                 <- Parallel interpolation (was paralel_interpolation.py)
    │   ├── loss_functions.py                         <- Custom loss functions
    │   └── memory_profiler.py                        <- Memory profiling (was get_model_memory_usage.py)
    │
    └── stardist3d/                                   <- StarDist3D implementation (preserved structure)
        ├── training_stardist3d.py                    <- StarDist3D training
        └── prediction_stardist3d.py                  <- StarDist3D prediction
```

--------

## Recent Reorganization (2025)

This project has been recently reorganized to improve code maintainability and clarity. The restructuring includes:

### 🔄 **File Renaming**
- **Meaningful English names**: All files now have descriptive, English names instead of mixed languages or unclear abbreviations
- **Consistent naming convention**: Files follow a clear naming pattern that reflects their functionality

### 📁 **Directory Structure**
- **`keras_models/`**: All Keras model architectures centralized in one location
- **`preprocessing/`**: Data preprocessing and augmentation utilities
- **`postprocessing/`**: Post-processing and result refinement tools
- **`utils/`**: General utility functions organized by purpose
- **`stardist3d/`**: StarDist3D implementation preserved as-is

### 📦 **Import Updates**
All import statements have been updated to reflect the new structure:
- Models now imported from `keras_models.{model_name}`
- Preprocessing functions from `preprocessing.{module_name}`  
- Utilities from `utils.{utility_type}`
- DataLoader moved to `preprocessing.dataloader`

### 🎯 **Benefits**
- **Better organization**: Related functionality grouped together
- **Improved maintainability**: Clear separation of concerns
- **Enhanced readability**: Meaningful file and directory names
- **Easier navigation**: Logical structure for finding specific components

--------

