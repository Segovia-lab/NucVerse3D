"""
Configuration file for 3D Nuclei Segmentation Project
=====================================================

This file contains all configuration parameters and constants used across the project.
Centralized configuration allows for easy parameter tuning and consistency across modules.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data paths
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
MODELS_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"
PREDICTION_ROOT = PROJECT_ROOT / "prediction"

# Default image directory (for prediction tasks)
DEFAULT_IMAGE_DIR = (
    DATA_ROOT / "c_elegans_nuclei" / "c_elegans_nuclei" / "test" / "images"
)

# For scaling moddel, base average radius
AVG_RADIUS = 10.0  # Do not change unless you trained the models with a diffrent value 

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model architecture parameters
INPUT_SHAPE = (64, 128, 128, 1)  # (depth, height, width, channels)
FILTER_NUM = 32  # Number of initial filters (factor of 2)
DROPOUT_RATE = 0.3  # Dropout rate for regularization
BATCH_NORM = True  # Whether to use batch normalization
NUM_CLASSES = 2  # Number of classes (background + nuclei)

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Training hyperparameters
BATCH_SIZE = 8  # Batch size for training
EPOCHS = 300  # Number of training epochs
LEARNING_RATE = 1e-3  # Learning rate for optimizer
STEP_PER_EPOCH = 0  # Steps per epoch

# Callback parameters
REDUCE_LR_FACTOR   = 0.5  # Factor to reduce learning rate
REDUCE_LR_PATIENCE = 10  # Patience for learning rate reduction
MONITOR_METRIC     = "val_loss"  # Metric to monitor for callbacks
MONITOR_MODE = "min"  # Mode for monitoring (min/max)

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 15  # Patience for early stopping
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum change to qualify as improvement
EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True  # Restore best weights when stopped

# Reproducibility parameters
RANDOM_SEED = -1  # Seed for reproducibility
SET_DETERMINISTIC_OPS = False  # Enable deterministic operations

# Model checkpoint filenames
BEST_MODEL_FILENAME = "best_model.weights.h5"
FINAL_MODEL_FILENAME = "modelSave.weights.h5"
TRAINING_HISTORY_FILENAME = "training_history.csv"
TRAINING_TIME_FILENAME = "training_time.txt"

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================

# Patch processing parameters
PATCH_SIZE = (64, 128, 128)  # Size of patches for processing
STEP_SIZE_TRAINING = (32, 96, 96)  # Step size for training patch generation 
STEP_SIZE_PREDICTION = (16, 64, 64)  # Step size for prediction patch generation

# Data filtering parameters
MAX_PERCENT_SIGNAL = 5  # Maximum percentage of signal for patch filtering
SMALL_OBJECT_VOX = 20  # Minimum voxels for small object filtering
VOX_THRESHOLD = 20  # Voxel threshold for nuclei segmentation
TEST_SIZE = 0.2  # Test size for train/validation split

# Data augmentation parameters
AUGMENTATION_PROBABILITY = 0.75  # Probability of applying augmentation
BRIGHTNESS_RANGE = (0.8, 1.2)  # Range for brightness adjustment (0.8, 1.2)
ZOOM_RANGE = (0.7, 1.2)  # Range for zoom augmentation
ROTATION_MAX_ANGLE = 45  # Maximum rotation angle in degrees

# =============================================================================
# IMAGE PROCESSING CONFIGURATION
# =============================================================================
PREPROCESSING_ENABLED = False  # If False, apply just normalization

# Normalization parameters
NORMALIZE_P_LOWER = 2  # Lower percentile for normalization
NORMALIZE_P_UPPER = 99.8  # Upper percentile for normalization

# Intensity correction parameters
INTENSITY_SIGMA = 5  # Sigma for Gaussian filter in intensity correction
INTENSITY_THRESHOLD = 0.01  # Threshold for intensity correction
INTENSITY_DOWNSCALE = 8  # Downscale factor for intensity correction

# PSF (Point Spread Function) parameters
PSF_SIZE = 15  # Size of PSF kernel
PSF_SIGMA = 1  # Sigma for PSF Gaussian kernel

# Deconvolution parameters
DECONVOLUTION_JOBS = None  # Number of parallel jobs (None = CPU count // 2)

# Border processing parameters
#BORDER_SIZE = 1  # Size of borders for label processing

# =============================================================================
# PREDICTION CONFIGURATION
# =============================================================================

# Prediction processing parameters
PREDICTION_BATCH_SIZE = 16  # Batch size for prediction
PREDICTION_NUM_DIV = 32  # Number of divisions for prediction processing
#PREDICTION_OVERLAP_THRESHOLD = 0.5  # Overlap threshold for patch blending

# Gradient descent parameters
GRADIENT_DESCENT_LEARNING_RATE = 0.2  # Learning rate for gradient descent
GRADIENT_DESCENT_N_ITER = 200          # Number of iterations for gradient descent
GRADIENT_DESCENT_TOLERANCE = 1e-6      # Tolerance for gradient descent convergence
GRADIENT_DESCENT_MOMENTUM = 0.8        # Momentum for gradient descent

# Centroid estimation parameters
CENTROID_GAUSSIAN_SIGMA = 1  # Sigma for Gaussian filter on centroids
CENTROID_THRESHOLD_STD_MULTIPLIER = 5  # Multiplier for threshold calculation

# Label calculation parameters
LABEL_RADIUS = 5  # Radius for label assignment

# =============================================================================
# SMOOTH PREDICTION CONFIGURATION
# =============================================================================

# Smooth prediction parameters
#SMOOTH_WINDOW_SIZE = 64  # Window size for smooth prediction
#SMOOTH_SUBDIVISIONS = 2  # Number of subdivisions for smooth prediction
#SMOOTH_MAX_BATCH = 10  # Maximum batch size for smooth prediction
#SMOOTH_WINDOW_MODE = "spline"  # Window mode (spline/gaus)

# Rotation and flip axes for smooth prediction
#SMOOTH_ROT_AXES = [(0, 1), (0, 2), (1, 2)]  # Rotation axes pairs
#SMOOTH_FLIP_AXES = [(1,), (2,)]  # Flip axes

# =============================================================================
# FILE EXTENSIONS AND FORMATS
# =============================================================================

# Supported file extensions
IMAGE_EXTENSIONS = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
MODEL_EXTENSIONS = [".h5", ".hdf5"]
DATA_EXTENSIONS = [".hdf5", ".h5"]

# Data types
FLOAT_DTYPE = "float16"  # Default float data type
INT_DTYPE = "uint16"  # Default integer data type
LABEL_DTYPE = "int16"  # Data type for labels

# =============================================================================
# MULTIPROCESSING CONFIGURATION
# =============================================================================

# Parallel processing parameters
DEFAULT_N_JOBS = None  # Number of parallel jobs (None = auto-detect)
PROGRESS_BAR_ENABLED = True  # Whether to show progress bars

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Logging parameters
LOG_LEVEL = "INFO"  # Logging level
VERBOSE = 2  # Verbosity level for training

# =============================================================================
# VALIDATION AND DEBUGGING
# =============================================================================

# Debug parameters
DEBUG_PLOTS = False  # Whether to generate debug plots
KEEP_TEMP_FILES = False  # Whether to keep temporary files
TEMP_FOLDER_NAME = "tmp"  # Name of temporary folder

# Memory management
MEMORY_GROWTH_ENABLED = True  # Enable GPU memory growth
CLEAR_SESSION_AFTER_PREDICTION = True  # Clear TensorFlow session after prediction

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_training_path(dataset_name):
    """Get training data path for a specific dataset."""
    return str(PROCESSED_DATA_ROOT / f"{dataset_name}" / "data_train.hdf5")


def get_validation_path(dataset_name):
    """Get validation data path for a specific dataset."""
    return str(PROCESSED_DATA_ROOT / f"{dataset_name}" / "data_val.hdf5")


def get_model_folder(model_name):
    """Get model folder path for a specific model."""
    return str(MODELS_ROOT / model_name)


def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_ROOT,
        RAW_DATA_ROOT,
        PROCESSED_DATA_ROOT,
        MODELS_ROOT,
        RESULTS_ROOT,
        PREDICTION_ROOT,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_n_jobs():
    """Get number of parallel jobs based on CPU count."""
    if DEFAULT_N_JOBS is None:
        cpu_count = os.cpu_count()
        return max(1, cpu_count // 2) if cpu_count else 1
    return DEFAULT_N_JOBS


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================


def validate_config():
    """Validate configuration parameters."""
    # Validate patch size and step size compatibility
    assert all(
        p >= s for p, s in zip(PATCH_SIZE, STEP_SIZE_TRAINING)
    ), "Patch size must be >= step size for training"

    assert all(
        p >= s for p, s in zip(PATCH_SIZE, STEP_SIZE_PREDICTION)
    ), "Patch size must be >= step size for prediction"

    # Validate input shape compatibility
    assert (
        INPUT_SHAPE[:-1] == PATCH_SIZE
    ), "Input shape (excluding channels) must match patch size"

    # Validate learning rate
    assert 0 < LEARNING_RATE <= 1, "Learning rate must be between 0 and 1"

    # Validate dropout rate
    assert 0 <= DROPOUT_RATE <= 1, "Dropout rate must be between 0 and 1"

    # Validate batch size
    assert BATCH_SIZE > 0, "Batch size must be positive"

    print("Configuration validation passed!")


# Run validation when module is imported
if __name__ == "__main__":
    validate_config()
    create_directories()
    print("Configuration loaded successfully!")
