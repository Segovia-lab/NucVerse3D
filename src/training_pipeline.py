# -*- coding: utf-8 -*-
"""
CLI for 3D Nuclei Segmentation Training Pipeline
===============================================

Created on Fri Jul 26 16:45:41 2024

@author: ws2
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]     = "0"  # Only the second GPU will be visible
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from utils.loss_functions import FocalTverskyLoss, MSE_loss, dice_coef
from keras_models.attention_unet_3d import Attention_ResUNet_3D
from preprocessing.dataloader import DataLoader
import time
from utils.memory_profiler import get_model_memory_usage
import logging
import typer

import config

# Create typer app
app = typer.Typer(
    name="nuclei-train",
    help="Train Attention ResUNet 3D models for 3D nuclei segmentation",
    no_args_is_help=True,
)


def setup_reproducibility():
    """
    Set up reproducibility by setting random seeds for Python, NumPy, and TensorFlow.
    """
    import random

    # Set Python random seed
    random.seed(config.RANDOM_SEED)

    # Set NumPy random seed
    np.random.seed(config.RANDOM_SEED)

    # Set TensorFlow random seed
    tf.random.set_seed(config.RANDOM_SEED)

    # Enable deterministic operations if requested
    if config.SET_DETERMINISTIC_OPS:
        tf.config.experimental.enable_op_determinism()

    print(f"Reproducibility setup completed with seed: {config.RANDOM_SEED}")


# Check si es que la gpu se esta ocupando
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, config.MEMORY_GROWTH_ENABLED)
if gpus:
    print(f"GPUs detectadas: {[gpu.name for gpu in gpus]}")
else:
    print("No se detectaron GPUs.")


def create_model(input_shape, dropout_rate, batch_norm, filter_num, learning_rate):
    """
    Crea y compila el modelo Attention ResUNet 3D.
    """

    model = Attention_ResUNet_3D(
        input_shape,
        dropout_rate=dropout_rate,
        batch_norm=batch_norm,
        FILTER_NUM=filter_num,
    )
    model.compile(
        optimizer=Adam(learning_rate),
        loss=[FocalTverskyLoss(), MSE_loss],
        metrics=["mae", dice_coef],
    )
    return model


def configure_callbacks(model_folder):
    """
    Configura los callbacks para el entrenamiento.
    """
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_folder, config.BEST_MODEL_FILENAME),
        save_weights_only=True,
        monitor=config.MONITOR_METRIC,
        mode=config.MONITOR_MODE,
        save_best_only=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor=config.MONITOR_METRIC,
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        verbose=config.VERBOSE,
    )
    early_stopping = EarlyStopping(
        monitor=config.MONITOR_METRIC,
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode=config.MONITOR_MODE,
        restore_best_weights=config.EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
        verbose=config.VERBOSE,
    )
    return [checkpoint, reduce_lr, early_stopping]


def save_training_results(history, model, model_folder):
    """
    Guarda el historial de entrenamiento y el modelo final.
    """
    # Guardar historial
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(
        os.path.join(model_folder, config.TRAINING_HISTORY_FILENAME), index=False
    )
    print("Historial de entrenamiento guardado.")

    # Guardar modelo final
    model.save(os.path.join(model_folder, config.FINAL_MODEL_FILENAME))
    print("Modelo final guardado.")


# ============================
# CLI Command
# ============================
@app.command()
def train(
    train_path: str = typer.Argument(..., help="Path to training data directory"),
    val_path: str = typer.Argument(..., help="Path to validation data directory"),
    model_name: str = typer.Option(
        None, "--model-name", "-m", help="Custom model name (defaults to timestamp)"
    ),
):
    """
    Train Attention ResUNet 3D model for nuclei segmentation.

    Examples:
        nuclei-train train data/processed/train data/processed/val
        nuclei-train train data/processed/train data/processed/val --model-name my_model
    """

    # Validate inputs
    if not os.path.exists(train_path):
        typer.echo(f"❌ Training path not found: {train_path}", err=True)
        raise typer.Exit(1)
    if not os.path.exists(val_path):
        typer.echo(f"❌ Validation path not found: {val_path}", err=True)
        raise typer.Exit(1)

    # Show configuration
    typer.echo("🚀 Nuclei Segmentation Training:")
    typer.echo(f"   📁 Train: {train_path}")
    typer.echo(f"   📁 Val: {val_path}")
    typer.echo(f"   🏷️  Model: {model_name or 'auto-generated'}")
    typer.echo(f"   📁 Output: {config.MODELS_ROOT}")
    typer.echo(f"   🔢 Epochs: {config.EPOCHS}")
    typer.echo(f"   📦 Batch size: {config.BATCH_SIZE}")
    typer.echo(f"   📈 Learning rate: {config.LEARNING_RATE}")
    typer.echo(f"   🎯 Dropout rate: {config.DROPOUT_RATE}")
    typer.echo(f"   🔧 Filters: {config.FILTER_NUM}")
    typer.echo(f"   ⏹️  Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    typer.echo(f"   🎲 Random seed: {config.RANDOM_SEED}")
    typer.echo()

    # Generate default model name if none provided
    if model_name is None:
        from datetime import datetime

        model_name = f"nuclei_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Call main training function with custom paths and model name
    main(train_path=train_path, val_path=val_path, model_name=model_name)


def main(train_path=None, val_path=None, model_name=None):
    """
    Función principal que configura, entrena y guarda el modelo.

    Args:
        train_path: Custom training data path (optional)
        val_path: Custom validation data path (optional)
        model_name: Custom model name (optional)
    """
    # Setup reproducibility
    if config.RANDOM_SEED > 0:
        setup_reproducibility()

    # Crear carpeta del modelo si no existe
    model_folder = config.get_model_folder(model_name)
    os.makedirs(model_folder, exist_ok=True)

    # Cargar los datasets
    data_loader = DataLoader(
        batch_size=config.BATCH_SIZE, image_shape=config.INPUT_SHAPE[:-1]
    )

    # Use the provided paths directly
    training_path = train_path
    validation_path = val_path

    train_dataset = data_loader.get_dataset(training_path, augment=True)
    val_dataset = data_loader.get_dataset(validation_path, augment=False)

    # Crear y compilar el modelo
    model = create_model(
        config.INPUT_SHAPE,
        config.DROPOUT_RATE,
        config.BATCH_NORM,
        config.FILTER_NUM,
        config.LEARNING_RATE,
    )

    # Evaluar el uso de memoria esperado
    #estimated_memory = get_model_memory_usage(config.BATCH_SIZE, model)
    #print(f"Uso estimado de memoria del modelo: {estimated_memory:.2f} GB.")
    
    # Configurar callbacks
    callbacks = configure_callbacks(model_folder)

    # Registrar tiempo de inicio
    start_time = time.time()

    print("Training started...")

    if config.STEP_PER_EPOCH > 0:
        print("Training on reduced dataset...")

        history_small = model.fit(
            train_dataset,
            epochs=config.EPOCHS,
    	    steps_per_epoch=config.STEP_PER_EPOCH,  # Train on only STEP_PER_EPOCH batches per epoch
    	    validation_data=val_dataset,
            callbacks=callbacks,
            verbose=2,
        )

    print("Training on full dataset...")
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=2,
    )
    print("Entrenamiento finalizado.")

    # Registrar tiempo de finalización
    end_time = time.time()

    # Calcular tiempo total
    total_training_time = end_time - start_time
    print(f"Tiempo total de entrenamiento: {total_training_time:.2f} segundos.")

    # Guardar el tiempo en un archivo
    time_log_path = os.path.join(model_folder, config.TRAINING_TIME_FILENAME)
    with open(time_log_path, "w") as f:
        f.write(f"Tiempo total de entrenamiento: {total_training_time:.2f} segundos.\n")

    # Guardar resultados y modelo
    save_training_results(history, model, model_folder)


if __name__ == "__main__":
    typer.run(train)
