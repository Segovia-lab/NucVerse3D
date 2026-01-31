import keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K


@tf.keras.utils.register_keras_serializable(name="mse_loss")
def MSE_loss(y_true, y_pred):

    error = y_true - y_pred
    sqr_error = K.square(error)
    mean_sqr_error = K.mean(sqr_error)
    loss = tf.identity(mean_sqr_error, name="mse")
    return 10 * loss


@tf.keras.utils.register_keras_serializable(name="bce_loss")
def bce_loss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred,
    )
    loss = tf.identity(loss, name="bce_loss")
    return loss


@tf.keras.utils.register_keras_serializable(name="cce_weights_loss")
def cce_weights_loss(y_true, y_pred):
    weights = [0.36650165, 11.27804614, 5.46951642]
    weights = tf.constant(weights, dtype=tf.float32)
    cce = tf.keras.losses.CategoricalCrossentropy()
    cce_loss = cce(y_true, y_pred)
    weighted_loss = tf.reduce_sum(weights * cce_loss)
    return weighted_loss


@tf.keras.utils.register_keras_serializable(name="total_loss")
def total_loss(y_true, y_pred):
    loss_1 = bce_loss(y_true[0], y_pred[0])
    loss_2 = MSE_loss(y_true[1], y_pred[1])
    total_loss = loss_1 + loss_2
    return total_loss


@tf.keras.utils.register_keras_serializable(name="categorical_total_loss")
def categorical_total_loss(y_true, y_pred):
    # Definir los pesos aquí
    weights = [0.36650165, 11.27804614, 5.46951642]
    loss_1 = cce_weights_loss(y_true[0], y_pred[0], weights)
    loss_2 = MSE_loss(y_true[1], y_pred[1])
    total_loss = loss_1 + loss_2
    return total_loss


def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.8, gamma=2, smooth=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        BCE = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        BCE_EXP = tf.exp(-BCE)
        focal_loss = self.alpha * tf.pow((1 - BCE_EXP), self.gamma) * BCE
        return focal_loss


class TverskyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        TP = tf.reduce_sum(y_pred * y_true)
        FP = tf.reduce_sum((1 - y_true) * y_pred)
        FN = tf.reduce_sum(y_true * (1 - y_pred))

        Tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return 1 - Tversky


class FocalTverskyLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        alpha=0.8,
        gamma=2,
        smooth=1,
        alpha_tversky=0.3,
        beta_tversky=0.7,
        smooth_tversky=1,
    ):
        super(FocalTverskyLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, smooth=smooth)
        self.tversky_loss = TverskyLoss(
            alpha=alpha_tversky, beta=beta_tversky, smooth=smooth_tversky
        )

    def call(self, y_true, y_pred):
        focal_loss = self.focal_loss(y_true, y_pred)
        tversky_loss = self.tversky_loss(y_true, y_pred)
        combined_loss = 10 * focal_loss + tversky_loss
        return combined_loss


###########################################################
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1, name="dice_loss"):
        super(DiceLoss, self).__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        # Descomentar si tu modelo no contiene una capa de activación sigmoid o equivalente
        # y_pred = tf.nn.sigmoid(y_pred)

        # Aplanar los tensores de etiquetas y predicciones
        y_true = tf.keras.backend.flatten(y_true)
        y_pred = tf.keras.backend.flatten(y_pred)

        intersection = tf.reduce_sum(y_true * y_pred)
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (
            tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + self.smooth
        )
        return dice_loss


class BCELoss(tf.keras.losses.Loss):
    def __init__(self, name="bce_loss"):
        super(BCELoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        loss = tf.identity(loss, name=self.name)
        return loss


class DiceBCELoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1, name="dice_bce_loss"):
        super(DiceBCELoss, self).__init__(name=name)
        self.smooth = smooth

    def dice_loss(self, y_true, y_pred):
        # Aplanar los tensores de etiquetas y predicciones
        y_true = tf.keras.backend.flatten(y_true)
        y_pred = tf.keras.backend.flatten(y_pred)

        intersection = tf.reduce_sum(y_true * y_pred)
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (
            tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + self.smooth
        )
        return dice_loss

    def call(self, y_true, y_pred):
        # Calcular la pérdida de Dice
        dice_loss = self.dice_loss(y_true, y_pred)

        # Calcular la pérdida binaria de entropía cruzada
        bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Combinar ambas pérdidas
        combined_loss = dice_loss + 10 * bce_loss
        return combined_loss
