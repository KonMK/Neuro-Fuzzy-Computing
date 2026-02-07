import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from config import IMG_SIZE, NUM_CLASSES


def build_model(l2_strength=0.001):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(32, 3, use_bias=False, kernel_regularizer=regularizers.l2(l2_strength)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, use_bias=False, kernel_regularizer=regularizers.l2(l2_strength)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, use_bias=False, kernel_regularizer=regularizers.l2(l2_strength)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(l2_strength)),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax", kernel_regularizer=regularizers.l2(l2_strength))
    ])

    return model
