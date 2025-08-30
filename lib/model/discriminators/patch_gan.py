#!/usr/bin/env python3
"""PatchGAN discriminator constructor."""
from __future__ import annotations

import tensorflow as tf

# Fix intellisense/linting for tf.keras' import system
keras = tf.keras
layers = keras.layers
Model = keras.models.Model


def build_patch_discriminator(input_shape: tuple[int, int, int],
                              base_filters: int = 64) -> Model:
    """Build a PatchGAN discriminator model.

    Parameters
    ----------
    input_shape: tuple[int, int, int]
        Input shape ``(height, width, channels)``.
    base_filters: int, optional
        Number of filters for the first convolutional layer.

    Returns
    -------
    Model
        A Keras :class:`~tensorflow.keras.models.Model` producing an ``N x N`` score map.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    filters = base_filters

    # Downsampling blocks
    for _ in range(3):
        x = layers.Conv2D(filters, kernel_size=4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.BatchNormalization()(x)
        filters *= 2

    # Final score map
    outputs = layers.Conv2D(1, kernel_size=1, padding="same")(x)
    model = Model(inputs, outputs, name="PatchGANDiscriminator")
    return model
