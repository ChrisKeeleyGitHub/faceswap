#!/usr/bin/env python3
"""PatchGAN Autoencoder Model."""
from __future__ import annotations

import logging

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input  # pylint:disable=import-error
from tensorflow.keras.models import Model as KModel  # pylint:disable=import-error

from lib.model.nn_blocks import Conv2DOutput, Conv2DBlock, UpscaleBlock
from lib.model.discriminators.patch_gan import build_patch_discriminator
from ._base import ModelBase
from ._base.io import Weights
from ._base.settings import Optimizer

keras = tf.keras
logger = logging.getLogger(__name__)


class Model(ModelBase):
    """Faceswap autoencoder with a PatchGAN discriminator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer = "patch_gan"
        self.input_shape = (self.config["patch_size"], self.config["patch_size"], 3)
        self.learn_mask = self.config["learn_mask"]
        self.encoder_dim = 512
        self.G = None  # Generator/autoencoder
        self.D = None  # Discriminator

    def build(self, inputs):
        """Build the generator and discriminator models."""
        input_a, input_b = inputs[0], inputs[1]

        encoder = self.encoder()
        encoded_a = [encoder(input_a)]
        encoded_b = [encoder(input_b)]

        dec_a = self.decoder("a")(encoded_a)
        dec_b = self.decoder("b")(encoded_b)

        gen_outputs = dec_a + dec_b
        autoencoder = KModel(inputs, gen_outputs, name="autoencoder")

        discriminator = build_patch_discriminator(self.input_shape)
        d_a = discriminator(dec_a[0])
        d_b = discriminator(dec_b[0])

        combined_outputs = gen_outputs + [d_a, d_b]
        combined = KModel(inputs, combined_outputs, name=self.model_name)
        return autoencoder, discriminator, combined

    def build_model(self, inputs):
        self.G, self.D, combined = self.build(inputs)
        return combined

    def encoder(self):
        input_ = Input(shape=self.input_shape)
        x = input_
        x = Conv2DBlock(128, activation="leakyrelu")(x)
        x = Conv2DBlock(256, activation="leakyrelu")(x)
        x = Conv2DBlock(512, activation="leakyrelu")(x)
        x = Dense(self.encoder_dim)(Flatten()(x))
        x = Dense(4 * 4 * 1024)(x)
        x = Reshape((4, 4, 1024))(x)
        x = UpscaleBlock(512, activation="leakyrelu")(x)
        return KModel(input_, x, name="encoder")

    def decoder(self, side):
        input_ = Input(shape=(8, 8, 512))
        x = input_
        x = UpscaleBlock(256, activation="leakyrelu")(x)
        x = UpscaleBlock(128, activation="leakyrelu")(x)
        x = UpscaleBlock(64, activation="leakyrelu")(x)
        x = Conv2DOutput(3, 5, name=f"face_out_{side}")(x)
        outputs = [x]
        if self.learn_mask:
            y = input_
            y = UpscaleBlock(256, activation="leakyrelu")(y)
            y = UpscaleBlock(128, activation="leakyrelu")(y)
            y = UpscaleBlock(64, activation="leakyrelu")(y)
            y = Conv2DOutput(1, 5, name=f"mask_out_{side}")(y)
            outputs.append(y)
        return KModel(input_, outputs=outputs, name=f"decoder_{side}")

    def _compile_model(self) -> None:  # type:ignore[override]
        """Compile generator and discriminator models."""
        logger.debug("Compiling PatchGAN model")
        if self.state.model_needs_rebuild:
            self._model = self._settings.check_model_precision(self._model, self._state)
            self.D = self._settings.check_model_precision(self.D, self._state)

        d_optimizer = Optimizer(self.config["optimizer"],
                                 self.config["d_learning_rate"],
                                 self.config["autoclip"],
                                 10 ** int(self.config["epsilon_exponent"])).optimizer
        g_optimizer = Optimizer(self.config["optimizer"],
                                 self.config["learning_rate"],
                                 self.config["autoclip"],
                                 10 ** int(self.config["epsilon_exponent"])).optimizer
        if self._settings.use_mixed_precision:
            d_optimizer = self._settings.loss_scale_optimizer(d_optimizer)
            g_optimizer = self._settings.loss_scale_optimizer(g_optimizer)

        weights = Weights(self)
        weights.load(self._io.model_exists)
        weights.freeze()

        bce = keras.losses.BinaryCrossentropy(from_logits=True)
        self.D.compile(optimizer=d_optimizer, loss=bce)

        self.D.trainable = False
        self._loss.configure(self.G)
        losses = list(self._loss.functions.values()) + [bce, bce]
        loss_weights = [1.0] * len(self._loss.functions) + [self.config["adv_weight"],
                                                             self.config["adv_weight"]]
        self._model.compile(optimizer=g_optimizer,
                            loss=losses,
                            loss_weights=loss_weights)
        self._state.add_session_loss_names(self._loss.names + ["adv_a", "adv_b"])
        logger.debug("Compiled PatchGAN model")
