#!/usr/bin/env python3
"""PatchGAN Trainer"""
from __future__ import annotations

import logging
import typing as T

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import errors_impl as tf_errors  # pylint:disable=no-name-in-module

from lib.utils import FaceswapError
from ._base import TrainerBase

if T.TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class Trainer(TrainerBase):
    """Trainer for models using a PatchGAN discriminator."""

    def train_one_step(self,
                       viewer: T.Callable[[np.ndarray, str], None] | None,
                       timelapse_kwargs: dict[T.Literal["input_a", "input_b", "output"],
                                              str] | None) -> None:
        """Run one training iteration for generator and discriminator."""
        self._model.state.increment_iterations()
        logger.trace("Training one step: (iteration: %s)", self._model.iterations)  # type: ignore
        snapshot_interval = self._model.command_line_arguments.snapshot_interval
        do_snapshot = (snapshot_interval != 0 and
                       self._model.iterations - 1 >= snapshot_interval and
                       (self._model.iterations - 1) % snapshot_interval == 0)

        model_inputs, model_targets = self._feeder.get_batch()
        flat_inputs = [item for sub in model_inputs for item in sub]
        flat_targets = [item for sub in model_targets for item in sub]

        batch_size = flat_inputs[0].shape[0]
        patch_shape = self._model.D.output_shape[1:]
        real_y = np.ones((batch_size,) + patch_shape, dtype=np.float32)
        fake_y = np.zeros((batch_size,) + patch_shape, dtype=np.float32)

        generator = getattr(self._model, "G")
        discriminator = getattr(self._model, "discriminator",
                                getattr(self._model, "D"))

        fake_outputs = generator.predict_on_batch(flat_inputs)
        if self._model.config["learn_mask"]:
            fake_a, fake_b = fake_outputs[0], fake_outputs[2]
        else:
            fake_a, fake_b = fake_outputs[0], fake_outputs[1]
        real_a = model_targets[0][0]
        real_b = model_targets[1][0]

        discriminator.trainable = True
        discriminator.train_on_batch(real_a, real_y)
        discriminator.train_on_batch(fake_a, fake_y)
        discriminator.train_on_batch(real_b, real_y)
        discriminator.train_on_batch(fake_b, fake_y)
        discriminator.trainable = False

        combined_targets = flat_targets + [real_y, real_y]
        try:
            loss = self._model.model.train_on_batch(flat_inputs, combined_targets)
        except tf_errors.ResourceExhaustedError as err:
            msg = ("You do not have enough GPU memory available to train the selected model at "
                   "the selected settings. You can try a number of things:"\
                   "\n1) Close any other application that is using your GPU (web browsers are "
                   "particularly bad for this)."\
                   "\n2) Lower the batchsize (the amount of images fed into the model each "
                   "iteration)."\
                   "\n3) Try enabling 'Mixed Precision' training."\
                   "\n4) Use a more lightweight model, or select the model's 'LowMem' option "
                   "(in config) if it has one.")
            raise FaceswapError(msg) from err

        self._log_tensorboard(loss)

        if self._model.config["learn_mask"]:
            reordered = [loss[1], loss[2], loss[5], loss[3], loss[4], loss[6]]
        else:
            reordered = [loss[1], loss[3], loss[2], loss[4]]

        loss_vals = self._collate_and_store_loss(reordered)
        self._print_loss(loss_vals)
        if do_snapshot:
            self._model.io.snapshot()
        self._update_viewers(viewer, timelapse_kwargs)
