#!/usr/bin/env python3
"""Tests for PatchGAN trainer."""

import argparse
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("tensorflow")

from plugins.train.model.patch_gan import Model
from plugins.train.trainer.patch_gan import Trainer as PatchGANTrainer


class _Feeder:
    """Minimal feeder returning random batches."""

    def __init__(self, batch_size: int, patch: int) -> None:
        self.batch_size = batch_size
        self.patch = patch

    def get_batch(self):
        a = np.random.rand(self.batch_size, self.patch, self.patch, 3).astype("float32")
        b = np.random.rand(self.batch_size, self.patch, self.patch, 3).astype("float32")
        return [[a], [b]], [[a], [b]]


def _args():
    return argparse.Namespace(
        configfile=None,
        snapshot_interval=0,
        no_logs=True,
        use_lr_finder=False,
    )


def test_train_one_step(tmp_path):
    """Both generator and discriminator update without error."""
    model = Model(str(tmp_path), _args())
    model.build()

    trainer = PatchGANTrainer.__new__(PatchGANTrainer)
    trainer._model = model
    trainer._feeder = _Feeder(1, model.input_shape[0])
    trainer._log_tensorboard = lambda *a, **k: None
    trainer._collate_and_store_loss = lambda losses: losses
    trainer._print_loss = lambda *a, **k: None
    trainer._update_viewers = lambda *a, **k: None

    trainer.train_one_step(None, None)
