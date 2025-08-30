#!/usr/bin/env python3
"""Tests for PatchGAN model."""

import argparse
import pytest

pytest.importorskip("tensorflow")

from plugins.train.model.patch_gan import Model


def _args():
    """Create minimal argparse Namespace for model."""
    return argparse.Namespace(
        configfile=None,
        snapshot_interval=0,
        no_logs=True,
        use_lr_finder=False,
    )


def test_build_and_compile(tmp_path):
    """Generator and discriminator models build and compile."""
    model = Model(str(tmp_path), _args())
    model.build()
    assert model.G is not None and model.G.optimizer is not None
    assert model.D is not None and model.D.optimizer is not None
