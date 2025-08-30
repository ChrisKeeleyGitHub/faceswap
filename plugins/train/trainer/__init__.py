#!/usr/bin/env python3
"""Faceswap training trainers."""

from .original import Trainer as Original
from .patch_gan import Trainer as PatchGan

__all__ = ["Original", "PatchGan"]
