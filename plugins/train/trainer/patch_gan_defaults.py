#!/usr/bin/env python3
"""Default options for the PatchGAN trainer."""

from .original_defaults import _DEFAULTS as ORIGINAL_DEFAULTS, _HELPTEXT as ORIGINAL_HELPTEXT

_HELPTEXT = ORIGINAL_HELPTEXT.replace("Original", "PatchGAN")
_DEFAULTS = ORIGINAL_DEFAULTS
