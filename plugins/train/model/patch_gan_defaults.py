#!/usr/bin/env python3
"""Default options for the PatchGAN model."""

_HELPTEXT = "Original autoencoder with a PatchGAN discriminator."

_DEFAULTS = dict(
    patch_size=dict(
        default=64,
        info="Input and output resolution for the generator.",
        datatype=int,
        rounding=16,
        min_max=(32, 256),
        choices=[],
        gui_radio=False,
        group="settings",
        fixed=True,
    ),
    adv_weight=dict(
        default=0.01,
        info="Weight applied to adversarial loss when training the generator.",
        datatype=float,
        rounding=3,
        min_max=(0.0, 1.0),
        choices=[],
        gui_radio=False,
        group="settings",
        fixed=False,
    ),
    learning_rate=dict(
        default=5e-05,
        info="Learning rate for the generator.",
        datatype=float,
        rounding=7,
        min_max=(1e-06, 1e-03),
        choices=[],
        gui_radio=False,
        group="settings",
        fixed=True,
    ),
    d_learning_rate=dict(
        default=1e-04,
        info="Learning rate for the discriminator.",
        datatype=float,
        rounding=7,
        min_max=(1e-06, 1e-03),
        choices=[],
        gui_radio=False,
        group="settings",
        fixed=True,
    ),
)
