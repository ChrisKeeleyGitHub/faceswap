#!/usr/bin/env python3
""" Handles the creation of display images for preview window and timelapses """
from __future__ import annotations

import logging
import time
import typing as T
import os
from itertools import combinations

import cv2
import numpy as np
import torch

from lib.image import hex_to_rgb
from lib.utils import FaceswapError, get_folder, get_image_paths

if T.TYPE_CHECKING:
    from keras import KerasTensor
    from lib.training import Feeder
    from plugins.train.model._base import ModelBase

logger = logging.getLogger(__name__)
SideLiteral = T.Literal["a", "b", "c"]


def _get_model_sides(model: "ModelBase") -> tuple[str, ...]:
    """Return the configured sides for the provided model."""

    sides = getattr(model, "sides", None)
    if sides:
        return tuple(sides)
    return ("a", "b")


class Samples():
    """ Compile samples for display for preview and time-lapse

    Parameters
    ----------
    model: plugin from :mod:`plugins.train.model`
        The selected model that will be running this trainer
    coverage_ratio: float
        Ratio of face to be cropped out of the training image.
    mask_opacity: int
        The opacity (as a percentage) to use for the mask overlay
    mask_color: str
        The hex RGB value to use the mask overlay

    Attributes
    ----------
    images: dict
        The :class:`numpy.ndarray` training images for generating previews on each side. The
        dictionary should contain 2 keys ("a" and "b") with the values being the training images
        for generating samples corresponding to each side.
    """
    def __init__(self,
                 model: ModelBase,
                 coverage_ratio: float,
                 mask_opacity: int,
                 mask_color: str) -> None:
        logger.debug("Initializing %s: model: '%s', coverage_ratio: %s, mask_opacity: %s, "
                     "mask_color: %s)",
                     self.__class__.__name__, model, coverage_ratio, mask_opacity, mask_color)
        self._model = model
        self._sides: tuple[str, ...] = _get_model_sides(model)
        self._display_mask = model.config["learn_mask"] or model.config["penalized_mask_loss"]
        self.images: dict[SideLiteral, list[np.ndarray]] = {}
        self._coverage_ratio = coverage_ratio
        self._mask_opacity = mask_opacity / 100.0
        self._mask_color = np.array(hex_to_rgb(mask_color))[..., 2::-1] / 255.
        logger.debug("Initialized %s", self.__class__.__name__)

    def toggle_mask_display(self) -> None:
        """ Toggle the mask overlay on or off depending on user input. """
        if not (self._model.config["learn_mask"] or self._model.config["penalized_mask_loss"]):
            return
        display_mask = not self._display_mask
        print("\x1b[2K", end="\r")  # Clear last line
        logger.info("Toggling mask display %s...", "on" if display_mask else "off")
        self._display_mask = display_mask

    @property
    def sides(self) -> tuple[str, ...]:
        """The configured sides for the preview samples."""

        return self._sides

    def show_sample(self) -> np.ndarray | None:
        """ Compile a preview image.

        Returns
        -------
        :class:`numpy.ndarray` | None
            A compiled preview image ready for display or saving or ``None`` if no data is
            available
        """
        logger.debug("Showing sample")
        feeds: dict[SideLiteral, np.ndarray] = {}
        model_input_shapes = list(self._model.model.input_shape)
        for idx, side in enumerate(self._sides):
            if side not in self.images or not self.images[side]:
                continue
            feed = self.images[side][0]
            shape_idx = min(idx, len(model_input_shapes) - 1)
            input_shape = model_input_shapes[shape_idx][1:]
            if input_shape[0] / feed.shape[1] != 1.0:
                feeds[T.cast(SideLiteral, side)] = self._resize_sample(
                    T.cast(SideLiteral, side), feed, input_shape[0])
            else:
                feeds[T.cast(SideLiteral, side)] = feed

        if not feeds:
            return None

        preds = self._get_predictions(feeds)
        return self._compile_preview(preds)

    @classmethod
    def _resize_sample(cls,
                       side: SideLiteral,
                       sample: np.ndarray,
                       target_size: int) -> np.ndarray:
        """ Resize a given image to the target size.

        Parameters
        ----------
        side: str
            The side that the samples are being generated for
        sample: :class:`numpy.ndarray`
            The sample to be resized
        target_size: int
            The size that the sample should be resized to

        Returns
        -------
        :class:`numpy.ndarray`
            The sample resized to the target size
        """
        scale = target_size / sample.shape[1]
        if scale == 1.0:
            # cv2 complains if we don't do this :/
            return np.ascontiguousarray(sample)
        logger.debug("Resizing sample: (side: '%s', sample.shape: %s, target_size: %s, scale: %s)",
                     side, sample.shape, target_size, scale)
        interpn = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        retval = np.array([cv2.resize(img, (target_size, target_size), interpolation=interpn)
                           for img in sample])
        logger.debug("Resized sample: (side: '%s' shape: %s)", side, retval.shape)
        return retval

    def _predict_for_inputs(self,
                            model_inputs: list[np.ndarray],
                            sides: list[str]) -> dict[SideLiteral, np.ndarray]:
        """Run the model for the provided inputs and collate outputs per side."""

        if not sides:
            return {}

        with torch.inference_mode():
            outputs = self._model.model(model_inputs)

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        outputs_np = [output.cpu().detach().numpy() for output in outputs]
        num_sides = len(sides)
        if num_sides == 0:
            return {}
        if len(outputs_np) % num_sides != 0:
            raise FaceswapError("Model outputs do not align with configured sides")
        outputs_per_side = len(outputs_np) // num_sides

        retval: dict[SideLiteral, np.ndarray] = {}
        for idx, side in enumerate(sides):
            start = idx * outputs_per_side
            side_outputs = outputs_np[start:start + outputs_per_side]
            if not side_outputs:
                continue
            sizes = {output.shape[1] for output in side_outputs}
            if len(sizes) > 1:
                max_size = max(sizes)
                side_outputs = [output for output in side_outputs if output.shape[1] == max_size]
            merged = (side_outputs[0] if len(side_outputs) == 1
                      else np.concatenate(side_outputs, axis=-1))
            retval[T.cast(SideLiteral, side)] = merged

        return retval

    def _get_predictions(self, feeds: dict[SideLiteral, np.ndarray]
                         ) -> dict[tuple[str, str], np.ndarray]:
        """Feed the samples to the model and return predictions for each side."""

        logger.debug("Getting Predictions")
        active_sides = [side for side in self._sides if side in feeds]
        if len(active_sides) < 2:
            raise FaceswapError("At least two sides are required to generate preview predictions")

        inputs = [feeds[T.cast(SideLiteral, side)] for side in active_sides]
        standard = self._predict_for_inputs(inputs, active_sides)

        preds: dict[tuple[str, str], np.ndarray] = {}
        for side in active_sides:
            preds[(side, side)] = standard[T.cast(SideLiteral, side)]

        for idx, jdx in combinations(range(len(active_sides)), 2):
            swap_inputs = inputs[:]
            swap_inputs[idx], swap_inputs[jdx] = swap_inputs[jdx], swap_inputs[idx]
            swapped = self._predict_for_inputs(swap_inputs, active_sides)
            src = active_sides[idx]
            dst = active_sides[jdx]
            preds[(src, dst)] = swapped[T.cast(SideLiteral, dst)]
            preds[(dst, src)] = swapped[T.cast(SideLiteral, src)]

        logger.debug("Returning predictions: %s",
                     {f"{src}_{dst}": val.shape for (src, dst), val in preds.items()})
        return preds

    def _compile_preview(self, predictions: dict[tuple[str, str], np.ndarray]
                         ) -> np.ndarray:
        """ Compile predictions and images into the final preview image.

        Parameters
        ----------
        predictions: dict[(str, str), :class:`numpy.ndarray`]
            The predictions from the model keyed by (source_side, target_side)

        Returns
        -------
        :class:`numpy.ndarry`
            A compiled preview image ready for display or saving
        """
        panels: list[np.ndarray] = []

        for side, samples in self.images.items():
            side_key = T.cast(SideLiteral, side)
            if (side, side) not in predictions:
                logger.debug("Skipping side '%s' due to missing self prediction", side)
                continue
            other_sides = [other for other in self._sides
                           if other != side and (other, side) in predictions]
            pred_list = [predictions[(side, side)]]
            pred_list.extend(predictions[(other, side)] for other in other_sides)
            display = self._to_full_frame(side_key, samples, pred_list)
            if not display:
                continue

            rows = [np.concatenate(row_images, axis=1)
                    for row_images in zip(*display)]
            if not rows:
                continue
            grid = np.concatenate(rows, axis=0)
            header = self._get_headers(side_key,
                                       display[0].shape[1],
                                       other_sides)
            panels.append(np.concatenate((header, grid), axis=0))

        if not panels:
            raise FaceswapError("No preview panels could be generated")

        figure = np.concatenate(panels, axis=0)

        logger.debug("Compiled sample")
        return np.clip(figure * 255, 0, 255).astype('uint8')

    def _to_full_frame(self,
                       side: SideLiteral,
                       samples: list[np.ndarray],
                       predictions: list[np.ndarray]) -> list[np.ndarray]:
        """ Patch targets and prediction images into images of model output size.

        Parameters
        ----------
        side: str
            The side that these samples are for
        samples: list
            List of :class:`numpy.ndarray` of feed images and sample images
        predictions: list
            List of :class: `numpy.ndarray` of predictions from the model

        Returns
        -------
        list
            The images resized and collated for display in the preview frame
        """
        logger.debug("side: '%s', number of sample arrays: %s, prediction.shapes: %s)",
                     side, len(samples), [pred.shape for pred in predictions])
        faces, full = samples[:2]

        if self._model.color_order.lower() == "rgb":  # Switch color order for RGB model display
            full = full[..., ::-1]
            faces = faces[..., ::-1]
            predictions = [pred[..., 2::-1] for pred in predictions]

        full = self._process_full(side, full, predictions[0].shape[1], (0., 0., 1.0))
        images = [faces] + predictions

        if self._display_mask:
            images = self._compile_masked(images, samples[-1])
        elif self._model.config["learn_mask"]:
            # Remove masks when learn mask is selected but mask toggle is off
            images = [batch[..., :3] for batch in images]

        images = [self._overlay_foreground(full.copy(), image) for image in images]

        return images

    def _process_full(self,
                      side: SideLiteral,
                      images: np.ndarray,
                      prediction_size: int,
                      color: tuple[float, float, float]) -> np.ndarray:
        """ Add a frame overlay to preview images indicating the region of interest.

        This applies the red border that appears in the preview images.

        Parameters
        ----------
        side: {"a" or "b"}
            The side that these samples are for
        images: :class:`numpy.ndarray`
            The input training images to to process
        prediction_size: int
            The size of the predicted output from the model
        color: tuple
            The (Blue, Green, Red) color to use for the frame

        Returns
        -------
        :class:`numpy,ndarray`
            The input training images, sized for output and annotated for coverage
        """
        logger.debug("full_size: %s, prediction_size: %s, color: %s",
                     images.shape[1], prediction_size, color)

        display_size = int((prediction_size / self._coverage_ratio // 2) * 2)
        images = self._resize_sample(side, images, display_size)  # Resize targets to display size
        padding = (display_size - prediction_size) // 2
        if padding == 0:
            logger.debug("Resized background. Shape: %s", images.shape)
            return images

        length = display_size // 4
        t_l, b_r = (padding - 1, display_size - padding)
        for img in images:
            cv2.rectangle(img, (t_l, t_l), (t_l + length, t_l + length), color, 1)
            cv2.rectangle(img, (b_r, t_l), (b_r - length, t_l + length), color, 1)
            cv2.rectangle(img, (b_r, b_r), (b_r - length, b_r - length), color, 1)
            cv2.rectangle(img, (t_l, b_r), (t_l + length, b_r - length), color, 1)
        logger.debug("Overlayed background. Shape: %s", images.shape)
        return images

    def _compile_masked(self, faces: list[np.ndarray], masks: np.ndarray) -> list[np.ndarray]:
        """ Add the mask to the faces for masked preview.

        Places an opaque red layer over areas of the face that are masked out.

        Parameters
        ----------
        faces: list
            The :class:`numpy.ndarray` sample faces and predictions that are to have the mask
            applied
        masks: :class:`numpy.ndarray`
            The masks that are to be applied to the faces

        Returns
        -------
        list
            List of :class:`numpy.ndarray` faces with the opaque mask layer applied
        """
        orig_masks = 1. - masks
        masks3: list[np.ndarray] | np.ndarray = []

        if faces[-1].shape[-1] == 4:  # Mask contained in alpha channel of predictions
            pred_masks = [1. - face[..., -1][..., None] for face in faces[-2:]]
            faces[-2:] = [face[..., :-1] for face in faces[-2:]]
            masks3 = [orig_masks, *pred_masks]
        else:
            masks3 = np.repeat(np.expand_dims(orig_masks, axis=0), 3, axis=0)

        retval: list[np.ndarray] = []
        overlays3 = np.ones_like(faces) * self._mask_color
        for previews, overlays, compiled_masks in zip(faces, overlays3, masks3):
            compiled_masks *= self._mask_opacity
            overlays *= compiled_masks
            previews *= (1. - compiled_masks)
            retval.append(previews + overlays)
        logger.debug("masked shapes: %s", [faces.shape for faces in retval])
        return retval

    @classmethod
    def _overlay_foreground(cls, backgrounds: np.ndarray, foregrounds: np.ndarray) -> np.ndarray:
        """ Overlay the preview images into the center of the background images

        Parameters
        ----------
        backgrounds: :class:`numpy.ndarray`
            Background images for placing the preview images onto
        backgrounds: :class:`numpy.ndarray`
            Preview images for placing onto the background images

        Returns
        -------
        :class:`numpy.ndarray`
            The preview images compiled into the full frame size for each preview
        """
        offset = (backgrounds.shape[1] - foregrounds.shape[1]) // 2
        for foreground, background in zip(foregrounds, backgrounds):
            background[offset:offset + foreground.shape[0],
                       offset:offset + foreground.shape[1], :3] = foreground
        logger.debug("Overlayed foreground. Shape: %s", backgrounds.shape)
        return backgrounds

    @classmethod
    def _get_headers(cls, side: SideLiteral, width: int, other_sides: list[str]) -> np.ndarray:
        """ Set header row for the final preview frame

        Parameters
        ----------
        side: {"a" or "b"}
            The side that the headers should be generated for
        width: int
            The width of each column in the preview frame

        Returns
        -------
        :class:`numpy.ndarray`
            The column headings for the given side
        """
        logger.debug("side: '%s', width: %s, other_sides: %s",
                     side, width, other_sides)
        titles = [f"Original ({side.upper()})", f"{side.upper()} > {side.upper()}"]
        titles.extend(f"{other.upper()} > {side.upper()}" for other in other_sides)
        height = int(width / 4.5)
        total_width = width * len(titles)
        logger.debug("height: %s, total_width: %s", height, total_width)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scaling = (width / 144) * 0.45
        text_sizes = [cv2.getTextSize(text, font, scaling, 1)[0]
                      for text in titles]
        tallest = max(text_sizes, key=lambda size: size[1])[1]
        text_y = int((height + tallest) / 2)
        text_x = [int((width - text_sizes[idx][0]) / 2) + width * idx
                  for idx in range(len(titles))]
        logger.debug("texts: %s, text_sizes: %s, text_x: %s, text_y: %s",
                     titles, text_sizes, text_x, text_y)
        header_box = np.ones((height, total_width, 3), np.float32)
        for idx, text in enumerate(titles):
            cv2.putText(header_box,
                        text,
                        (text_x[idx], text_y),
                        font,
                        scaling,
                        (0, 0, 0),
                        1,
                        lineType=cv2.LINE_AA)
        logger.debug("header_box.shape: %s", header_box.shape)
        return header_box

class Timelapse():
    """ Create a time-lapse preview image.

    Parameters
    ----------
    model: plugin from :mod:`plugins.train.model`
        The selected model that will be running this trainer
    coverage_ratio: float
        Ratio of face to be cropped out of the training image.
    image_count: int
        The number of preview images to be displayed in the time-lapse
    mask_opacity: int
        The opacity (as a percentage) to use for the mask overlay
    mask_color: str
        The hex RGB value to use the mask overlay
    feeder: :class:`~lib.training.generator.Feeder`
        The feeder for generating the time-lapse images.
    image_paths: dict
        The full paths to the training images for each side of the model
    """
    def __init__(self,
                 model: ModelBase,
                 coverage_ratio: float,
                 image_count: int,
                 mask_opacity: int,
                 mask_color: str,
                 feeder: Feeder,
                 image_paths: dict[SideLiteral, list[str]]) -> None:
        logger.debug("Initializing %s: model: %s, coverage_ratio: %s, image_count: %s, "
                     "mask_opacity: %s, mask_color: %s, feeder: %s, image_paths: %s)",
                     self.__class__.__name__, model, coverage_ratio, image_count, mask_opacity,
                     mask_color, feeder, len(image_paths))
        self._num_images = image_count
        self._samples = Samples(model, coverage_ratio, mask_opacity, mask_color)
        self._sides = self._samples.sides
        self._model = model
        self._feeder = feeder
        self._image_paths = image_paths
        self._output_file = ""
        logger.debug("Initialized %s", self.__class__.__name__)

    def _setup(self, output: str | None = None, **inputs: str) -> None:
        """ Setup the time-lapse folder locations and the time-lapse feed.

        Parameters
        ----------
        output: str, optional
            The full path to the time-lapse output folder. If ``None`` is provided this will
            default to the model folder
        """
        logger.debug("Setting up time-lapse")
        if not output:
            output = get_folder(os.path.join(str(self._model.io.model_dir),
                                             f"{self._model.name}_timelapse"))
        self._output_file = output
        logger.debug("Time-lapse output set to '%s'", self._output_file)

        # Rewrite paths to pull from the training images so mask and face data can be accessed
        images: dict[SideLiteral, list[str]] = {}
        for side in self._sides:
            key = f"input_{side}"
            input_path = inputs.get(key)
            if not input_path or side not in self._image_paths:
                continue
            training_path = os.path.dirname(self._image_paths[T.cast(SideLiteral, side)][0])
            images[T.cast(SideLiteral, side)] = [
                os.path.join(training_path, os.path.basename(pth))
                for pth in get_image_paths(input_path)]

        if not images:
            raise FaceswapError("No timelapse inputs were provided for the configured sides")

        batchsize = min(*(len(img_list) for img_list in images.values()), self._num_images)
        self._feeder.set_timelapse_feed(images, batchsize)
        logger.debug("Set up time-lapse")

    def output_timelapse(self, timelapse_kwargs: dict[str, str]) -> None:
        """ Generate the time-lapse samples and output the created time-lapse to the specified
        output folder.

        Parameters
        ----------
        timelapse_kwargs: dict:
            The keyword arguments for setting up the time-lapse. All values should be full paths
            keyed by ``input_<side>`` for each configured side alongside ``output``
        """
        logger.debug("Ouputting time-lapse")
        if not self._output_file:
            self._setup(**timelapse_kwargs)

        logger.debug("Getting time-lapse samples")
        self._samples.images = self._feeder.generate_preview(is_timelapse=True)
        logger.debug("Got time-lapse samples: %s",
                     {side: len(images) for side, images in self._samples.images.items()})

        image = self._samples.show_sample()
        if image is None:
            return
        filename = os.path.join(self._output_file, str(int(time.time())) + ".jpg")

        cv2.imwrite(filename, image)
        logger.debug("Created time-lapse: '%s'", filename)


def _stack_images(images: np.ndarray) -> np.ndarray:
    """ Stack images evenly for preview.

    Parameters
    ----------
    images: :class:`numpy.ndarray`
        The preview images to be stacked

    Returns
    -------
    :class:`numpy.ndarray`
        The stacked preview images
    """
    logger.debug("Stack images")

    def get_transpose_axes(num):
        if num % 2 == 0:
            logger.debug("Even number of images to stack")
            y_axes = list(range(1, num - 1, 2))
            x_axes = list(range(0, num - 1, 2))
        else:
            logger.debug("Odd number of images to stack")
            y_axes = list(range(0, num - 1, 2))
            x_axes = list(range(1, num - 1, 2))
        return y_axes, x_axes, [num - 1]

    images_shape = np.array(images.shape)
    new_axes = get_transpose_axes(len(images_shape))
    new_shape = [np.prod(images_shape[x]) for x in new_axes]
    logger.debug("Stacked images")
    return np.transpose(images, axes=np.concatenate(new_axes)).reshape(new_shape)
