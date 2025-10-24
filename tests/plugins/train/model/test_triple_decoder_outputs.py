"""Smoke tests for triple decoder support in Original and DFaker models."""
import pytest

_ = pytest.importorskip("keras")
from keras import backend as K, initializers

from plugins.train.model.original import Model as OriginalModel
from plugins.train.model.dfaker import Model as DFakerModel


@pytest.fixture(autouse=True)
def cleanup_keras_session():
    """Ensure we don't leak graph state between tests."""
    yield
    K.clear_session()


def _collect_output_names(model: OriginalModel) -> list[str]:
    inputs = model._get_inputs()
    autoencoder = model.build_model(inputs)
    return [tensor.name for tensor in autoencoder.outputs]


def _expected_sequence(sides: list[str], learn_mask: bool) -> list[str]:
    sequence: list[str] = []
    for side in sides:
        sequence.append(f"face_out_{side}")
        if learn_mask:
            sequence.append(f"mask_out_{side}")
    return sequence


def _init_original(third_side: bool, learn_mask: bool) -> OriginalModel:
    model = OriginalModel.__new__(OriginalModel)
    model.input_shape = (64, 64, 3)
    model.low_mem = False
    model.learn_mask = learn_mask
    model.encoder_dim = 1024
    model.sides = ["a", "b"] + (["c"] if third_side else [])
    return model


def _init_dfaker(third_side: bool, learn_mask: bool, output_size: int = 128) -> DFakerModel:
    model = DFakerModel.__new__(DFakerModel)
    model.input_shape = (output_size // 2, output_size // 2, 3)
    model.low_mem = False
    model.learn_mask = learn_mask
    model.encoder_dim = 1024
    model._output_size = output_size
    model.kernel_initializer = initializers.RandomNormal(0, 0.02)
    model.sides = ["a", "b"] + (["c"] if third_side else [])
    return model


@pytest.mark.parametrize("learn_mask", [False, True], ids=["faces_only", "faces_and_masks"])
def test_original_outputs_are_ordered(learn_mask: bool) -> None:
    model = _init_original(third_side=True, learn_mask=learn_mask)
    outputs = _collect_output_names(model)
    expected = _expected_sequence(model.sides, learn_mask)
    assert len(outputs) == len(expected)
    for name, fragment in zip(outputs, expected):
        assert fragment in name


@pytest.mark.parametrize("learn_mask", [False, True], ids=["faces_only", "faces_and_masks"])
def test_dfaker_outputs_are_ordered(learn_mask: bool) -> None:
    model = _init_dfaker(third_side=True, learn_mask=learn_mask)
    outputs = _collect_output_names(model)
    expected = _expected_sequence(model.sides, learn_mask)
    assert len(outputs) == len(expected)
    for name, fragment in zip(outputs, expected):
        assert fragment in name


def test_original_default_two_sided() -> None:
    model = _init_original(third_side=False, learn_mask=False)
    outputs = _collect_output_names(model)
    expected = _expected_sequence(model.sides, False)
    assert len(outputs) == len(expected) == 2
    for name, fragment in zip(outputs, expected):
        assert fragment in name
