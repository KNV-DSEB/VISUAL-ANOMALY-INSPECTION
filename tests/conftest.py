import copy

import pytest

from src.data import load_protocol
from tests.test_oneclass import _toy_mvtec


@pytest.fixture
def toy(tmp_path):
    """Tiny MVTec-like dataset (one category, 40x40 images) and a protocol pointing at it."""
    _toy_mvtec(tmp_path / "mvtec")
    protocol = copy.deepcopy(load_protocol())
    protocol["dataset"]["root"] = str(tmp_path / "mvtec")
    protocol["dataset"]["categories"] = ["toy"]
    protocol["preprocessing"].update(resize=36, crop=32)
    return protocol, tmp_path
