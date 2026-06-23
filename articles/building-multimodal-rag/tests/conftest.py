import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rgb_image():
    return Image.new("RGB", (32, 32), color=(100, 150, 200))


@pytest.fixture
def flat_embedding():
    dim = 512
    return [float(i) / dim for i in range(dim)]
