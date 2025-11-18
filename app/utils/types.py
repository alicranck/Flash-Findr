from enum import Enum
from typing import Union, Sequence, Optional, get_args, get_origin, Iterable, Dict, Tuple, List, Any
from typing_extensions import Annotated

import io

from pathlib import Path
from numbers import Number
import xml.etree.ElementTree as ET

import numpy as np
import PIL
from PIL import Image

from pydantic import AfterValidator


# Type validators
def check_is_pil_image(image: PIL.Image.Image) -> bool:
    assert isinstance(image, PIL.Image.Image)
    return image

def check_is_pil_mask(image: PIL.Image.Image) -> bool:
    is_pil_image = check_is_pil_image(image)
    is_gray_scale = (image.mode == "L")
    assert is_pil_image and is_gray_scale
    return image

def is_non_empty_pil_mask(image: PIL.Image.Image) -> bool:
    is_pil_mask = check_is_pil_mask(image)
    is_non_empty = np.any(np.array(image))
    assert is_pil_mask and is_non_empty
    return image

def check_is_numpy_mask(mask: np.ndarray) -> bool:
    assert isinstance(mask, np.ndarray)
    assert mask.ndim in [2, 3]
    if mask.ndim == 3:
        assert mask.shape[2] == 1
    return mask


# Type aliases
Pathlike = Union[str, Path]
ImageHandle = Union[str, Path, np.ndarray, PIL.Image.Image, io.BytesIO, io.IOBase]   # Can be an image path, the image itself or a stream of an image received by an end user
AnnotationHandle = Union[str, Path, np.ndarray, PIL.Image.Image, io.BytesIO, io.IOBase, ET.Element]

BBox = Sequence[Number]

PILImage = Annotated[PIL.Image.Image, AfterValidator(check_is_pil_image)]
PILMask = Annotated[PIL.Image.Image, AfterValidator(check_is_pil_mask)]
NonEmptyPILMask = Annotated[PIL.Image.Image, AfterValidator(is_non_empty_pil_mask)]
NumpyMask = Annotated[np.ndarray, AfterValidator(check_is_numpy_mask)]

# Image constants
IMG_EXTENSIONS = [ext.lower() for ext in ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'tiff', "webp"]]
IMG_MODES = ['1', 'L', 'P', 'RGB', 'RGBA', 'CMYK', 'YCbCr', 'LAB', 'HSV', 'I', 'F']