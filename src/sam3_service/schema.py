from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy.typing as npt
from PIL import Image
from pydantic import BaseModel

NDArray = npt.NDArray[Any]


class BboxPrompt(NamedTuple):
    """
    SAM3 bounding box with positive/negative label.

    The Prompt may contain zero or more bounding boxes to refine the selection.
    """

    x: float
    y: float
    width: float
    height: float
    label: bool


class Prompt(BaseModel):
    """
    Single SAM3 Prompt: may combine text and 0 or more bounding boxes.
    """

    text: str | None = None
    boxes: list[BboxPrompt] | None = None


@dataclass
class Sample:
    """
    Sample for inference: a named image with one or more Prompts.

    Each Prompt in turn may be a compound text+bounding boxes prompt.
    """

    name: str
    image: Image.Image | NDArray
    prompts: list[Prompt]


@dataclass
class SAM3Result:
    """
    SAM3 output for a single Prompt. A multi-prompt Sample will generate
    several SAM3Results.
    """

    key: str
    scores: list[float]
    boxes: list[list[float]]
    masks: NDArray


class BatchRequest(BaseModel):
    """
    Request to run inference on all Samples defined in a WebDataset-compliant
    tar archive.
    """

    dataset_path: Path


class ImageRequest(BaseModel):
    """
    Request to run inference on a single image URI.
    """

    image_uri: str
    text_prompt: str


class BatchResponse(BaseModel):
    """
    Response for completed WebDataset batch inference task
    """

    result_dir: Path


class ImageResponse(BaseModel):
    """
    Response for completed single-image inference task
    """

    num_objects: int
    boxes: list[list[float]]
    scores: list[float]


class ErrorResponse(BaseModel):
    error: str
