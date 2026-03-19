from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy.typing as npt
from PIL.Image import Image
from pydantic import BaseModel

NDArray = npt.NDArray[Any]


class BboxPrompt(NamedTuple):
    x: float
    y: float
    width: float
    height: float
    label: bool


@dataclass
class SamplePrompt:
    """Parsed prompt payload for a single image."""

    key: str
    image: Image | NDArray
    text: str | None = None
    boxes: list[BboxPrompt] | None = None


@dataclass
class SAM3Result:
    """
    SAM3 postprocessed return type
    """

    key: str
    scores: list[float]
    boxes: list[list[float]]
    masks: NDArray


class ProcessWebDatasetRequest(BaseModel):
    dataset_path: Path


class ProcessImageRequest(BaseModel):
    image_uri: str
    text_prompt: str


class ProcessWebDatasetResponse(BaseModel):
    result_dir: Path | None = None


class ErrorResponse(BaseModel):
    error: str


class ProcessImageResponse(BaseModel):
    num_objects: int
    boxes: list[list[float]]
    scores: list[float]
