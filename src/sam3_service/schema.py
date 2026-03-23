import base64
import gzip
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, NamedTuple

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    computed_field,
)

from .tar_helpers import add_member

NDArray = npt.NDArray[Any]


def to_ndarray(obj: Any) -> NDArray:
    if isinstance(obj, str):
        obj = base64.b64decode(obj)

    if isinstance(obj, bytes):
        if obj[:2] == b"\x1f\x8b":
            return np.load(BytesIO(gzip.decompress(obj)), allow_pickle=False)
        else:
            return np.load(BytesIO(obj), allow_pickle=False)

    if isinstance(obj, np.ndarray):
        return obj

    raise ValueError(f"Expected str, bytes, or ndarray; got {type(obj)}")


def encode_npy(arr: NDArray) -> str:
    buf = BytesIO()
    with gzip.open(buf, "wb", compresslevel=1) as gzfp:
        np.save(gzfp, arr)
        return base64.b64encode(buf.getbuffer()).decode()


CompressedNDArray = Annotated[
    NDArray,
    BeforeValidator(to_ndarray),
    PlainSerializer(encode_npy, return_type=str, when_used="json"),
]


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
    prompt: Prompt


class BatchResponse(BaseModel):
    """
    Response for completed WebDataset batch inference task
    """

    result_path: Path


class SAM3ImageResult(BaseModel):
    """
    Response for completed single-image inference task
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_image: str
    prompt: Prompt
    batch_slug: str
    boxes: list[list[float]]
    scores: list[float]
    labelmap_npy: CompressedNDArray

    @computed_field
    @property
    def num_objects(self) -> int:
        return len(self.scores)

    def dump_to_tarfile(self, tf: tarfile.TarFile) -> None:
        meta = self.model_dump_json(exclude={"labelmap_npy"}, indent=2)
        add_member(tf, f"{self.batch_slug}.json", BytesIO(meta.encode()))
        with BytesIO() as buf:
            np.save(buf, self.labelmap_npy)
            add_member(tf, f"{self.batch_slug}.labels.npy", buf)


class ErrorResponse(BaseModel):
    error: str
