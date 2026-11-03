import base64
import gzip
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    computed_field,
    model_validator,
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

    return base64.b64encode(buf.getvalue()).decode()


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


class StagedFile(BaseModel):
    """
    A file in a user's staging area, as `anchor` (the trusted, canonical
    <staging root>/<user id> directory) plus the path components below it.

    Workers open it by walking `parts` from `anchor` with O_NOFOLLOW (see
    safe_fs), so a symlink swapped in after validation is never followed.
    """

    anchor: Path
    parts: tuple[str, ...]

    @property
    def path(self) -> Path:
        return self.anchor.joinpath(*self.parts)


class BatchRequest(BaseModel):
    """
    Request to run inference on all Samples defined in a WebDataset-compliant
    tar archive.
    """

    dataset: StagedFile


class ImageRequest(BaseModel):
    """
    Request to run inference on a single staged image.
    """

    image: StagedFile
    prompt: Prompt


class SegmentationTask(BaseModel):
    """
    Body of the gateway's `promptable-segmentation` task.
    """

    inference_type: Literal["single-image", "batch"]
    data_uri: str
    single_image_prompt: str | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_weights_override(cls, data: Any) -> Any:
        """
        Checkpoints are configured only at server launch (SAM3_DIR). Reject the
        retired `weights_dir_override` field loudly rather than silently
        ignoring it and serving results from different weights.
        """
        if isinstance(data, dict) and data.get("weights_dir_override") is not None:
            raise ValueError(
                "weights_dir_override is not supported: model weights are "
                "configured at server launch"
            )
        return data


class BatchResponse(BaseModel):
    """
    Response for completed WebDataset batch inference task
    """

    result_path: Path
    runtime_info: dict[str, Any] | None = None


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
