import io
import json
import logging
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import IO, Any, Callable

import numpy as np
import numpy.typing as npt
import tifffile
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import BatchEncoding, Sam3Processor

from .config import CHECKPOINT_DIR
from . import safe_fs
from .schema import Prompt
from .utils import quantile_norm

NDArray = npt.NDArray[Any]

logger = logging.getLogger(__name__)
processor = Sam3Processor.from_pretrained(
    CHECKPOINT_DIR, local_files_only=True, trust_remote_code=False
)


def _pil_load(path: IO[bytes] | Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img


def _np_safeload(buf: IO[bytes] | Path) -> NDArray:
    """Load a .npy array; object (pickled) arrays are rejected."""
    if isinstance(buf, Path):
        with buf.open("rb") as fp:
            return np.load(BytesIO(fp.read()), allow_pickle=False)
    return np.load(BytesIO(buf.read()), allow_pickle=False)


@dataclass
class Sample:
    """
    Sample for inference: a named image with one or more Prompts.

    Each Prompt in turn may be a compound text+bounding boxes prompt.
    """

    image_filename: str
    image: Image.Image | NDArray
    prompts: list[Prompt]


@dataclass
class InputInfo:
    image_filename: str
    batch_slug: str
    prompt: Prompt


class WebDataset(Dataset[Sample]):
    _decoders: dict[str, Callable[[IO[bytes] | Path], Image.Image | NDArray]] = {
        "jpg": _pil_load,
        "jpeg": _pil_load,
        "png": _pil_load,
        "webp": _pil_load,
        "tif": tifffile.imread,
        "tiff": tifffile.imread,
        "npy": _np_safeload,
    }

    def __init__(self, tar_fd: int, name: str) -> None:
        """
        `tar_fd` is an already safely-opened fd (see safe_fs.open_file); it is
        inherited by forked DataLoader workers and must outlive the loader.
        """
        self.tar_fd = tar_fd
        self.name = name
        with self._open_tar() as tf:
            filenames = [f.name for f in tf if f.isfile()]

        self.suffix_map: dict[str, list[str]] = defaultdict(list)

        for name in filenames:
            base, *suffix = name.rsplit(".", 1)
            if suffix and (ext := suffix[0]):
                self.suffix_map[base].append(ext)

        self.suffix_map = {
            key: suffixes
            for key, suffixes in self.suffix_map.items()
            if "json" in suffixes and any(ext in suffixes for ext in self._decoders)
        }

        # Have basename keys like 'foo/001' corresponding to pairs such as
        # foo/001.json and foo/001.tiff
        self.keys = sorted(self.suffix_map.keys())
        if self.keys:
            logger.info(f"Initialized WebDataset with {len(self.keys)} keys")
        else:
            logger.warning(f"No samples in WebDataset {self.name}")

    def _open_tar(self) -> tarfile.TarFile:
        # pread-based reader: independent offset per open, safe across forks.
        fileobj = io.BufferedReader(safe_fs.PreadReader(self.tar_fd))
        return tarfile.open(fileobj=fileobj, mode="r")

    def __getitem__(self, index: int) -> Sample:
        key = self.keys[index]

        with self._open_tar() as tf:
            assert (json_fp := tf.extractfile(f"{key}.json")) is not None
            meta = json.load(json_fp)
            if not isinstance(meta, dict):
                raise ValueError(f"{key}.json: expected a JSON object")
            prompts = meta.get("prompts", [])
            if not isinstance(prompts, list):
                raise ValueError(f"{key}.json: 'prompts' must be a list")

            ext = next(e for e in self.suffix_map[key] if e in self._decoders)
            image_name = f"{key}.{ext}"
            assert (image_fp := tf.extractfile(image_name)) is not None
            image = self._decoders[ext](image_fp)

        return Sample(
            image_filename=image_name,
            image=image,
            prompts=[Prompt.model_validate(p) for p in prompts],
        )

    def __len__(self):
        return len(self.keys)


def preprocess_image(image: Image.Image, prompt: Prompt) -> BatchEncoding:
    """
    Preprocess single image + text prompt with optional boxes
    """
    box_kwargs = {}
    if prompt.boxes:
        box_kwargs["input_boxes"] = [[list(box[:4]) for box in prompt.boxes]]
        box_kwargs["input_boxes_labels"] = [[int(box.label) for box in prompt.boxes]]

    return processor(
        images=quantile_norm(image),
        text=prompt.text,
        return_tensors="pt",
        **box_kwargs,
    )


def preprocess_batch(samples: list[Sample]) -> tuple[list[InputInfo], BatchEncoding]:
    """
    Preprocess Samples from webdataset.
    """
    images = []
    text_prompts = []
    box_prompts = []
    box_labels = []
    info_list = []

    for sample in samples:
        sample.image = quantile_norm(sample.image)
        for i, prompt in enumerate(sample.prompts):
            images.append(sample.image)
            text_prompts.append(prompt.text)
            box_prompts.append(
                [list(box[:4]) for box in prompt.boxes] if prompt.boxes else None
            )
            box_labels.append(
                [int(box.label) for box in prompt.boxes] if prompt.boxes else None
            )
            basename = sample.image_filename.rsplit(".")[0]
            info_list.append(
                InputInfo(
                    image_filename=sample.image_filename,
                    batch_slug=f"{basename}_{i}",
                    prompt=prompt,
                )
            )

    process_kwargs: dict[str, Any] = {
        "images": images,
        "return_tensors": "pt",
    }

    if any(text_prompts):
        process_kwargs["text"] = text_prompts
    if any(box_prompts):
        process_kwargs["input_boxes"] = box_prompts
        process_kwargs["input_boxes_labels"] = box_labels

    inputs = processor(**process_kwargs)
    return info_list, inputs


def build_dataloader(
    tar_fd: int, name: str, batch_size: int, num_workers: int
) -> DataLoader:
    dataset = WebDataset(tar_fd, name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=preprocess_batch,
        pin_memory=True,
        in_order=False,
        persistent_workers=False,
        multiprocessing_context="fork",
    )
