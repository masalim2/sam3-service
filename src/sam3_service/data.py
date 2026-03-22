import json
import logging
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import IO, Any, Callable

import numpy as np
import numpy.typing as npt
import tifffile
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import BatchEncoding, Sam3Processor

from .config import CHECKPOINT_DIR
from .schema import Prompt, Sample

NDArray = npt.NDArray[Any]
logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

processor = Sam3Processor.from_pretrained(CHECKPOINT_DIR, local_files_only=True)

def _pil_load(path: Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img

class WebDataset(Dataset[Sample]):
    _decoders: dict[str, Callable[[IO[bytes]], Image.Image | NDArray]] = {
        ".jpg": _pil_load,
        ".jpeg": _pil_load,
        ".png": _pil_load,
        ".webp": _pil_load,
        ".tif": tifffile.imread,
        ".tiff": tifffile.imread,
        ".npy": np.load,
    }

    def __init__(self, tar_path: str | Path) -> None:
        self.tar_path = tar_path
        with tarfile.open(tar_path) as tf:
            filenames = [f.name for f in tf if f.isfile()]

        self.suffix_map: dict[str, list[str]] = defaultdict(list)
        self.path_map = {}
        for name in filenames:
            key = str(Path(name).with_suffix(""))
            suffix = Path(name).suffix
            self.suffix_map[key].append(suffix)
            self.path_map[(key, suffix)] = name

        self.suffix_map = {
            key: suffixes
            for key, suffixes in self.suffix_map.items()
            if ".json" in suffixes and any(ext in suffixes for ext in self._decoders)
        }
        self.keys = sorted(self.suffix_map.keys())

    def __getitem__(self, index: int) -> Sample:
        key = self.keys[index]
        with tarfile.open(self.tar_path) as tf:
            json_path = self.path_map[key, ".json"]
            assert (json_fp := tf.extractfile(json_path)) is not None
            prompts = json.load(json_fp).get("prompts", [])

            image_ext = next(
                ext for ext in self.suffix_map[key] if ext in self._decoders
            )
            image_path = self.path_map[key, image_ext]
            assert (image_fp := tf.extractfile(image_path)) is not None
            image = self._decoders[image_ext](image_fp)

        return Sample(name=key, image=image, prompts=[Prompt(**p) for p in prompts])

    def __len__(self):
        return len(self.keys)


def quantile_norm(image: NDArray | Image.Image) -> NDArray:
    if isinstance(image, Image.Image):
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = np.array(image)

    # Vision models expect a color channel:
    if image.ndim == 2:
        image = image[np.newaxis, :, :]

    assert image.ndim == 3

    # If floating dtype, normalize and clamp pixel intensities such that
    # the 1st percentile is 0 and 99th percentile is 255

    # N.B. this is required to get any results out of float32 TIFFs but a major
    # bottleneck (reduces throughput from 4.7 to 0.8 img/sec on tomo dataset)
    # Suggest running this on the client to save on transfer time too:
    if not np.issubdtype(image.dtype, np.integer):
        p_lo, p_hi = np.percentile(image, [1, 99])
        image = np.clip((image - p_lo) / (p_hi - p_lo), 0, 1)
        image = (image * 255).astype(np.uint8)

    return image


def preprocess_image(image: Image.Image, text_prompt: str) -> BatchEncoding:
    """
    Preprocess single image + text prompt
    """
    return processor(images=quantile_norm(image), text=text_prompt, return_tensors="pt")


def preprocess_batch(samples: list[Sample]) -> tuple[list[Sample], BatchEncoding]:
    """
    Preprocess Samples from webdataset.
    """
    import os

    logger.info(f"Preprocessing {[s.name for s in samples]} on {os.getpid()=}")

    images = []
    text_prompts = []
    box_prompts = []
    box_labels = []

    for sample in samples:
        sample.image = quantile_norm(sample.image)
        for prompt in sample.prompts:
            images.append(sample.image)
            text_prompts.append(prompt.text)
            box_prompts.append(
                [list(box[:4]) for box in prompt.boxes] if prompt.boxes else None
            )
            box_labels.append(
                [int(box.label) for box in prompt.boxes] if prompt.boxes else None
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
    return samples, inputs


def build_dataloader(
    tar_path: Path | str, batch_size: int = 4, num_workers: int = 4
) -> DataLoader:
    dataset = WebDataset(tar_path)
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
