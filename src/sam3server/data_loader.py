from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, NamedTuple

import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader
from transformers import Sam3Processor

from sam3server.config import CHECKPOINT_DIR

sam3_processor = Sam3Processor.from_pretrained(CHECKPOINT_DIR, local_files_only=True)


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
    image: Image.Image
    text: str | None = None
    boxes: list[BboxPrompt] | None = None


def expand_prompts(samples: Iterable[dict[str, Any]]) -> Iterator[SamplePrompt]:
    """
    Takes an iterator of decoded WebDataset samples and yields
    one SamplePrompt per prompt entry in the JSON.
    """
    extensions = ("jpg", "png", "webp", "tif", "tiff")
    for sample in samples:
        key = sample["__key__"]

        image = next((sample[ext] for ext in extensions if ext in sample), None)
        if image is None:
            continue

        prompts = sample.get("json", {}).get("prompts", [])

        for i, prompt in enumerate(prompts):
            yield SamplePrompt(
                key=f"{key}_{i}",
                image=image,
                text=prompt.get("text", "").strip() or None,
                boxes=[BboxPrompt(*b) for b in prompt.get("boxes", [])] or None,
            )


def build_dataset(tar_path: Path | str) -> wds.WebDataset:
    return (
        wds.WebDataset(Path(tar_path).as_posix(), shardshuffle=False, empty_check=False)
        .decode("pil")
        .compose(expand_prompts)
    )


def collate(samples: list[SamplePrompt]):
    text_prompts = [s.text for s in samples]
    box_prompts = [
        [list(box[:4]) for box in s.boxes] if s.boxes else None for s in samples
    ]
    box_labels = [
        [int(box.label) for box in s.boxes] if s.boxes else None for s in samples
    ]

    process_kwargs = {"images": [s.image for s in samples], "return_tensors": "pt"}

    if any(text_prompts):
        process_kwargs["text"] = text_prompts
    if any(box_prompts):
        process_kwargs["input_boxes"] = box_prompts
        process_kwargs["input_boxes_labels"] = box_labels

    inputs = sam3_processor(**process_kwargs)
    return samples, inputs


def build_webdataset_loader(
    path: Path, batch_size: int, num_workers: int
) -> DataLoader:
    dataset = build_dataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate,
        pin_memory=True,
        in_order=False,
        persistent_workers=False,
    )
