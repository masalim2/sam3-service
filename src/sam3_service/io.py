import io
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Iterator

import numpy as np
import tifffile
from torch.utils.data import IterableDataset
from webdataset.autodecode import handle_extension
from webdataset.compat import WebDataset

from .schema import BboxPrompt, SAM3Result, SamplePrompt

logger = logging.getLogger(__name__)


def tiff_decoder(data):
    arr = tifffile.imread(io.BytesIO(data))
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    return arr


def _expand_prompts(samples: Iterable[dict[str, Any]]) -> Iterator[SamplePrompt]:
    """
    Takes an iterator of decoded WebDataset samples and yields one SamplePrompt
    per prompt entry in the JSON. This structure enables submitting multiple
    prompts for the same image.
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


def build_dataset(tar_path: Path | str) -> IterableDataset[SamplePrompt]:
    """
    Load SamplePrompts from a .tar archive.
    """
    return (
        WebDataset(Path(tar_path).as_posix(), shardshuffle=False, empty_check=False)
        .decode(
            handle_extension(".tif", tiff_decoder),
            handle_extension(".tiff", tiff_decoder),
            "pil",  # Fallback
        )
        .compose(_expand_prompts)
    )


def save_batch(results: list[SAM3Result], output_dir: Path) -> None:
    checked_dirs = set()

    t0 = perf_counter()
    for result in results:
        out = {
            "key": result.key,
            "num_objects": len(result.scores),
            "scores": result.scores,
            "boxes": result.boxes,
            "masks_file": f"{result.key}_masks.npy",
        }

        out_path = output_dir / f"{result.key}.result.json"
        if out_path.parent not in checked_dirs:
            out_path.parent.mkdir(exist_ok=True, parents=True)
            checked_dirs.add(out_path.parent)

        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        np.save(output_dir / f"{result.key}_masks.npy", result.masks)
    elapsed = perf_counter() - t0
    logger.info(f"Saved {len(results)} results to {output_dir} in {elapsed:.1f} sec")
