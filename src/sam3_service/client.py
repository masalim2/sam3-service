import base64
import gzip
import io
import json
import logging
import tarfile
from concurrent.futures import ProcessPoolExecutor, wait
from io import BytesIO
from math import ceil
from pathlib import Path

import numpy as np
import requests
import tifffile
import typer
from PIL import Image
from rich.logging import RichHandler

from .tar_helpers import add_member

cli = typer.Typer()
logging.basicConfig(level="INFO", handlers=[RichHandler()])
logger = logging.getLogger("sam3_service.client")


def load_b64_npy(s: str):
    return np.load(io.BytesIO(gzip.decompress(base64.b64decode(s))))


def convert_image(img_path: Path) -> BytesIO:
    """
    Converts images to uint8-quantized npy format with a channel dimension.
    SAM3 preprocessing expects 1 or 3 channel dims and will squash intensities
    in a smaller dynamic range.  So, we might as well cut the precision down
    locally before we send data over.
    """
    if img_path.suffix in (".tif", ".tiff"):
        image = tifffile.imread(img_path)
    else:
        image = Image.open(img_path)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = np.array(image)

    if image.ndim == 2:
        image = image[np.newaxis, :, :]

    if not np.issubdtype(image.dtype, np.integer):
        lo, hi = np.percentile(image, (1, 99))
        image = np.clip((image - lo) / (hi - lo), 0, 1)
        image = (image * 255).astype(np.uint8)

    buf = BytesIO()
    np.save(buf, image)
    buf.seek(0)
    return buf


def write_wds_shard(
    tar_path: Path, image_paths: list[Path], text_prompts: list[str]
) -> None:
    """
    Create a tar (aka WebDataset) of paired .npy/.json prompts to SAM3.
    """
    prompt_json = BytesIO(
        json.dumps({"prompts": [{"text": tp} for tp in text_prompts]}).encode()
    )

    logger.info(f"Writing {len(image_paths)} images to shard: {tar_path}")
    with tarfile.open(tar_path, mode="w") as writer:
        for path in sorted(image_paths):
            add_member(writer, path.with_suffix(".npy").name, convert_image(path))
            add_member(writer, path.with_suffix(".json").name, prompt_json)


def plot_results(img, results, path):
    # from sam3.visualization_utils import COLORS, plot_bbox, plot_mask
    # plt.figure(figsize=(12, 8))
    # plt.imshow(img)

    # nb_objects = len(results["scores"])

    # for i in range(nb_objects):
    #     color = COLORS[i % len(COLORS)]
    #     plot_mask(results["masks"][i].squeeze(0).cpu(), color=color)
    #     w, h = img.size
    #     prob = results["scores"][i].item()
    #     plot_bbox(
    #         h,
    #         w,
    #         results["boxes"][i].cpu(),
    #         text=f"(id={i}, {prob=:.2f})",
    #         box_format="XYXY",
    #         color=color,
    #         relative_coords=False,
    #     )
    # plt.savefig(path)
    # logger.info(f"Saved result preview to {path}.")
    pass


@cli.command()
def submit_batch(
    dataset_path: str,
    base_url: str = "http://sophia-gpu-10:8000",
) -> None:
    logger.info("Sending request...")
    resp = requests.post(
        f"{base_url}/process-wds",
        json={"dataset_path": dataset_path},
    )
    resp.raise_for_status()
    logger.info(resp.json())


@cli.command()
def submit_image(
    image_uri: str,
    prompt: str,
    base_url: str = "http://sophia-gpu-10:8000",
) -> None:
    logger.info("Sending request...")
    resp = requests.post(
        f"{base_url}/process-image",
        json={"image_uri": image_uri, "text_prompt": prompt},
    )
    resp.raise_for_status()
    logger.info(resp.json())


@cli.command()
def create_webdataset(
    image_dir: Path,
    image_ext: str,
    text_prompts: list[str],
    *,
    output_dir: Path | None = None,
    shard_size: int = 100,
    num_workers: int = 4,
) -> None:
    """
    Package images in IMAGE_DIR with suffix IMAGE_EXT into WebDataset format with one or
    more SAM3 TEXT_PROMPTS. Shards of SHARD_SIZE are written to OUTPUT_DIR using
    NUM_WORKERS parallel workers.  For example:

    python prepare_webdataset.py /example/tomo_00104/sand1/ tiff 'grain' 'granule'
    """
    if output_dir is None:
        output_dir = image_dir.with_name(f"{image_dir.name}-webdataset-shards")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_ext = image_ext.strip(".")
    source_paths = list(Path(image_dir).glob(f"*.{image_ext}"))
    if not source_paths:
        raise RuntimeError(f"No '.{image_ext}' files found in {image_dir}")

    num_shards = ceil(len(source_paths) / shard_size)
    shards = [
        source_paths[i * shard_size : (i + 1) * shard_size] for i in range(num_shards)
    ]

    logger.info(f"Writing {num_shards=} using {num_workers=}")
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futs = []
        for i, shard in enumerate(shards):
            tar_path = output_dir / f"shard-{i:05d}.tar"
            futs.append(pool.submit(write_wds_shard, tar_path, shard, text_prompts))

        logger.info("Waiting for shard writes to finish...")
        wait(futs)
        logger.info("Done!")


if __name__ == "__main__":
    cli()
