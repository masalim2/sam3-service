import json
import logging
import tarfile
from concurrent.futures import ProcessPoolExecutor, wait
from io import BytesIO
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import requests
import smart_open
import tifffile
import typer
from PIL import Image
from rich.logging import RichHandler

from sam3_service.schema import SAM3ImageResult
from sam3_service.tar_helpers import add_member
from sam3_service.utils import preview_sam3_result, quantile_norm

NDArray = npt.NDArray[Any]

logging.basicConfig(level="INFO", handlers=[RichHandler()])
logger = logging.getLogger("sam3_service.client")

cli = typer.Typer(no_args_is_help=True)


def load_image(img_path: Path) -> NDArray:
    if img_path.suffix in (".tif", ".tiff"):
        image = tifffile.imread(img_path)
    else:
        image = Image.open(img_path)
    return quantile_norm(image)


def to_npy(arr: NDArray) -> BytesIO:
    buf = BytesIO()
    np.save(buf, arr)
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
            add_member(writer, path.with_suffix(".npy").name, to_npy(load_image(path)))
            add_member(writer, path.with_suffix(".json").name, prompt_json)


@cli.command()
def submit_batch(
    dataset_path: Path,
    base_url: str = "http://sophia-gpu-10:8000",
) -> None:
    """
    Submit a WebDataset-structured tar file for batch inference.
    Example:
    sam3-service submit-batch ./test.tar
    """
    logger.info("Sending request...")
    resp = requests.post(
        f"{base_url}/process-batch",
        json={"dataset_path": dataset_path.resolve().as_posix()},
    )
    resp.raise_for_status()
    logger.info(resp.json())


@cli.command()
def submit_image(
    image_uri: str,
    prompt: str,
    base_url: str = "http://sophia-gpu-10:8000",
    save_preview: Path | None = None,
) -> None:
    logger.info("Sending request...")

    if "://" not in image_uri:
        image_uri = "file://" + Path(image_uri).resolve().as_posix()

    resp = requests.post(
        f"{base_url}/process-image",
        json={"image_uri": image_uri, "prompt": {"text": prompt}},
    )
    resp.raise_for_status()

    result = SAM3ImageResult.model_validate(resp.json())
    logger.info(result.model_dump(exclude={"labelmap_npy"}))

    logger.info("Generating local preview of segmentation results...")
    if save_preview and result.num_objects > 0:
        with smart_open.open(image_uri, "rb") as fp:
            image = quantile_norm(Image.open(fp))
            preview_sam3_result(image, result, save_preview)
            logger.info(f"Saved segmentation result preview to {save_preview}")


@cli.command()
def preview_batch_results(input_tar: Path, result_tar: Path) -> None:
    preview_dir = result_tar.with_suffix(".preview")
    preview_dir.mkdir(exist_ok=True)

    with tarfile.open(result_tar) as tf, tarfile.open(input_tar) as img_tf:
        jsons = [f.name for f in tf if f.isfile() and f.name.endswith(".json")]
        for fname in jsons:
            assert (result_fp := tf.extractfile(fname)) is not None

            result = json.load(result_fp)

            if result["num_objects"] < 1:
                logger.info(f"Skipping {fname}: no objects detected.")
                continue

            assert (img_fp := img_tf.extractfile(result["input_image"])) is not None
            image = quantile_norm(np.load(BytesIO(img_fp.read())))

            assert (
                label_fp := tf.extractfile(fname.replace(".json", ".labels.npy"))
            ) is not None
            labelmap = np.load(BytesIO(label_fp.read()))

            result = SAM3ImageResult(**result, labelmap_npy=labelmap)

            png_path = preview_dir / Path(fname).with_suffix(".png").name
            logger.info(f"Generating preview: {png_path.name} ...")

            preview_sam3_result(image, result, png_path)
            logger.info(f"Saved preview: {png_path}")


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
    Bundle images+text prompts into WebDataset tar archives.
    Example:
    sam3-service create-webdataset ./images/ .tiff "granule" "white shape"
    """
    if output_dir is None:
        output_dir = image_dir.with_name(f"{image_dir.name}-webdataset-shards")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_ext = image_ext.strip(".")
    source_paths = list(Path(image_dir).glob(f"*.{image_ext}"))
    if not source_paths:
        raise RuntimeError(f"No '.{image_ext}' files found in {image_dir}")

    assert len(text_prompts) > 0

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
