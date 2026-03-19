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
import typer
from rich.logging import RichHandler

cli = typer.Typer()
logging.basicConfig(level="INFO", handlers=[RichHandler()])
logger = logging.getLogger("sam3_service.client")


def load_b64_npy(s: str):
    return np.load(io.BytesIO(gzip.decompress(base64.b64decode(s))))


def write_wds_shard(
    tar_path: Path, image_paths: list[Path], text_prompts: list[str]
) -> None:
    prompt_json = BytesIO(
        json.dumps({"prompts": [{"text": tp} for tp in text_prompts]}).encode()
    )
    prompt_size = len(prompt_json.getvalue())

    logger.info(f"Writing {len(image_paths)} images to shard: {tar_path}")
    with tarfile.open(tar_path, mode="w:gz", compresslevel=4) as writer:
        for path in sorted(image_paths):
            writer.add(str(path), arcname=path.name)

            info = tarfile.TarInfo(path.with_suffix(".json").name)
            info.size = prompt_size
            prompt_json.seek(0)
            writer.addfile(info, prompt_json)

    logger.info(f"Wrote shard: {tar_path}")


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
def submit_webdataset(
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
    shard_dir: Path | None = None,
    shard_size: int = 100,
    num_workers: int = 4,
) -> None:
    """
    Package images in IMAGE_DIR with suffix IMAGE_EXT into WebDataset format with one or
    more SAM3 TEXT_PROMPTS. Shards of SHARD_SIZE are written to SHARD_DIR using
    NUM_WORKERS parallel workers.  For example:

    python prepare_webdataset.py /example/tomo_00104/sand1/ tiff 'grain' 'granule'
    """
    if shard_dir is None:
        shard_dir = image_dir.with_name(f"{image_dir.name}-webdataset-shards")

    shard_dir = Path(shard_dir).resolve()
    shard_dir.mkdir(parents=True, exist_ok=True)

    image_ext = image_ext.strip(".")
    source_paths = [f for f in Path(image_dir).glob(f"*.{image_ext}")]
    if not source_paths:
        raise RuntimeError(
            f"No files found in {image_dir} with the .{image_ext} extension"
        )

    num_shards = ceil(len(source_paths) / shard_size)
    shards = [
        source_paths[i * shard_size : (i + 1) * shard_size] for i in range(num_shards)
    ]

    logger.info(f"Writing {num_shards=} using {num_workers=}")
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futs = []
        for i, shard in enumerate(shards):
            tar_path = shard_dir / f"shard-{i:05d}.tar.gz"
            futs.append(pool.submit(write_wds_shard, tar_path, shard, text_prompts))

        logger.info("Waiting for shard writes to finish...")
        wait(futs)
        logger.info("Done!")


if __name__ == "__main__":
    cli()
