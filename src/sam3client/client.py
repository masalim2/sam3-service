import base64
import gzip
import io

import matplotlib.pyplot as plt
import numpy as np
import requests
import smart_open
import typer
from PIL import Image
from rich import print
from sam3.visualization_utils import COLORS, plot_bbox, plot_mask


def load_b64_npy(s: str):
    return np.load(io.BytesIO(gzip.decompress(base64.b64decode(s))))


def plot_results(img, results, path):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    nb_objects = len(results["scores"])

    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        plot_mask(results["masks"][i].squeeze(0).cpu(), color=color)
        w, h = img.size
        prob = results["scores"][i].item()
        plot_bbox(
            h,
            w,
            results["boxes"][i].cpu(),
            text=f"(id={i}, {prob=:.2f})",
            box_format="XYXY",
            color=color,
            relative_coords=False,
        )
    plt.savefig(path)
    print(f"Saved result preview to {path}.")


cli = typer.Typer()


@cli.command()
def infer(
    image_uri: str,
    text_prompt: str,
    base_url: str = "http://sophia-gpu-10:8000",
    result_path: str = "result-preview.png",
) -> None:
    resp = requests.post(
        base_url,
        json={
            "image_uri": image_uri,
            "text_prompt": text_prompt,
            "box_prompts": [],
        },
    )
    resp.raise_for_status()
    results = resp.json()
    scores = results["scores"]
    boxes = results["boxes"]
    results["masks"] = load_b64_npy(results["masks"])

    print(f"Found {len(scores)} objects")
    print(f"Scores: {scores}")
    print("Bounding boxes:", *boxes, sep="\n")

    with smart_open.open(image_uri, "rb") as fp:
        image = Image.open(fp)
        plot_results(image, results, result_path)


if __name__ == "__main__":
    cli()
