import base64
import concurrent.futures
import gzip
import io
import logging
import multiprocessing as mp
import os
from contextlib import asynccontextmanager
from multiprocessing.sharedctypes import SynchronizedBase
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image
import smart_open
from fastapi import FastAPI
from pydantic import BaseModel
from rich.logging import RichHandler

CHECKPOINT_PATH = Path(
    "/eagle/datascience/msalim/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
)
NDArrayAny = npt.NDArray[Any]

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("sam3_inference_server")


class BoxPrompt(BaseModel):
    label: bool
    x: float
    y: float
    width: float
    height: float


class SegmentImageRequest(BaseModel):
    image_uri: str
    text_prompt: str | None
    box_prompts: list[BoxPrompt] | None


class SegmentImageResult(BaseModel):
    scores: list[float]
    boxes: list[list[float]]
    masks: str


def ndarray_to_base64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
        np.save(f, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Sam3Wrapper:
    def __init__(self, checkpoint_path: Path, gpu_id: int) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"Loading SAM3 on {gpu_id=} PID={os.getpid()}")

        import torch

        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # use bfloat16
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        torch.inference_mode().__enter__()

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path, load_from_HF=False, compile=False
        )
        self.processor = Sam3Processor(model, confidence_threshold=0.5)
        logger.info(f"Loaded SAM3 on {gpu_id=} PID={os.getpid()}")

    def predict(
        self, image: Image, text_prompt: str | None, boxes: list[BoxPrompt] | None
    ) -> SegmentImageResult:
        import torch

        from sam3.model.box_ops import box_xywh_to_cxcywh
        from sam3.visualization_utils import normalize_bbox

        state = self.processor.set_image(image)

        if text_prompt:
            state = self.processor.set_text_prompt(state=state, prompt=text_prompt)

        if boxes:
            box_input_xywh = [[bb.x, bb.y, bb.width, bb.height] for bb in boxes]
            box_input_cxcywh = box_xywh_to_cxcywh(
                torch.tensor(box_input_xywh).view(-1, 4)
            )
            norm_boxes_cxcywh = normalize_bbox(box_input_cxcywh, *image.size)
            labels = [bb.label for bb in boxes]
            for box, label in zip(norm_boxes_cxcywh, labels):
                state = self.processor.add_geometric_prompt(
                    state=state, box=box, label=label
                )

        return SegmentImageResult(
            scores=state["scores"].cpu().float().numpy().tolist(),
            boxes=state["boxes"].cpu().float().numpy().tolist(),
            masks=ndarray_to_base64(state["masks"].cpu().bool().squeeze(1).numpy()),
        )


sam3_wrapper: Sam3Wrapper
worker_pool: concurrent.futures.ProcessPoolExecutor


def load_sam3_model(counter: SynchronizedBase) -> None:
    global sam3_wrapper
    with counter.get_lock():
        counter.value += 1
        gpu_id = counter.value - 1

    sam3_wrapper = Sam3Wrapper(CHECKPOINT_PATH, gpu_id)


@asynccontextmanager
async def worker_lifespan(_app: FastAPI):
    global worker_pool
    import torch

    num_gpu = torch.cuda.device_count()
    if num_gpu == 0:
        raise RuntimeError(f"No GPUs detected")

    counter = mp.Value("i")
    logger.info(f"Initializing SAM3 GPU Worker Pool with {num_gpu=}")
    worker_pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=num_gpu,
        initializer=load_sam3_model,
        initargs=(counter,),
    )
    yield
    worker_pool.shutdown(cancel_futures=True)


app = FastAPI(lifespan=worker_lifespan)


def segment(req: SegmentImageRequest) -> SegmentImageResult:
    with smart_open.open(req.image_uri, "rb") as fp:
        image = Image.open(fp)
        return sam3_wrapper.predict(image, req.text_prompt, req.box_prompts)


@app.post("/", response_model=SegmentImageResult)
def process_image(request: SegmentImageRequest) -> SegmentImageResult:
    return worker_pool.submit(segment, request).result()
