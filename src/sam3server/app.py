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
import smart_open
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel, field_validator
from rich.logging import RichHandler

CHECKPOINT_PATH = Path(
    os.environ.get(
        "SAM3_CHECKPOINT",
        "/eagle/datascience/msalim/huggingface/hub/models--facebook--sam3/sam3.pt",
    )
)
NDArrayAny = npt.NDArray[Any]
BoxPrompt = tuple[bool,float,float,float,float]

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

logger = logging.getLogger("sam3_inference_server")
sam3_wrapper: "Sam3Wrapper"
worker_pool: concurrent.futures.ProcessPoolExecutor


class SegmentImageRequest(BaseModel):
    """
    Text and bounding box prompts to segment image at `image_uri`.  
    
    Box prompts are of the form (label,x,y,width,height) where x,y is the upper
    left corner, and the boolean label indicates whether the annotated object is a
    positive or negative example of the target class.
    """
    image_uri: str
    text_prompt: str | None
    box_prompts: list[BoxPrompt]

    @field_validator("image_uri")
    @classmethod
    def validate_uri(cls, v: str) -> str:
        if not v.startswith(("s3://", "http://", "https://", "file://", "/")):
            raise ValueError("image_uri must be an S3, HTTP(S), file URI, or absolute path")
        return v


class SegmentImageResult(BaseModel):
    """
    Segmentation result: confidence scores, bounding boxes, and object masks.
    """
    scores: list[float]
    boxes: list[list[float]]
    masks_npy: str


def ndarray_to_base64(arr: NDArrayAny) -> str:
    """
    Gzip and base64 encode a numpy array
    """
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
        np.save(f, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class Sam3Wrapper:
    def __init__(self, checkpoint_path: Path, gpu_id: int) -> None:
        """
        Load model at `checkpoint_path` onto CUDA device ID `gpu_id`.
        """
        logger.info(f"Loading SAM3 on {gpu_id=} PID={os.getpid()}")

        # We are in a multiprocessing context and assume the parent process sees
        # all CUDA devices So let's select the GPU before loading the model:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Deferring expensive imports to worker processes
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

    def infer_image(
        self, image: Image, text_prompt: str | None, boxes: list[BoxPrompt] | None
    ) -> SegmentImageResult:
        """
        Run inference on single image+prompt input.
        """
        import torch
        from sam3.model.box_ops import box_xywh_to_cxcywh
        from sam3.visualization_utils import normalize_bbox

        state = self.processor.set_image(image)

        if text_prompt:
            state = self.processor.set_text_prompt(state=state, prompt=text_prompt)

        if boxes:
            box_input_xywh = [[x, y, w, h] for (_,x,y,w,h) in boxes]
            box_input_cxcywh = box_xywh_to_cxcywh(
                torch.tensor(box_input_xywh).view(-1, 4)
            )
            norm_boxes_cxcywh = normalize_bbox(box_input_cxcywh, *image.size)
            labels = [label for (label,*_) in boxes]
            for box, label in zip(norm_boxes_cxcywh, labels):
                state = self.processor.add_geometric_prompt(
                    state=state, box=box, label=label
                )

        return SegmentImageResult(
            scores=state["scores"].cpu().float().numpy().tolist(),
            boxes=state["boxes"].cpu().float().numpy().tolist(),
            masks_npy=ndarray_to_base64(state["masks"].cpu().bool().squeeze(1).numpy()),
        )


def load_sam3_model(counter: SynchronizedBase) -> None:
    """
    Initialize worker: each worker takes the next available GPU and
    loads SAM3 onto it.
    """
    global sam3_wrapper
    with counter.get_lock():
        counter.value += 1
        gpu_id = counter.value - 1

    sam3_wrapper = Sam3Wrapper(CHECKPOINT_PATH, gpu_id)


@asynccontextmanager
async def worker_lifespan(_app: FastAPI):
    """
    Set up a Process pool for the duration of the API: one worker per GPU; each
    worker initializes sam3_wrapper onto its assigned device.
    """
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
    """
    Image segmentation task
    """
    with smart_open.open(req.image_uri, "rb") as fp:
        image = Image.open(fp)
        return sam3_wrapper.infer_image(image, req.text_prompt, req.box_prompts)


@app.post("/", response_model=SegmentImageResult)
def process_image(request: SegmentImageRequest) -> SegmentImageResult:
    """
    Submit a single image with text and/or bounding box prompts for
    segmentation. 
    """
    return worker_pool.submit(segment, request).result()
