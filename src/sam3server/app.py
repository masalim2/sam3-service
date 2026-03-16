import concurrent.futures
import logging
import multiprocessing as mp
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from rich.logging import RichHandler

import sam3server.model

logging.basicConfig(
    level="INFO",
    format="[PID:%(process)d] %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

logger = logging.getLogger("sam3_inference_server")
worker_pool: concurrent.futures.ProcessPoolExecutor


class Sam3Request(BaseModel):
    dataset_path: Path
    result_path: Path


@asynccontextmanager
async def worker_lifespan(_app: FastAPI):
    """
    Set up a Process pool for the duration of the API: one worker per GPU
    """
    global worker_pool

    num_gpu = torch.cuda.device_count()
    if num_gpu == 0:
        raise RuntimeError("No GPUs detected")

    counter = mp.Value("i")
    logger.info(f"Initializing SAM3 GPU Worker Pool with {num_gpu=}")
    worker_pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=num_gpu,
        initializer=sam3server.model.load_sam3_model,
        initargs=(counter,),
    )
    yield
    worker_pool.shutdown(cancel_futures=True)


app = FastAPI(lifespan=worker_lifespan)


@app.post("/")
def process_webdataset(request: Sam3Request) -> dict[str, Any]:
    """
    Submit a path to a WebDataset for SAM3 segmentation.
    """
    return worker_pool.submit(
        sam3server.model.segment, request.dataset_path, request.result_path
    ).result()
