import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from rich.logging import RichHandler

from .executor import SAM3Executor
from .schema import (
    ProcessImageRequest,
    ProcessImageResponse,
    ProcessWebDatasetRequest,
    ProcessWebDatasetResponse,
)

logging.basicConfig(
    level="INFO",
    format="[%(process)d] %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

logger = logging.getLogger("sam3_inference_server")


async def get_executor(request: Request) -> SAM3Executor:
    return request.app.state.executor


Executor = Annotated[SAM3Executor, Depends(get_executor)]


@asynccontextmanager
async def worker_lifespan(app: FastAPI):
    """
    Set up a Process pool for the duration of the API: one worker per GPU
    """
    executor = SAM3Executor()
    executor.start()
    app.state.executor = executor
    yield
    executor.shutdown()


app = FastAPI(lifespan=worker_lifespan)


@app.post("/process-wds")
async def process_webdataset(
    payload: ProcessWebDatasetRequest, executor: Executor
) -> ProcessWebDatasetResponse:
    """
    Submit a path to a WebDataset for SAM3 segmentation.
    """
    future = executor.submit(payload)
    return await asyncio.wrap_future(future)


@app.post("/process-image")
async def process_image(
    payload: ProcessImageRequest, executor: Executor
) -> ProcessImageResponse:
    """
    Submit a single image URI for SAM3 segmentation.
    """
    future = executor.submit(payload)
    return await asyncio.wrap_future(future)
