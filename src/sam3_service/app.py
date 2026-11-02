import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from rich.logging import RichHandler

from .config import CHECKPOINT_DIR
from .executor import SAM3Error, SAM3Executor
from .schema import (
    BatchRequest,
    BatchResponse,
    ImageRequest,
    Prompt,
    SAM3ImageResult,
    SegmentationTask,
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


@app.exception_handler(SAM3Error)
async def sam3_error_handler(_req: Request, exc: SAM3Error):
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/health")
async def health(executor: Executor) -> JSONResponse:
    """
    Healthy once the model is loaded on every GPU, until a GPU worker dies.
    """
    ready = executor.ready()
    return JSONResponse(status_code=200 if ready else 503, content={"ready": ready})


def resolve_weights(override: Path | None) -> Path:
    """
    Fine-tuned weights must live under the server's weights directory.
    """
    if override is None:
        return CHECKPOINT_DIR
    if not override.resolve().is_relative_to(CHECKPOINT_DIR.resolve()):
        raise HTTPException(400, f"weights_dir_override must be under {CHECKPOINT_DIR}")
    return override


@app.post("/v1/promptable-segmentation")
async def promptable_segmentation(
    task: SegmentationTask, executor: Executor
) -> BatchResponse | SAM3ImageResult:
    """
    Segment one image by text prompt, or a WebDataset .tar of (image, prompts).
    """
    weights_dir = resolve_weights(task.weights_dir_override)

    if task.inference_type == "single-image":
        if not task.single_image_prompt:
            raise HTTPException(400, "single-image requires single_image_prompt")
        image_uri = task.data_uri
        if "://" not in image_uri:
            image_uri = "file://" + Path(image_uri).resolve().as_posix()
        image = ImageRequest(
            image_uri=image_uri,
            prompt=Prompt(text=task.single_image_prompt),
            weights_dir=weights_dir,
        )
        return await asyncio.wrap_future(executor.submit(image))

    dataset_path = Path(task.data_uri).resolve()
    if not dataset_path.is_file():
        raise HTTPException(400, f"Could not locate {dataset_path}")
    if dataset_path.suffix != ".tar":
        raise HTTPException(400, f"Expected .tar file extension: {dataset_path}")
    batch = BatchRequest(dataset_path=dataset_path, weights_dir=weights_dir)
    return await asyncio.wrap_future(executor.submit(batch))
