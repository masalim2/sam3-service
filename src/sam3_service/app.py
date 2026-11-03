import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import unquote, urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from rich.logging import RichHandler

from .config import STAGING_ROOT
from .executor import SAM3Error, SAM3Executor, SAM3InputError
from .schema import (
    BatchRequest,
    BatchResponse,
    ImageRequest,
    Prompt,
    SAM3ImageResult,
    SegmentationTask,
    StagedFile,
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


@app.exception_handler(SAM3InputError)
async def sam3_input_error_handler(_req: Request, exc: SAM3InputError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/health")
async def health(executor: Executor) -> JSONResponse:
    """
    Healthy once the model is loaded on every GPU, until a GPU worker dies.
    """
    ready = executor.ready()
    return JSONResponse(status_code=200 if ready else 503, content={"ready": ready})


# Must be usable as a single path component.
_USER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._@-]{0,127}")


def resolve_data_uri(data_uri: str, user_id: str) -> StagedFile:
    """
    Confine data_uri to a file in the caller's staging area,
    <STAGING_ROOT>/<user_id>/. Accepts a file:// URI or an absolute path.

    This is only the up-front check; the worker re-opens the file by walking
    the returned parts with O_NOFOLLOW, so swapping a component for a symlink
    after this check is caught there.
    """
    if not _USER_ID_RE.fullmatch(user_id) or ".." in user_id:
        raise HTTPException(400, "invalid X-First-User-Id")

    anchor = STAGING_ROOT / user_id
    bad = HTTPException(
        400,
        f"data_uri must be a file:// URI or absolute path to a file in your "
        f"staging area ({anchor}/); got: {data_uri!r}",
    )

    path = data_uri
    if "://" in data_uri or data_uri.startswith("file:"):
        url = urlsplit(data_uri)
        if (
            url.scheme != "file"
            or url.netloc not in ("", "localhost")
            or url.query
            or url.fragment
        ):
            raise bad
        path = unquote(url.path)

    if "\0" in path or not os.path.isabs(path):
        raise bad
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(anchor):
        raise bad
    parts = resolved.relative_to(anchor).parts
    if not parts:
        raise bad
    return StagedFile(anchor=anchor, parts=parts)


@app.post("/v1/promptable-segmentation")
async def promptable_segmentation(
    task: SegmentationTask,
    executor: Executor,
    x_first_user_id: Annotated[str, Header()],
) -> BatchResponse | SAM3ImageResult:
    """
    Segment one image by text prompt, or a WebDataset .tar of (image, prompts).

    The FIRST gateway authenticates the caller and sets X-First-User-Id;
    data_uri must point into that user's staging area.
    """
    staged = resolve_data_uri(task.data_uri, x_first_user_id)

    if task.inference_type == "single-image":
        if not task.single_image_prompt:
            raise HTTPException(400, "single-image requires single_image_prompt")
        image = ImageRequest(
            image=staged,
            prompt=Prompt(text=task.single_image_prompt),
        )
        return await asyncio.wrap_future(executor.submit(image))

    if staged.path.suffix != ".tar":
        raise HTTPException(400, f"Expected .tar file extension: {staged.path}")
    batch = BatchRequest(dataset=staged)
    return await asyncio.wrap_future(executor.submit(batch))
