import logging
import multiprocessing as mp
import os
import tarfile
from contextlib import ExitStack
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.synchronize import Event
from queue import Empty, Queue
from threading import Thread
from time import perf_counter
from typing import NewType, overload
from uuid import uuid4

import torch
from PIL import Image, UnidentifiedImageError
from rich.logging import RichHandler

from . import safe_fs

from .config import CHECKPOINT_DIR
from .data import build_dataloader
from .model import Sam3Wrapper
from .schema import (
    BatchRequest,
    BatchResponse,
    ImageRequest,
    SAM3ImageResult,
)

logger = logging.getLogger(__name__)

Request = BatchRequest | ImageRequest
RequestId = NewType("RequestId", str)
MP_CONTEXT = mp.get_context("spawn")


class SAM3Error(RuntimeError): ...


class SAM3InputError(SAM3Error):
    """The caller's staged data was missing, unsafe, or undecodable."""


# Failures caused by the caller's staged data rather than by the server.
INPUT_ERRORS = (
    safe_fs.UnsafePathError,
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    PermissionError,
    tarfile.TarError,
    UnidentifiedImageError,
    Image.DecompressionBombError,
)


Result = BatchResponse | SAM3ImageResult | SAM3Error


class SAM3Executor:
    def __init__(self):
        num_gpus = torch.cuda.device_count()
        if num_gpus < 1:
            raise RuntimeError(
                "torch.cuda.device_count() does not report any GPUs available"
            )

        self.exit_event = MP_CONTEXT.Event()
        self.request_q: Queue[tuple[RequestId, Request]] = MP_CONTEXT.Queue()  # type: ignore
        self.result_q: Queue[tuple[RequestId, Result]] = MP_CONTEXT.Queue()  # type: ignore
        self.futures: dict[RequestId, Future[Result]] = {}

        self.workers = [
            _SAM3Worker(
                gpu_id=gpu_id,
                exit_event=self.exit_event,
                request_q=self.request_q,
                result_q=self.result_q,
            )
            for gpu_id in range(num_gpus)
        ]

    @overload
    def submit(self, request: BatchRequest) -> Future[BatchResponse]: ...
    @overload
    def submit(self, request: ImageRequest) -> Future[SAM3ImageResult]: ...
    def submit(self, request: Request) -> Future[Result]:
        req_id = RequestId(uuid4().hex)
        self.request_q.put_nowait((req_id, request))
        self.futures[req_id] = Future()
        return self.futures[req_id]

    def ready(self) -> bool:
        """True once every GPU worker has loaded the model and none has died."""
        return not self.exit_event.is_set() and all(
            worker.ready.is_set() for worker in self.workers
        )

    def _health_checker(self):
        while not self.exit_event.wait(timeout=3.0):
            for worker in self.workers:
                if not worker.is_alive():
                    logger.critical("Detected dead worker process; shutting down")
                    self.exit_event.set()
                    break

    def _result_collector(self):
        while not self.exit_event.is_set():
            try:
                req_id, result = self.result_q.get(timeout=1.0)
            except Empty:
                continue

            future = self.futures.pop(req_id)

            if isinstance(result, Exception):
                future.set_exception(result)
            else:
                future.set_result(result)

    def start(self):
        logger.info(
            f"Initializing SAM3 GPU Worker Pool with {len(self.workers)} workers"
        )
        for worker in self.workers:
            worker.start()

        Thread(target=self._health_checker, daemon=True).start()
        Thread(target=self._result_collector, daemon=True).start()

    def shutdown(self):
        logger.info("Shutdown: setting exit event")
        self.exit_event.set()

        for worker in self.workers:
            worker.join(timeout=10.0)

        for worker in self.workers:
            if worker.is_alive():
                worker.kill()

        logger.info("SAM3Executor shutdown gracefully")


class _SAM3Worker(MP_CONTEXT.Process):  # type: ignore[unsupported-base]
    model: Sam3Wrapper
    writer_thread: ThreadPoolExecutor

    def __init__(
        self,
        gpu_id: int,
        exit_event: Event,
        request_q: Queue[tuple[RequestId, Request]],
        result_q: Queue[tuple[RequestId, Result]],
    ) -> None:
        super().__init__()
        self.gpu_id = gpu_id
        self.exit_event = exit_event
        self.request_q = request_q
        self.result_q = result_q
        self.ready = MP_CONTEXT.Event()

    def run(self) -> None:
        logging.basicConfig(
            level="INFO",
            format="[%(process)d] %(message)s",
            datefmt="[%X]",
            handlers=[RichHandler()],
        )

        logger.info(f"Loading SAM3Wrapper on {self.gpu_id=}")
        self.model = Sam3Wrapper(self.gpu_id)
        self.ready.set()

        # Must be single-threaded; tarfile is not threadsafe:
        self.writer_thread = ThreadPoolExecutor(max_workers=1)

        while not self.exit_event.is_set():
            try:
                req_id, request = self.request_q.get(timeout=1.0)
            except Empty:
                continue

            try:
                self.dispatch(req_id, request)
            except INPUT_ERRORS as exc:
                logger.warning(f"Rejected staged data for {request=}: {exc!r}")
                self.result_q.put((req_id, SAM3InputError(f"Rejected input: {exc}")))
            except Exception as exc:
                logger.exception(msg := f"Worker exception during {request=}: {exc}")
                self.result_q.put((req_id, SAM3Error(msg)))

        self.writer_thread.shutdown(wait=True)

    def dispatch(self, req_id: RequestId, request: Request) -> None:
        if isinstance(request, BatchRequest):
            self.handle_batch_request(req_id, request)
        elif isinstance(request, ImageRequest):
            self.handle_image_request(req_id, request)
        else:
            raise AssertionError(f"Unknown request type: {request}")

    def handle_batch_request(self, req_id: RequestId, request: BatchRequest) -> None:
        staged = request.dataset
        *dir_parts, tar_name = staged.parts
        result_name = tar_name.removesuffix(".tar") + ".results.tar"
        result_path = staged.path.with_name(result_name)
        write_futs: list[Future] = []

        with ExitStack() as stack:
            # Open the input once, without following symlinks; the DataLoader
            # workers inherit this fd and never re-open the staged path.
            dir_fd = safe_fs.open_dir(staged.anchor, dir_parts)
            stack.callback(os.close, dir_fd)
            tar_fd = safe_fs.open_file(dir_fd, tar_name)
            stack.callback(os.close, tar_fd)

            loader = build_dataloader(tar_fd, tar_name, batch_size=4, num_workers=8)

            t0 = perf_counter()
            with (
                safe_fs.atomic_write(dir_fd, result_name) as fp,
                tarfile.open(fileobj=fp, mode="w") as tf,
            ):
                for result in self.model.infer_batch(loader):
                    fut = self.writer_thread.submit(result.dump_to_tarfile, tf)
                    write_futs.append(fut)

                for fut in write_futs:
                    fut.result(timeout=90)

        logger.info(f"Batch inference outputs written to {result_path}")
        info = {
            "model_weights": CHECKPOINT_DIR.as_posix(),
            "prompts/sec": len(write_futs) / (perf_counter() - t0),
        }
        self.result_q.put(
            (req_id, BatchResponse(result_path=result_path, runtime_info=info))
        )

    def handle_image_request(self, req_id: RequestId, request: ImageRequest) -> None:
        result = self.model.infer_image(request)
        self.result_q.put((req_id, result))
