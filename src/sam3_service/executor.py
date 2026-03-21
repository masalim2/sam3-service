import logging
import multiprocessing as mp
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.synchronize import Event
from queue import Empty, Queue
from threading import Thread
from typing import NewType, overload
from uuid import uuid4

import smart_open
import torch
from PIL import Image
from rich.logging import RichHandler

from .io import build_dataloader, save_batch
from .model import Sam3Wrapper
from .schema import (
    BatchRequest,
    BatchResponse,
    ImageRequest,
    ImageResponse,
    SAM3Result,
)

logger = logging.getLogger(__name__)

Request = BatchRequest | ImageRequest
RequestId = NewType("RequestId", str)
MP_CONTEXT = mp.get_context("spawn")


class SAM3Error(RuntimeError): ...


Result = BatchResponse | ImageResponse | SAM3Error


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
    def submit(self, request: ImageRequest) -> Future[ImageResponse]: ...
    def submit(self, request: Request) -> Future[Result]:
        req_id = RequestId(uuid4().hex)
        self.request_q.put_nowait((req_id, request))
        self.futures[req_id] = Future()
        return self.futures[req_id]

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
    writer_pool: ThreadPoolExecutor

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

    def run(self) -> None:
        logging.basicConfig(
            level="INFO",
            format="[%(process)d] %(message)s",
            datefmt="[%X]",
            handlers=[RichHandler()],
        )

        logger.info(f"Loading SAM3Wrapper on {self.gpu_id=}")
        self.model = Sam3Wrapper(self.gpu_id)
        self.writer_pool = ThreadPoolExecutor(max_workers=5)

        while not self.exit_event.is_set():
            try:
                req_id, request = self.request_q.get(timeout=1.0)
            except Empty:
                continue

            try:
                self.dispatch(req_id, request)
            except Exception as exc:
                logger.exception(msg := f"Worker exception during {request=}: {exc}")
                self.result_q.put((req_id, SAM3Error(msg)))

        self.writer_pool.shutdown(wait=True)

    def dispatch(self, req_id: RequestId, request: Request) -> None:
        if isinstance(request, BatchRequest):
            self.handle_batch_request(req_id, request)
        elif isinstance(request, ImageRequest):
            self.handle_image_request(req_id, request)
        else:
            raise AssertionError(f"Unknown request type: {request}")

    def handle_batch_request(self, req_id: RequestId, request: BatchRequest) -> None:
        loader = build_dataloader(request.dataset_path, batch_size=4, num_workers=4)
        results = self.model.infer_batch(loader)
        self.writer_pool.submit(self.save_batch_results, req_id, results, request)

    def handle_image_request(self, req_id: RequestId, request: ImageRequest) -> None:
        with smart_open.open(request.image_uri, "rb") as fp:
            image = Image.open(fp)
            res = self.model.infer_image(image, request.text_prompt)

        resp = ImageResponse(
            num_objects=len(res.scores),
            boxes=res.boxes,
            scores=res.scores,
        )
        self.result_q.put((req_id, resp))

    def save_batch_results(
        self,
        req_id: RequestId,
        results: list[SAM3Result],
        request: BatchRequest,
    ) -> None:
        try:
            result_dir = request.dataset_path.with_suffix(".results")
            save_batch(results, result_dir)
        except Exception as exc:
            logger.exception(
                msg
                := f"Worker Exception while saving batch results for {request=}: {exc}"
            )
            self.result_q.put((req_id, SAM3Error(msg)))
        else:
            self.result_q.put((req_id, BatchResponse(result_dir=result_dir)))
