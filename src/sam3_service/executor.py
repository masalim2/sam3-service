import numpy as np
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

from .io import build_dataset, save_batch
from .model import Sam3Wrapper
from .schema import (
    ErrorResponse,
    ProcessImageRequest,
    ProcessImageResponse,
    ProcessWebDatasetRequest,
    ProcessWebDatasetResponse,
    SAM3Result,
)

logger = logging.getLogger(__name__)

Task = ProcessWebDatasetRequest | ProcessImageRequest
Result = ProcessWebDatasetResponse | ProcessImageResponse | ErrorResponse
TaskId = NewType("TaskId", str)
MP_CONTEXT = mp.get_context("spawn")


class SAM3Executor:
    def __init__(self):
        num_gpus = torch.cuda.device_count()
        if num_gpus < 1:
            raise RuntimeError(
                "torch.cuda.device_count() does not report any GPUs available"
            )

        self.exit_event = MP_CONTEXT.Event()
        self.task_q: Queue[tuple[TaskId, Task]] = MP_CONTEXT.Queue()  # type: ignore
        self.result_q: Queue[tuple[TaskId, Result]] = MP_CONTEXT.Queue()  # type: ignore
        self.futures: dict[TaskId, Future[Result]] = {}

        self.workers = [
            SAM3Worker(
                gpu_id=gpu_id,
                exit_event=self.exit_event,
                task_q=self.task_q,
                result_q=self.result_q,
            )
            for gpu_id in range(num_gpus)
        ]

    @overload
    def submit(
        self, request: ProcessWebDatasetRequest
    ) -> Future[ProcessWebDatasetResponse]: ...
    @overload
    def submit(self, request: ProcessImageRequest) -> Future[ProcessImageResponse]: ...
    def submit(self, request: Task) -> Future[Result]:
        task_id = TaskId(uuid4().hex)
        self.task_q.put_nowait((task_id, request))
        self.futures[task_id] = Future()
        return self.futures[task_id]

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
                task_id, result = self.result_q.get(timeout=1.0)
            except Empty:
                continue
            else:
                future = self.futures.pop(task_id)
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


class SAM3Worker(MP_CONTEXT.Process):  # type: ignore[unsupported-base]
    model: Sam3Wrapper
    writer_pool: ThreadPoolExecutor

    def __init__(
        self,
        gpu_id: int,
        exit_event: Event,
        task_q: Queue[tuple[TaskId, Task]],
        result_q: Queue[tuple[TaskId, Result]],
    ) -> None:
        super().__init__()
        self.gpu_id = gpu_id
        self.exit_event = exit_event
        self.task_q = task_q
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
                task_id, task = self.task_q.get(timeout=1.0)
            except Empty:
                continue

            try:
                self.handle_task(task_id, task)
            except Exception as exc:
                logger.exception(msg := f"SAM3 Worker Error for {task=}: {exc}")
                self.result_q.put((task_id, ErrorResponse(error=msg)))

        self.writer_pool.shutdown(wait=True)

    def handle_task(self, task_id: TaskId, task: Task) -> None:
        if isinstance(task, ProcessWebDatasetRequest):
            self.handle_webdataset_request(task_id, task)
        elif isinstance(task, ProcessImageRequest):
            self.handle_image_request(task_id, task)
        else:
            raise AssertionError(f"Unknown request type: {task}")

    def handle_webdataset_request(
        self, task_id: TaskId, task: ProcessWebDatasetRequest
    ) -> None:
        dataset = build_dataset(task.dataset_path)
        results = self.model.infer_batch(dataset, batch_size=4, loader_num_workers=4)
        self.writer_pool.submit(self.save_webdataset_results, task_id, results, task)

    def handle_image_request(self, task_id: TaskId, task: ProcessImageRequest) -> None:
        with smart_open.open(task.image_uri, "rb") as fp:
            # NB: a typical tomographic reconstruction is a monochromatic TIFF of dtype float32
            # with values in teh range [-5.0e-5, 2.0e-4].  Calling PIL.Image.convert('RGB') will 
            # indiscriminately squash this to a uint8 of pure zeroes.  You really want to do a 
            # quantile normalization prior to feeding it to the SAM3 preprocessor!
            image = np.array(Image.open(fp))
            res = self.model.infer_image(image, task.text_prompt)

        resp = ProcessImageResponse(
            num_objects=len(res.scores),
            boxes=res.boxes,
            scores=res.scores,
        )
        self.result_q.put((task_id, resp))

    def save_webdataset_results(
        self,
        task_id: TaskId,
        results: list[SAM3Result],
        task: ProcessWebDatasetRequest,
    ) -> None:
        try:
            result_dir = task.dataset_path.with_suffix(".results")
            save_batch(results, result_dir)
        except Exception as exc:
            logger.exception(msg := f"SAM3 Worker Error for {task=}: {exc}")
            self.result_q.put((task_id, ErrorResponse(error=msg)))
        else:
            self.result_q.put(
                (task_id, ProcessWebDatasetResponse(result_dir=result_dir))
            )
