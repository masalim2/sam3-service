import json
import logging
import time
from multiprocessing.sharedctypes import SynchronizedBase
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import Sam3Model

from sam3server.config import CHECKPOINT_DIR
from sam3server.data_loader import SamplePrompt, build_webdataset_loader, sam3_processor

_sam3_wrapper: "Sam3Wrapper"

logger = logging.getLogger(__name__)


def load_sam3_model(counter: SynchronizedBase) -> None:
    """
    Initialize worker: each worker takes the next available GPU and
    loads SAM3 onto it.
    """
    global _sam3_wrapper
    with counter.get_lock():
        counter.value += 1
        gpu_id = counter.value - 1

    _sam3_wrapper = Sam3Wrapper(gpu_id)


class Sam3Wrapper:
    def __init__(self, gpu_id: int) -> None:
        """
        Load model onto CUDA device ID `gpu_id`.
        """
        self.device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        logger.info(f"Loading SAM3 on {self.device}")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model = Sam3Model.from_pretrained(
            CHECKPOINT_DIR, local_files_only=True
        ).to(self.device)
        logger.info(f"Loaded SAM3 on {gpu_id=}")

    @torch.autocast("cuda", dtype=torch.bfloat16)
    @torch.inference_mode()
    def run_inference(self, dataset_path: Path, result_dir: Path) -> dict[str, Any]:
        loader = build_webdataset_loader(dataset_path, batch_size=8, num_workers=2)
        total_images = 0
        total_time = 0.0

        for samples, inputs in loader:
            t0 = time.perf_counter()

            inputs = inputs.to(self.device, non_blocking=True)
            outputs = self.model(**inputs)

            target_sizes = inputs["original_sizes"].tolist()

            results = sam3_processor.post_process_instance_segmentation(
                outputs,
                target_sizes=target_sizes,
            )

            torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - t0

            batch_size = len(samples)
            total_images += batch_size
            total_time += elapsed
            logger.info(
                f"Batch of {batch_size} in {elapsed:.3f}s "
                f"({batch_size / elapsed:.1f} img/s) | "
                f"cumulative: {total_images} images, "
                f"{total_images / total_time:.1f} img/s avg"
            )
            self.save(samples, results, result_dir)

        logger.info(
            f"Done. {total_images} images in {total_time:.2f}s "
            f"({total_images / total_time:.1f} img/s overall)"
        )
        return {"total_images": total_images, "total_time": total_time}

    @staticmethod
    def save(
        samples: list[SamplePrompt], results: list[dict], output_dir: Path
    ) -> None:
        checked_dirs = set()

        for sample, result in zip(samples, results):
            out = {
                "key": sample.key,
                "num_objects": len(result["masks"]),
                "scores": result["scores"].cpu().tolist(),
                "boxes": result["boxes"].cpu().tolist(),
                "masks_file": f"{sample.key}_masks.npy",
            }

            out_path = output_dir / f"{sample.key}.result.json"
            if out_path.parent not in checked_dirs:
                out_path.parent.mkdir(exist_ok=True, parents=True)
                checked_dirs.add(out_path.parent)

            with open(out_path, "w") as f:
                json.dump(out, f, indent=2)

            np.save(
                output_dir / f"{sample.key}_masks.npy", result["masks"].cpu().numpy()
            )


def segment(dataset_path: Path, result_dir: Path) -> dict[str, Any]:
    return _sam3_wrapper.run_inference(dataset_path, result_dir)
