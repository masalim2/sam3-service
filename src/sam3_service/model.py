import logging
from time import perf_counter
from typing import Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import BatchEncoding, Sam3Model

from .config import CHECKPOINT_DIR
from .data import preprocess_image, processor
from .schema import SAM3Result, Sample

logger = logging.getLogger(__name__)


def to_labelmap(masks: torch.Tensor):
    if masks.shape[0] > 0:
        labelmap = masks.argmax(dim=0).byte() + 1
        labelmap[~masks.any(dim=0)] = 0
        return labelmap.cpu().numpy()
    return np.zeros(masks.shape[1:])


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
        ).to(self.device)  # type: ignore

        logger.info(f"Loaded SAM3 on {gpu_id=}")

    @torch.autocast("cuda", dtype=torch.bfloat16)
    @torch.inference_mode()
    def infer_image(self, image: Image.Image, text_prompt: str) -> SAM3Result:
        inputs = preprocess_image(image, text_prompt).to(self.device)
        outputs = self.model(**inputs)
        r = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),  # type: ignore[unresolved-attribute]
        )[0]
        return SAM3Result(
            key=text_prompt,
            scores=r["scores"].cpu().tolist(),
            boxes=r["boxes"].cpu().tolist(),
            masks=to_labelmap(r["masks"]),
        )

    @torch.autocast("cuda", dtype=torch.bfloat16)
    @torch.inference_mode()
    def infer_batch(
        self,
        loader: DataLoader[tuple[list[Sample], BatchEncoding]],
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> Iterator[SAM3Result]:
        total_images = 0
        t0 = perf_counter()

        samples: list[Sample]
        inputs: BatchEncoding

        for samples, inputs in loader:
            inputs = inputs.to(self.device, non_blocking=True)

            outputs = self.model(**inputs)

            postprocessed = processor.post_process_instance_segmentation(
                outputs,
                target_sizes=inputs["original_sizes"].tolist(),
                threshold=threshold,
                mask_threshold=mask_threshold,
            )

            yield from (
                SAM3Result(
                    key=s.name,
                    scores=r["scores"].cpu().tolist(),
                    boxes=r["boxes"].cpu().tolist(),
                    masks=to_labelmap(r["masks"]),
                )
                for s, r in zip(samples, postprocessed)
            )

            total_images += len(samples)
            elapsed = perf_counter() - t0
            logger.info(
                f"Cumulative: {total_images} images , "
                f"{total_images / elapsed:.1f} img/s avg"
            )
