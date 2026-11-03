import logging
from io import BytesIO
from time import perf_counter
from typing import Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import BatchEncoding, Sam3Model

from .config import CHECKPOINT_DIR
from . import safe_fs
from .data import InputInfo, preprocess_image, processor
from .schema import ImageRequest, SAM3ImageResult

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
        Load the server's launch-configured weights (CHECKPOINT_DIR) onto CUDA
        device ID `gpu_id`.
        """
        self.device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        logger.info(f"Loading SAM3 on {self.device}")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # safetensors only: refuse pickle-based (.bin/.pt) weight files, which
        # can execute arbitrary code when loaded.
        self.model = Sam3Model.from_pretrained(
            CHECKPOINT_DIR,
            local_files_only=True,
            use_safetensors=True,
            trust_remote_code=False,
        ).to(self.device)  # type: ignore

        logger.info(f"Loaded SAM3 from {CHECKPOINT_DIR} on {gpu_id=}")

    @torch.autocast("cuda", dtype=torch.bfloat16)
    @torch.inference_mode()
    def infer_image(self, request: ImageRequest) -> SAM3ImageResult:
        staged = request.image
        data = safe_fs.read_file(staged.anchor, staged.parts)
        with Image.open(BytesIO(data)) as image:
            inputs = preprocess_image(image, request.prompt).to(self.device)

        outputs = self.model(**inputs)

        r = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),  # type: ignore[unresolved-attribute]
        )[0]

        return SAM3ImageResult(
            input_image=request.image.path.as_uri(),
            prompt=request.prompt,
            batch_slug="",
            scores=r["scores"].cpu().tolist(),
            boxes=r["boxes"].cpu().tolist(),
            labelmap_npy=to_labelmap(r["masks"]),
        )

    @torch.autocast("cuda", dtype=torch.bfloat16)
    @torch.inference_mode()
    def infer_batch(
        self,
        loader: DataLoader,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> Iterator[SAM3ImageResult]:
        total_images = 0
        t0 = perf_counter()

        info_list: list[InputInfo]
        inputs: BatchEncoding

        for info_list, inputs in loader:
            inputs = inputs.to(self.device, non_blocking=True)

            outputs = self.model(**inputs)

            postprocessed = processor.post_process_instance_segmentation(
                outputs,
                target_sizes=inputs["original_sizes"].tolist(),
                threshold=threshold,
                mask_threshold=mask_threshold,
            )

            yield from (
                SAM3ImageResult(
                    input_image=info.image_filename,
                    prompt=info.prompt,
                    batch_slug=info.batch_slug,
                    scores=r["scores"].cpu().tolist(),
                    boxes=r["boxes"].cpu().tolist(),
                    labelmap_npy=to_labelmap(r["masks"]),
                )
                for info, r in zip(info_list, postprocessed)
            )

            total_images += len(info_list)
            elapsed = perf_counter() - t0
            logger.info(
                f"Cumulative: {total_images} images , "
                f"{total_images / elapsed:.1f} img/s avg"
            )
