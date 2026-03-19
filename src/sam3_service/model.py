import logging
from time import perf_counter
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from transformers import BatchEncoding, Sam3Model, Sam3Processor

from .config import CHECKPOINT_DIR
from .schema import SAM3Result, SamplePrompt

logger = logging.getLogger(__name__)
processor = Sam3Processor.from_pretrained(CHECKPOINT_DIR, local_files_only=True)

def quantile_norm(img):
    import numpy as np
    p_lo, p_hi = np.percentile(img, [1, 99])
    img = np.clip((img - p_lo) / (p_hi - p_lo), 0, 1)

    # Convert to uint8 0-255 range, which is what vision preprocessors expect
    img = (img * 255).astype(np.uint8)
    return img[np.newaxis,:,:]

def preprocess(samples: list[SamplePrompt]) -> tuple[list[SamplePrompt], BatchEncoding]:
    """
    Preprocess SamplePrompts into SAM3 input batches.
    """
    text_prompts = [s.text for s in samples]
    box_prompts = [
        [list(box[:4]) for box in s.boxes] if s.boxes else None for s in samples
    ]
    box_labels = [
        [int(box.label) for box in s.boxes] if s.boxes else None for s in samples
    ]

    process_kwargs: dict[str, Any] = {
        "images": [quantile_norm(s.image) for s in samples],
        "return_tensors": "pt",
    }

    if any(text_prompts):
        process_kwargs["text"] = text_prompts
    if any(box_prompts):
        process_kwargs["input_boxes"] = box_prompts
        process_kwargs["input_boxes_labels"] = box_labels

    inputs = processor(**process_kwargs)
    return samples, inputs


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
        image = quantile_norm(image)
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(
            self.device
        )
        outputs = self.model(**inputs)
        r = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        return SAM3Result(
            key=text_prompt,
            scores=r["scores"].cpu().tolist(),
            boxes=r["boxes"].cpu().tolist(),
            masks=r["masks"].cpu().float().numpy(),
        )

    @torch.autocast("cuda", dtype=torch.bfloat16)
    @torch.inference_mode()
    def infer_batch(
        self,
        dataset: IterableDataset[SamplePrompt],
        threshold: float = 0.3,
        mask_threshold: float = 0.3,
        batch_size: int = 16,
        loader_num_workers: int = 2,
    ) -> list[SAM3Result]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=loader_num_workers,
            collate_fn=preprocess,
            pin_memory=True,
            in_order=False,
            persistent_workers=False,
            multiprocessing_context="fork",
        )

        total_images = 0
        results = []

        t0 = perf_counter()
        for samples, inputs in loader:
            inputs = inputs.to(self.device, non_blocking=True)

            outputs = self.model(**inputs)

            postprocessed = processor.post_process_instance_segmentation(
                outputs,
                target_sizes=inputs["original_sizes"].tolist(),
                threshold=threshold,
                mask_threshold=mask_threshold,
            )

            results.extend(
                SAM3Result(
                    key=s.key,
                    scores=r["scores"].cpu().tolist(),
                    boxes=r["boxes"].cpu().tolist(),
                    masks=r["masks"].cpu().float().numpy(),
                )
                for s, r in zip(samples, postprocessed)
            )

            total_images += len(samples)
            elapsed = perf_counter() - t0
            logger.info(
                f"Cumulative: {total_images} images , "
                f"{total_images / elapsed:.1f} img/s avg"
            )

        return results
