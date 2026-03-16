import os
from pathlib import Path

CHECKPOINT_DIR = Path(
    os.environ.get(
        "SAM3_DIR",
        "/eagle/datascience/msalim/huggingface/hub/models--facebook--sam3/",
    )
)
