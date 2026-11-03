import os
from pathlib import Path

# Everything here is fixed when the server launches and is never taken from an
# HTTP request.

# Root of per-user staging areas. Requests may only read data from (and write
# results to) <STAGING_ROOT>/<X-First-User-Id>/; everything there is untrusted.
STAGING_ROOT = Path(
    os.environ.get(
        "SAM3_STAGING_ROOT",
        "/eagle/IRIBeta/inference_data_staging/user-staging",
    )
).resolve()

# Model checkpoint directory. Loading a checkpoint can execute code, so only
# an operator-controlled path may be used, and never one in a staging area.
CHECKPOINT_DIR = Path(
    os.environ.get(
        "SAM3_DIR",
        "/eagle/datascience/msalim/huggingface/hub/models--facebook--sam3/",
    )
).resolve()

if (
    CHECKPOINT_DIR.is_relative_to(STAGING_ROOT)
    or "user-staging" in CHECKPOINT_DIR.parts
):
    raise RuntimeError(
        f"Refusing to load checkpoints from a user staging area: {CHECKPOINT_DIR}"
    )
