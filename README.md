# Setup

Run `setup.sh` or follow the steps therein to get the virtual environment and dependencies installed.

If you have read access to the weights file at /eagle/datascience/msalim/huggingface/hub/models--facebook--sam3/sam3.pt, the 
server will read from that location by default.  Otherwise, set the environment variable `SAM3_DIR` to the path 
of a SAM3 weights directory that you have access to.

# Launching the server

The server detects the visible CUDA devices and runs one worker per GPU.  It runs on
FIRST as a pilot replica listening on a Unix domain socket; see the `sophia/sam3`
LaunchTemplate in gateway-api-fixtures.  From an interactive session on a compute
node you can run it the same way:

```
uvicorn sam3_service.app:app --uds /tmp/sam3.sock
```

`GET /health` returns 503 until the model is loaded on every GPU, and again if a
GPU worker dies.

Set `SAM3_DIR` to the weights directory before launch.  This is the only way to
choose a checkpoint: requests cannot select weights, and a request that still
sends `weights_dir_override` is rejected with 422.  To serve fine-tuned weights,
launch a server with `SAM3_DIR` pointing at them.  Only `model.safetensors` weights
are loaded; pickle-based `.bin`/`.pt` checkpoints are refused, and the server
will not start if `SAM3_DIR` is inside a user staging area.

# Staged data

Requests must carry an `X-First-User-Id` header (the FIRST gateway sets it after
authenticating the caller).  `data_uri` must be a `file://` URI or absolute path to
a file under that user's staging area,
`$SAM3_STAGING_ROOT/<user id>/` (default root:
`/eagle/IRIBeta/inference_data_staging/user-staging`); anything else is rejected
with 400.  Staged files are opened without following symlinks, and hardlinked or
special files are refused.

# Testing the API

```bash
curl --unix-socket /tmp/sam3.sock http://localhost/v1/promptable-segmentation \
  -H 'Content-Type: application/json' \
  -H 'X-First-User-Id: <id>' \
  -d '{
    "inference_type": "single-image",
    "data_uri": "/eagle/IRIBeta/inference_data_staging/user-staging/<id>/groceries.jpg",
    "single_image_prompt": "light"
  }'
```

For a WebDataset tar, send `"inference_type": "batch"` with `data_uri` set to the
`.tar` path.  Results are written beside it as `<name>.results.tar`.

The `sam3-service submit-*` commands send the header from `--user-id` (or
`$FIRST_USER_ID`) when calling the server directly.
