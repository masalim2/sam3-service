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

Set `SAM3_DIR` to the default weights directory.  Fine-tuned weights live in
subdirectories of it, and load on first use when a request names them in
`weights_dir_override`.

# Testing the API

```bash
curl --unix-socket /tmp/sam3.sock http://localhost/v1/promptable-segmentation \
  -H 'Content-Type: application/json' \
  -d '{
    "inference_type": "single-image",
    "data_uri": "/home/msalim/sam3server/examples/images/groceries.jpg",
    "single_image_prompt": "light"
  }'
```

For a WebDataset tar, send `"inference_type": "batch"` with `data_uri` set to the
`.tar` path.  Results are written beside it as `<name>.results.tar`.
