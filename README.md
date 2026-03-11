# Setup

Run `setup.sh` or follow the steps therein to get the virtual environment and dependencies installed.

If you have read access to the weights file at /eagle/datascience/msalim/huggingface/hub/models--facebook--sam3/sam3.pt, the 
server will read from that location by default.  Otherwise, set the environment variable `SAM3_CHECKPOINT` the path 
of the SAM3 model checkpoint file that you have access to.

# Launching the server interactively

From an interactive session on a Sophia compute node, you can run the SAM3 inference server using:

```
uvicorn sam3server:app --host 0.0.0.0 --port 8000
```

The server will detect the number of available CUDA devices and launch one
worker per GPU.  The models are lazily loaded onto each GPU, so the first 8
requests will be rather slow.  After that, you will find the response time of
the API is quite a bit faster!

# Testing the API

```python
import requests

resp = requests.post(
    'http://sophia-gpu-10:8000/', 
    json={
        "image_uri": "file:///home/msalim/sam3server/examples/images/groceries.jpg", 
        "text_prompt": "light", 
        "box_prompts": []
    }
)
```