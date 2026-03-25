from typing import TypedDict, Literal

from globus_compute_sdk import Client

gcc = Client()

class Payload(TypedDict):
    inference_type: Literal["single-image", "batch"]
    data_uri: str
    single_image_prompt: str | None


def submit(payload: Payload):
    from pathlib import Path

    import requests

    BASE_URL = "http://localhost:8000"

    if payload["inference_type"] == "single-image":
        image_uri = payload["data_uri"]
        prompt  = payload["single_image_prompt"]

        if "://" not in image_uri:
            image_uri = "file://" + Path(image_uri).resolve().as_posix()

        resp = requests.post(
            f"{BASE_URL}/process-image",
            json={"image_uri": image_uri, "prompt": {"text": str(prompt)}},
        )

    elif payload["inference_type"] == "batch":
        dataset_path = Path(dataset_path).resolve()

        if not dataset_path.is_file():
            raise FileNotFoundError(f"Could not locate {dataset_path}")
        if dataset_path.suffix != ".tar":
            raise ValueError(f"Expected .tar file extension: {dataset_path}")

        resp = requests.post(
            f"{BASE_URL}/process-batch",
            json={"dataset_path": dataset_path.as_posix()},
        )

    else:
        raise ValueError(f"Unknown {payload['inference_type']=}")

    resp.raise_for_status()
    return resp.json()


func_uuid = gcc.register_function(submit)
print(f"Registered {submit.__name__}: {func_uuid=}")
