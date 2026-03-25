from pathlib import Path

from globus_compute_sdk import Client

gcc = Client()

def submit_batch(dataset_path: str | Path):
    from pathlib import Path

    import requests
    BASE_URL = "http://localhost:8000"

    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Could not locate {dataset_path}")
    if dataset_path.suffix != ".tar":
        raise ValueError(f"Expected .tar file extension: {dataset_path}")

    resp = requests.post(
        f"{BASE_URL}/process-batch",
        json={"dataset_path": dataset_path.as_posix()},
    )
    resp.raise_for_status()
    return resp.json()


def submit_image(image_uri: str, prompt: str):
    from pathlib import Path

    import requests
    BASE_URL = "http://localhost:8000"

    if "://" not in image_uri:
        image_uri = "file://" + Path(image_uri).resolve().as_posix()

    resp = requests.post(
        f"{BASE_URL}/process-image",
        json={"image_uri": image_uri, "prompt": {"text": prompt}},
    )
    resp.raise_for_status()
    return resp.json()


for func in [submit_batch, submit_image]:
    func_uuid = gcc.register_function(func)
    print(f"Registered {func.__name__}: {func_uuid=}")
