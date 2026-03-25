import time

from globus_compute_sdk import Client
from globus_compute_sdk.errors import TaskPending

gcc = Client()

# sophia-sam3 endpoint @ /home/openinference_svc/
# Registered submit_batch: func_uuid='805ad46e-e7d0-4319-ac75-8556a1e4de48'
# Registered submit_image: func_uuid='6846d472-3714-49c9-87d1-2cec3124d582'
ENDPOINT_ID = "25662117-d605-4f43-af9b-1b9ae390e4e2"
FUNCS = {
    "submit_batch": "37e51c9c-b63f-4ac4-aee2-c070a5b301eb",
    "submit_image": "e8cc9f78-2c7f-4cb1-954a-f949b2f642fa",
}


batch_task_id = gcc.run(
    dataset_path="/eagle/inference_service/sam3-service/examples/wds-test/shard-00000.tar",
    endpoint_id=ENDPOINT_ID,
    function_id=FUNCS["submit_batch"],
)
print(f"Submitted {batch_task_id=}")

image_task_id = gcc.run(
    image_uri="https://raw.githubusercontent.com/masalim2/sam3-service/refs/heads/main/examples/images/groceries.jpg",
    prompt="grocery bag",
    endpoint_id=ENDPOINT_ID,
    function_id=FUNCS["submit_image"],
)
print(f"Submitted {image_task_id=}")

print("Polling on batch task...")
while True:
  try:
    print(gcc.get_result(batch_task_id))
    break
  except TaskPending as exc:
    print(f"Pending: {exc}")
  except Exception as e:
    print(f"Batch task failed: {e}")
    break
  time.sleep(10)

print("Polling on single image task...")
while True:
  try:
    print(gcc.get_result(image_task_id))
    break
  except TaskPending as exc:
    print(f"Pending: {exc}")
  except Exception as e:
    print(f"Single image task failed: {e}")
    break
  time.sleep(10)