from globus_compute_sdk import Client, Executor

gcc = Client()

# sophia-sam3 endpoint @ /home/openinference_svc/
ENDPOINT_ID = "6699c6a0-2ce3-4798-b85c-3e65d93b68dd"
FUNC_ID = "c5653f2c-98c8-4dd3-800d-ef7439c7e140"

with Executor(
    endpoint_id=ENDPOINT_ID,
    client=gcc,
    user_endpoint_config={
        "sam3_weights_dir": "/eagle/inference_service/sam3-service/weights/synaps-i/"
    },
) as gce:
    fut = gce.submit_to_registered_function(
        function_id=FUNC_ID,
        args=[{
            "inference_type": "batch",
            "data_uri": "/eagle/inference_service/sam3-service/examples/wds-test/shard-00000.tar",
        }],
    )
    print(f"Submitted batch inference {fut.task_id=}")
    print("Awaiting results...")
    print(fut.result(timeout=300))

with Executor(endpoint_id=ENDPOINT_ID, client=gcc) as gce:
    fut = gce.submit_to_registered_function(
        function_id=FUNC_ID,
        args=[{
            "inference_type": "single-image",
            "data_uri": "https://raw.githubusercontent.com/masalim2/sam3-service/refs/heads/main/examples/images/groceries.jpg",
            "single_image_prompt": "grocery bag",
        }],
    )
    print(f"Submitted single image inference {fut.task_id=}")
    print(fut.result(timeout=300))

print("Done")
