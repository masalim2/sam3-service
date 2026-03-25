from globus_compute_sdk import Client, Executor

gcc = Client()

# sophia-sam3 endpoint @ /home/openinference_svc/
ENDPOINT_ID = "25662117-d605-4f43-af9b-1b9ae390e4e2"
FUNC_ID = "26ff10a2-c944-4f1d-bf53-d3df96bf5baa"

with Executor(
    endpoint_id=ENDPOINT_ID,
    client=gcc,
    user_endpoint_config={
        "sam3_weights_dir": "/eagle/inference_service/sam3-service/weights/synaps-i/"
    },
) as gce:
    fut = gce.submit_to_registered_function(
        function_id=FUNC_ID,
        kwargs={
            "inference_type": "batch",
            "dataset_path": "/eagle/inference_service/sam3-service/examples/wds-test/shard-00000.tar",
        },
    )
    print(f"Submitted batch inference {fut.task_id=}")
    print("Awaiting results...")
    print(fut.result(timeout=300))

with Executor(endpoint_id=ENDPOINT_ID, client=gcc) as gce:
    fut = gce.submit_to_registered_function(
        function_id=FUNC_ID,
        kwargs=dict(
            image_uri="https://raw.githubusercontent.com/masalim2/sam3-service/refs/heads/main/examples/images/groceries.jpg",
            prompt="grocery bag",
        ),
    )
    print(f"Submitted single image inference {fut.task_id=}")
    print(fut.result(timeout=300))

print("Done")
