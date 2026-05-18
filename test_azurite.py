import sys
import traceback

CONN = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
    "QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"
)

try:
    from azure.storage.blob import BlobServiceClient
    from azure.storage.queue import QueueServiceClient

    print("--- Blob containers ---")
    blob_svc = BlobServiceClient.from_connection_string(CONN)
    for name in ["pipeline-input", "pipeline-logs"]:
        try:
            blob_svc.create_container(name)
            print(f"  created: {name}")
        except Exception as e:
            print(f"  {name}: {e}")

    print("--- Upload test file ---")
    container = blob_svc.get_container_client("pipeline-input")
    with open("test-input.md", "rb") as f:
        container.upload_blob("input/test-input.md", f, overwrite=True)
    print("  uploaded: input/test-input.md")

    print("--- Queue ---")
    queue_svc = QueueServiceClient.from_connection_string(CONN)
    try:
        queue_svc.create_queue("indexing-jobs")
        print("  created: indexing-jobs")
    except Exception as e:
        print(f"  indexing-jobs: {e}")

    print("--- Verify blobs ---")
    blobs = [b.name for b in container.list_blobs()]
    print(f"  blobs: {blobs}")

    print("DONE")

except Exception:
    traceback.print_exc()
    sys.exit(1)
