"""
Quick test to verify markdown file upload and indexing works.
"""
import requests
import time

BASE_URL = "http://127.0.0.1:8000"

# 1. Create a test markdown file
with open("test_upload.md", "w", encoding="utf-8") as f:
    f.write("""# GraphRAG Test Document

This is a test document for verifying markdown file support.

## Key Points

- GraphRAG supports knowledge graph extraction
- Multiple search methods are available
- The system uses OpenAI models for processing

## Conclusion

This document tests the markdown file indexing capability.
""")

print("✓ Created test_upload.md")

# 2. Delete existing test collection if it exists
try:
    requests.delete(f"{BASE_URL}/api/collections/test_md")
    print("✓ Cleaned up old collection")
except:
    pass

# 3. Create collection
print("\n1. Creating collection...")
response = requests.post(f"{BASE_URL}/api/collections", json={"name": "test_md"})
if response.status_code == 201:
    print(f"✓ Collection created: {response.json()['name']}")
else:
    print(f"✗ Failed to create collection: {response.status_code} - {response.text}")
    exit(1)

# 4. Upload markdown file
print("\n2. Uploading markdown file...")
with open("test_upload.md", "rb") as f:
    files = {"file": ("test_upload.md", f, "text/markdown")}
    response = requests.post(f"{BASE_URL}/api/collections/test_md/documents", files=files)

if response.status_code == 201:
    print(f"✓ File uploaded: {response.json()['name']} ({response.json()['size']} bytes)")
else:
    print(f"✗ Failed to upload file: {response.status_code} - {response.text}")
    exit(1)

# 5. Start indexing
print("\n3. Starting indexing...")
response = requests.post(f"{BASE_URL}/api/collections/test_md/index")
if response.status_code == 202:
    print(f"✓ Indexing started")
else:
    print(f"✗ Failed to start indexing: {response.status_code} - {response.text}")
    exit(1)

# 6. Poll indexing status
print("\n4. Monitoring indexing progress...")
for i in range(60):  # Max 2 minutes
    response = requests.get(f"{BASE_URL}/api/collections/test_md/index")
    if response.status_code == 200:
        status = response.json()
        print(f"   [{i*2}s] Status: {status['status']}, Progress: {status.get('progress', 0):.1f}%")
        
        if status['status'] == 'completed':
            print(f"\n✓ Indexing completed successfully!")
            print(f"   Started: {status.get('started_at')}")
            print(f"   Completed: {status.get('completed_at')}")
            break
        elif status['status'] == 'failed':
            print(f"\n✗ Indexing failed: {status.get('error')}")
            exit(1)
    
    time.sleep(2)
else:
    print(f"\n✗ Indexing timed out after 2 minutes")
    exit(1)

# 7. Cleanup
print("\n5. Cleaning up...")
import os
os.remove("test_upload.md")
requests.delete(f"{BASE_URL}/api/collections/test_md")
print("✓ Cleanup complete")

print("\n" + "="*60)
print("✓ ALL TESTS PASSED! Markdown file support is working.")
print("="*60)
