"""
Verification script for Markdown file indexing.
"""

import requests
import time
import json
import os
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
COLLECTION_NAME = "test_markdown_indexing"
TEST_DOCUMENT = "test_document.md"

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def create_test_markdown():
    """Create a dummy markdown file."""
    with open(TEST_DOCUMENT, "w", encoding="utf-8") as f:
        f.write("# Test Document\n\nThis is a test markdown document for GraphRAG indexing.\n")
    print(f"Created {TEST_DOCUMENT}")

def cleanup_test_file():
    """Remove the dummy markdown file."""
    if os.path.exists(TEST_DOCUMENT):
        os.remove(TEST_DOCUMENT)
        print(f"Removed {TEST_DOCUMENT}")

def test_markdown_indexing():
    try:
        create_test_markdown()
        
        # 1. Create Collection
        print_section("1. Create Collection")
        requests.delete(f"{BASE_URL}/api/collections/{COLLECTION_NAME}") # Ensure clean state
        response = requests.post(f"{BASE_URL}/api/collections", json={"name": COLLECTION_NAME})
        assert response.status_code == 201
        print("Collection created.")

        # 2. Upload Markdown Document
        print_section("2. Upload Markdown Document")
        with open(TEST_DOCUMENT, 'rb') as f:
            files = {'file': (TEST_DOCUMENT, f, 'text/markdown')}
            response = requests.post(
                f"{BASE_URL}/api/collections/{COLLECTION_NAME}/documents",
                files=files
            )
        print(f"Upload Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 201
        print("Markdown document uploaded.")

        # 3. Start Indexing
        print_section("3. Start Indexing")
        response = requests.post(f"{BASE_URL}/api/collections/{COLLECTION_NAME}/index")
        assert response.status_code == 202
        print("Indexing started.")

        # 4. Poll Status
        print_section("4. Poll Indexing Status")
        max_wait = 300
        start_time = time.time()
        while time.time() - start_time < max_wait:
            response = requests.get(f"{BASE_URL}/api/collections/{COLLECTION_NAME}/index")
            status_data = response.json()
            print(f"Status: {status_data['status']}, Progress: {status_data.get('progress', 0)}%")
            
            if status_data['status'] == 'completed':
                print("✅ Indexing completed successfully!")
                return True
            elif status_data['status'] == 'failed':
                print(f"❌ Indexing failed: {status_data.get('error')}")
                return False
            
            time.sleep(5)
        
        print("❌ Indexing timed out")
        return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Cleanup
        requests.delete(f"{BASE_URL}/api/collections/{COLLECTION_NAME}")
        cleanup_test_file()

if __name__ == "__main__":
    test_markdown_indexing()
