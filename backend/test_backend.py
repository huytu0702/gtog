"""
End-to-end test script for GraphRAG FastAPI backend.

This script tests the complete workflow:
1. Create a collection
2. Upload a document
3. Start indexing
4. Poll for indexing completion
5. Perform all 4 search methods
6. Cleanup
"""

import requests
import time
import json
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
COLLECTION_NAME = "test_graphrag"
TEST_DOCUMENT = "test_document.txt"


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_health_check():
    """Test the health check endpoint."""
    print_section("1. Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    print("✅ Health check passed")


def test_create_collection():
    """Create a test collection."""
    print_section("2. Create Collection")
    data = {
        "name": COLLECTION_NAME,
        "description": "Test collection for GraphRAG backend"
    }
    response = requests.post(f"{BASE_URL}/api/collections", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 201
    print(f"✅ Collection '{COLLECTION_NAME}' created")


def test_list_collections():
    """List all collections."""
    print_section("3. List Collections")
    response = requests.get(f"{BASE_URL}/api/collections")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Total collections: {result['total']}")
    for col in result['collections']:
        print(f"  - {col['name']} (documents: {col['document_count']}, indexed: {col['indexed']})")
    assert response.status_code == 200
    print("✅ Collections listed")


def test_upload_document():
    """Upload a test document."""
    print_section("4. Upload Document")
    
    # Check if test document exists
    test_file = Path(TEST_DOCUMENT)
    if not test_file.exists():
        print(f"❌ Test document '{TEST_DOCUMENT}' not found")
        return False
    
    with open(test_file, 'rb') as f:
        files = {'file': (TEST_DOCUMENT, f, 'text/plain')}
        response = requests.post(
            f"{BASE_URL}/api/collections/{COLLECTION_NAME}/documents",
            files=files
        )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 201
    print(f"✅ Document '{TEST_DOCUMENT}' uploaded")
    return True


def test_list_documents():
    """List documents in the collection."""
    print_section("5. List Documents")
    response = requests.get(f"{BASE_URL}/api/collections/{COLLECTION_NAME}/documents")
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Total documents: {result['total']}")
    for doc in result['documents']:
        print(f"  - {doc['name']} ({doc['size']} bytes)")
    assert response.status_code == 200
    print("✅ Documents listed")


def test_start_indexing():
    """Start the indexing process."""
    print_section("6. Start Indexing")
    response = requests.post(f"{BASE_URL}/api/collections/{COLLECTION_NAME}/index")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 202
    print("✅ Indexing started")


def test_poll_indexing_status():
    """Poll indexing status until completion."""
    print_section("7. Poll Indexing Status")
    
    max_wait = 300  # 5 minutes max
    poll_interval = 5  # Check every 5 seconds
    elapsed = 0
    
    while elapsed < max_wait:
        response = requests.get(f"{BASE_URL}/api/collections/{COLLECTION_NAME}/index")
        if response.status_code != 200:
            print(f"❌ Polling failed: {response.status_code} - {response.text}")
            return False
            
        status_data = response.json()
        if 'status' not in status_data:
            print(f"❌ Invalid response: {status_data}")
            return False
            
        print(f"[{elapsed}s] Status: {status_data['status']}, Progress: {status_data.get('progress', 0):.1f}%")
        if status_data.get('message'):
            print(f"       Message: {status_data['message']}")
        
        if status_data['status'] == 'completed':
            print("✅ Indexing completed successfully!")
            return True
        elif status_data['status'] == 'failed':
            print(f"❌ Indexing failed: {status_data.get('error', 'Unknown error')}")
            return False
        
        time.sleep(poll_interval)
        elapsed += poll_interval
    
    print("❌ Indexing timed out")
    return False


def test_global_search():
    """Test global search."""
    print_section("8. Global Search")
    data = {
        "query": "What are the main topics covered in the documents?",
        "response_type": "Multiple Paragraphs"
    }
    response = requests.post(
        f"{BASE_URL}/api/collections/{COLLECTION_NAME}/search/global",
        json=data
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuery: {result['query']}")
        print(f"Method: {result['method']}")
        print(f"\nResponse:\n{result['response'][:500]}...")  # First 500 chars
        print("✅ Global search successful")
    else:
        print(f"❌ Global search failed: {response.text}")


def test_local_search():
    """Test local search."""
    print_section("9. Local Search")
    data = {
        "query": "What is ToG search and how does it work?",
        "community_level": 2,
        "response_type": "Multiple Paragraphs"
    }
    response = requests.post(
        f"{BASE_URL}/api/collections/{COLLECTION_NAME}/search/local",
        json=data
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuery: {result['query']}")
        print(f"Method: {result['method']}")
        print(f"\nResponse:\n{result['response'][:500]}...")
        print("✅ Local search successful")
    else:
        print(f"❌ Local search failed: {response.text}")


def test_tog_search():
    """Test ToG search."""
    print_section("10. ToG Search")
    data = {
        "query": "How are the different search methods related to each other?"
    }
    response = requests.post(
        f"{BASE_URL}/api/collections/{COLLECTION_NAME}/search/tog",
        json=data
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuery: {result['query']}")
        print(f"Method: {result['method']}")
        print(f"\nResponse:\n{result['response'][:500]}...")
        print("✅ ToG search successful")
    else:
        print(f"❌ ToG search failed: {response.text}")


def test_drift_search():
    """Test DRIFT search."""
    print_section("11. DRIFT Search")
    data = {
        "query": "What are the relationships between knowledge graphs and search methods?",
        "community_level": 2,
        "response_type": "Multiple Paragraphs"
    }
    response = requests.post(
        f"{BASE_URL}/api/collections/{COLLECTION_NAME}/search/drift",
        json=data
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuery: {result['query']}")
        print(f"Method: {result['method']}")
        print(f"\nResponse:\n{result['response'][:500]}...")
        print("✅ DRIFT search successful")
    else:
        print(f"❌ DRIFT search failed: {response.text}")


def test_cleanup():
    """Clean up test collection."""
    print_section("12. Cleanup")
    response = requests.delete(f"{BASE_URL}/api/collections/{COLLECTION_NAME}")
    print(f"Status: {response.status_code}")
    assert response.status_code == 204
    print(f"✅ Collection '{COLLECTION_NAME}' deleted")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  GraphRAG FastAPI Backend - End-to-End Test")
    print("="*60)
    
    try:
        # Basic tests
        test_health_check()
        test_create_collection()
        test_list_collections()
        
        # Document upload
        if not test_upload_document():
            print("\n⚠️  Skipping remaining tests due to missing test document")
            return
        
        test_list_documents()
        
        # Indexing
        test_start_indexing()
        indexing_success = test_poll_indexing_status()
        
        if not indexing_success:
            print("\n⚠️  Skipping search tests due to indexing failure")
            test_cleanup()
            return
        
        # Search tests
        test_global_search()
        test_local_search()
        test_tog_search()
        test_drift_search()
        
        # Cleanup
        test_cleanup()
        
        print_section("Test Summary")
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to cleanup
        try:
            test_cleanup()
        except:
            pass


if __name__ == "__main__":
    main()
