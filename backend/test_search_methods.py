"""
Quick test script for all 4 search methods.
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"
COLLECTION_NAME = "test_graphrag"


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_global_search():
    """Test global search."""
    print_section("1. Global Search")
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
        print(f"\nResponse:\n{result['response'][:800]}...")
        print("\n✅ Global search successful")
    else:
        print(f"❌ Global search failed: {response.text}")
    return response.status_code == 200


def test_local_search():
    """Test local search."""
    print_section("2. Local Search")
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
        print(f"\nResponse:\n{result['response'][:800]}...")
        print("\n✅ Local search successful")
    else:
        print(f"❌ Local search failed: {response.text}")
    return response.status_code == 200


def test_tog_search():
    """Test ToG search."""
    print_section("3. ToG Search")
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
        print(f"\nResponse:\n{result['response'][:800]}...")
        print("\n✅ ToG search successful")
    else:
        print(f"❌ ToG search failed: {response.text}")
    return response.status_code == 200


def test_drift_search():
    """Test DRIFT search."""
    print_section("4. DRIFT Search")
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
        print(f"\nResponse:\n{result['response'][:800]}...")
        print("\n✅ DRIFT search successful")
    else:
        print(f"❌ DRIFT search failed: {response.text}")
    return response.status_code == 200


def main():
    """Run all search tests."""
    print("\n" + "="*60)
    print("  Testing All 4 Search Methods")
    print("="*60)
    
    results = {
        "Global": test_global_search(),
        "Local": test_local_search(),
        "ToG": test_tog_search(),
        "DRIFT": test_drift_search()
    }
    
    print_section("Test Summary")
    for method, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{method} Search: {status}")
    
    total = sum(results.values())
    print(f"\n{total}/4 search methods passed")


if __name__ == "__main__":
    main()
