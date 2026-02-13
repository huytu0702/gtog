#!/usr/bin/env python3
"""Deterministic smoke test script for Cosmos-backed GraphRAG API."""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


def create_collection(api_base_url: str, collection_name: str) -> bool:
    """Create a new collection."""
    url = f"{api_base_url}/api/collections"
    payload = {
        "name": collection_name,
        "description": f"Smoke test collection created at {datetime.now().isoformat()}",
    }
    
    print(f"  Creating collection '{collection_name}'...")
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 201:
            print(f"    Created successfully")
            return True
        elif response.status_code == 409:
            print(f"    Collection already exists (OK)")
            return True
        else:
            print(f"    Failed: HTTP {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"    Failed: {e}")
        return False


def upload_document(api_base_url: str, collection_name: str, doc_path: Path) -> bool:
    """Upload a document to a collection."""
    url = f"{api_base_url}/api/collections/{collection_name}/documents"
    
    print(f"  Uploading document '{doc_path.name}'...")
    try:
        with open(doc_path, "rb") as f:
            files = {"file": (doc_path.name, f, "text/plain")}
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 201:
            print(f"    Uploaded successfully")
            return True
        else:
            print(f"    Failed: HTTP {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"    Failed: {e}")
        return False


def start_indexing(api_base_url: str, collection_name: str) -> bool:
    """Start indexing for a collection."""
    url = f"{api_base_url}/api/collections/{collection_name}/index"
    
    print(f"  Starting indexing...")
    try:
        response = requests.post(url, timeout=30)
        if response.status_code == 202:
            print(f"    Indexing started")
            return True
        else:
            print(f"    Failed: HTTP {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"    Failed: {e}")
        return False


def wait_for_indexing(
    api_base_url: str,
    collection_name: str,
    timeout: int = 300,
    interval: int = 5,
) -> bool:
    """Poll indexing status until completed or timeout."""
    url = f"{api_base_url}/api/collections/{collection_name}/status"
    start_time = time.time()
    
    print(f"  Waiting for indexing to complete (timeout: {timeout}s)...")
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                
                if status == "indexed":
                    print(f"    Indexing completed successfully")
                    return True
                elif status == "error":
                    print(f"    Indexing failed with error")
                    return False
                else:
                    elapsed = int(time.time() - start_time)
                    print(f"    Status: {status} ({elapsed}s elapsed)")
            else:
                print(f"    Error checking status: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"    Error checking status: {e}")
        
        time.sleep(interval)
    
    print(f"    Timeout: Indexing did not complete within {timeout} seconds")
    return False


def execute_query(
    api_base_url: str,
    collection_name: str,
    query: str,
    method: str = "local",
) -> bool:
    """Execute a query against a collection."""
    url = f"{api_base_url}/api/search"
    payload = {
        "query": query,
        "collection_name": collection_name,
        "method": method,
    }
    
    print(f"  Executing {method} query...")
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            print(f"    Query completed successfully")
            print(f"    Response preview: {str(data)[:200]}...")
            return True
        else:
            print(f"    Failed: HTTP {response.status_code} - {response.text[:200]}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"    Failed: {e}")
        return False


def delete_collection(api_base_url: str, collection_name: str) -> bool:
    """Delete a collection."""
    url = f"{api_base_url}/api/collections/{collection_name}"
    
    print(f"  Deleting collection '{collection_name}'...")
    try:
        response = requests.delete(url, timeout=30)
        if response.status_code in (200, 204):
            print(f"    Deleted successfully")
            return True
        else:
            print(f"    Failed: HTTP {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"    Failed: {e}")
        return False


def create_sample_document() -> Path:
    """Create a sample document for testing."""
    sample_text = """
GraphRAG is a research project from Microsoft that enables knowledge graph-based
retrieval-augmented generation. It combines graph databases with large language
models to provide more accurate and contextual answers.

The system extracts entities and relationships from documents, builds a knowledge
graph, and uses graph traversal algorithms to find relevant information for queries.

Key features include:
- Entity extraction from unstructured text
- Relationship building between entities
- Community detection using graph algorithms
- Multiple search methods (global, local, ToG, DRIFT)
"""
    
    doc_path = Path("/tmp/smoke_test_sample.txt")
    doc_path.write_text(sample_text)
    return doc_path


def run_smoke_test(
    api_base_url: str,
    cleanup: bool = False,
    indexing_timeout: int = 300,
) -> dict:
    """Run the complete smoke test flow."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collection_name = f"smoke-{timestamp}"
    
    results = {
        "collection_name": collection_name,
        "steps": {},
        "passed": False,
    }
    
    print(f"\n{'=' * 60}")
    print(f"Cosmos API Smoke Test")
    print(f"Collection: {collection_name}")
    print(f"{'=' * 60}\n")
    
    # Step 1: Create collection
    print("Step 1: Create Collection")
    results["steps"]["create_collection"] = create_collection(api_base_url, collection_name)
    print()
    
    if not results["steps"]["create_collection"]:
        print("FAILED: Could not create collection\n")
        return results
    
    # Step 2: Upload document
    print("Step 2: Upload Document")
    doc_path = create_sample_document()
    results["steps"]["upload_document"] = upload_document(api_base_url, collection_name, doc_path)
    print()
    
    if not results["steps"]["upload_document"]:
        print("FAILED: Could not upload document\n")
        return results
    
    # Step 3: Start indexing
    print("Step 3: Start Indexing")
    results["steps"]["start_indexing"] = start_indexing(api_base_url, collection_name)
    print()
    
    if not results["steps"]["start_indexing"]:
        print("FAILED: Could not start indexing\n")
        return results
    
    # Step 4: Wait for indexing
    print("Step 4: Wait for Indexing")
    results["steps"]["wait_indexing"] = wait_for_indexing(
        api_base_url,
        collection_name,
        timeout=indexing_timeout,
    )
    print()
    
    if not results["steps"]["wait_indexing"]:
        print("FAILED: Indexing did not complete\n")
        return results
    
    # Step 5: Execute query
    print("Step 5: Execute Query")
    results["steps"]["execute_query"] = execute_query(
        api_base_url,
        collection_name,
        query="What is GraphRAG and what are its key features?",
        method="local",
    )
    print()
    
    # Determine overall pass/fail
    results["passed"] = all(results["steps"].values())
    
    # Cleanup if requested
    if cleanup:
        print("Cleanup: Delete Collection")
        delete_collection(api_base_url, collection_name)
        print()
    
    return results


def print_summary(results: dict):
    """Print test summary."""
    print(f"{'=' * 60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"Collection: {results['collection_name']}")
    print()
    
    for step_name, passed in results["steps"].items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {step_name}: {status}")
    
    print()
    if results["passed"]:
        print("OVERALL: PASSED ✓")
    else:
        print("OVERALL: FAILED ✗")
    print(f"{'=' * 60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Deterministic smoke test for Cosmos-backed GraphRAG API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run smoke test without cleanup
  %(prog)s --cleanup          # Run smoke test and cleanup after
  %(prog)s --api-base-url http://host:8000  # Use different API endpoint
        """
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:8000",
        help="Base URL for backend API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete test collection after test completes",
    )
    parser.add_argument(
        "--indexing-timeout",
        type=int,
        default=300,
        help="Timeout for indexing in seconds (default: 300)",
    )
    
    args = parser.parse_args()
    
    # Run smoke test
    results = run_smoke_test(
        api_base_url=args.api_base_url,
        cleanup=args.cleanup,
        indexing_timeout=args.indexing_timeout,
    )
    
    # Print summary
    print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
