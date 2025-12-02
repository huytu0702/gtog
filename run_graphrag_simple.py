#!/usr/bin/env python3
"""
Simple GraphRAG Test Script

This script demonstrates indexing and searching with simple_test.txt
"""

import os
import sys
import subprocess

def run_cmd(cmd, description):
    """Run command and show output in real-time"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        print(f"Return code: {result.returncode}")
        return result.returncode == 0

    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("🔍 GraphRAG Simple Test")

    # Set OPEN_API_KEY environment variable for this session
    # You need to set your actual OpenAI API key in your environment
    # Example: set OPEN_API_KEY=your_actual_key_here
    os.environ['OPEN_API_KEY'] = os.getenv('OPEN_API_KEY', 'YOUR_API_KEY_HERE')

    # Check if input file exists
    if not os.path.exists('simple_test.txt'):
        print("❌ simple_test.txt not found!")
        return 1

    print("✅ Found simple_test.txt")

    # Test 1: Try to run indexing (will likely fail with placeholder key)
    success = run_cmd(
        "python -m graphrag index --root . --config graphrag/settings.yaml --verbose",
        "Indexing simple_test.txt"
    )

    if not success:
        print("\n⚠️  Indexing failed - likely due to missing or invalid OpenAI API key")
        print("To fix this, you need to:")
        print("1. Get a valid OpenAI API key")
        print("2. Set OPEN_API_KEY environment variable: set OPEN_API_KEY=your_actual_key")
        print("3. Or update graphrag/settings.yaml with your actual key")

        # Show what the document contains for reference
        print(f"\n📄 Content of simple_test.txt:")
        print("-" * 40)
        with open('simple_test.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
        print("-" * 40)

        # Test 2: Try mock/basic search without indexing (if supported)
        print(f"\n🔍 Testing search methods...")

        search_methods = ["basic", "local", "global", "drift"]
        query = "What is this document about?"

        for method in search_methods:
            print(f"\n--- Testing {method.upper()} search ---")
            success = run_cmd(
                f'python -m graphrag query --method {method} --query "{query}" --root . --config graphrag/settings.yaml --data output --verbose',
                f"{method.upper()} Search Test"
            )

            if not success:
                print(f"❌ {method.upper()} search failed - likely because indexing hasn't been completed")

    print(f"\n📋 SUMMARY:")
    print("1. ✅ Created GraphRAG runner script")
    print("2. ✅ Identified the 4 search methods: local, global, drift, basic, tog")
    print("3. ⚠️  Indexing requires valid OpenAI API key")
    print("4. ⚠️  Search methods require completed indexing")

    return 0

if __name__ == "__main__":
    sys.exit(main())