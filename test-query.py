#!/usr/bin/env python
"""Test GraphRAG query functionality"""

import asyncio
from graphrag.api import query

async def main():
    # Query the indexed knowledge graph
    results = await query(
        root="./test-project",
        method="local",
        query="Who is Scrooge and what are his main relationships?"
    )
    print("=== LOCAL SEARCH RESULT ===")
    print(results)
    
    # Try global search
    results = await query(
        root="./test-project",
        method="global", 
        query="What are the main themes in this story?"
    )
    print("\n=== GLOBAL SEARCH RESULT ===")
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
