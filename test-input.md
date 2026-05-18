# GraphRAG Test Document

## Overview

GraphRAG is a system developed by Microsoft Research that combines knowledge graph construction with retrieval-augmented generation.

## Key Components

**Entity Extraction**: The system identifies entities such as people, organizations, and locations from unstructured text.

**Relationship Building**: Semantic relationships between entities are created to form a knowledge graph.

**Community Detection**: Entities are grouped into communities using the Leiden algorithm.

## Query Methods

- **Global Search**: Broad overviews using map-reduce over community reports.
- **Local Search**: Entity-centric retrieval with direct evidence.
- **ToG Search**: Iterative graph exploration with beam search and deep reasoning.

## Benefits

GraphRAG enables more accurate and transparent answers by grounding responses in structured knowledge rather than raw text similarity alone.
