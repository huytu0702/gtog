# ToG Chunk-Aware Context Flow

## Goal

Visualize the proposed ToG search flow after adding chunk context so the final reasoning context includes:

- chunks
- entities
- relationships

## Summary

Today, ToG search is initialized with entities and relationships, explores graph paths, and formats final reasoning context from graph evidence only.

The proposed flow adds `text_units` (chunks) into ToG, collects chunk evidence from explored entities and relationships, then merges all evidence into a unified reasoning context before generating the final answer.

## Proposed Flow

```mermaid
flowchart TD
    subgraph InputLayer[Input & Wiring]
        Q[User Query]
        F[Query Factory<br/>build ToG engine]
        I[ToGSearch inputs:<br/>Entities, Relationships, TextUnits]
        Q --> F --> I
    end

    subgraph ExplorationLayer[Exploration]
        S[Find starting entities]
        E[Iterative exploration<br/>relation scoring + entity scoring + beam prune]
        C[Collect chunk evidence<br/>from explored entities and relationships]
        I --> S --> E --> C
    end

    subgraph ReasoningLayer[Reasoning Context]
        R[Build reasoning context bundle]
        M[Format unified context:<br/>Chunks + Entities + Relationships]
        LLM[ToG reasoning LLM]
        C --> R --> M --> LLM
    end

    subgraph OutputLayer[Output]
        O[SearchResult]
        OT[context_text<br/>chunk-aware]
        OD[context_data<br/>paths + chunk evidence]
        LLM --> O
        O --> OT
        O --> OD
    end

    classDef input fill:#E8F0FE,stroke:#4A73B8,stroke-width:1px,color:#111;
    classDef explore fill:#EAF7EE,stroke:#2E8B57,stroke-width:1px,color:#111;
    classDef reason fill:#FFF5E6,stroke:#C97A00,stroke-width:1px,color:#111;
    classDef output fill:#F3E8FF,stroke:#7A3EB1,stroke-width:1px,color:#111;

    class Q,F,I input;
    class S,E,C explore;
    class R,M,LLM reason;
    class O,OT,OD output;
```

## Sequence View

```mermaid
sequenceDiagram
    participant U as User
    participant API as API / Eval caller
    participant FAC as query.factory.get_tog_search_engine
    participant TOG as ToGSearch
    participant EXP as GraphExplorer
    participant CTX as Chunk Context Builder
    participant REA as ToGReasoning

    U->>API: Submit query
    API->>FAC: Build ToG engine with entities, relationships, text_units
    FAC->>TOG: Initialize ToGSearch(..., text_units)
    TOG->>EXP: Find starting entities
    loop depth / beam exploration
        TOG->>EXP: Get relations for frontier nodes
        TOG->>EXP: Resolve entity + relationship details
    end
    TOG->>CTX: Gather chunks from explored entities/relationships
    CTX-->>TOG: Deduplicated chunk evidence
    TOG->>REA: Generate answer with chunks + entities + relationships
    REA-->>TOG: Answer + formatted context
    TOG-->>API: SearchResult(response, context_text, context_data)
    API-->>U: Final response
```

## Main Changes in the New Flow

1. **ToG input expands**
   - `ToGSearch` receives `text_units` in addition to `entities` and `relationships`.

2. **Chunk evidence is collected during/after exploration**
   - explored entities contribute chunk IDs via `Entity.text_unit_ids`
   - explored relationships contribute chunk IDs via `Relationship.text_unit_ids`

3. **Reasoning context becomes unified**
   - context formatter includes:
     - `=== CHUNKS ===`
     - `=== ENTITIES ===`
     - `=== RELATIONSHIPS ===`

4. **SearchResult becomes chunk-aware**
   - `context_text` contains chunk evidence
   - `context_data` can expose chunk metadata in addition to exploration paths

## Key File Touchpoints

- `graphrag/query/factory.py`
- `graphrag/query/structured_search/tog_search/search.py`
- `graphrag/query/structured_search/tog_search/exploration.py`
- `graphrag/query/structured_search/tog_search/reasoning.py`
- `graphrag/query/structured_search/tog_search/state.py`
- `graphrag/query/context_builder/source_context.py`
- `graphrag/query/input/retrieval/text_units.py`
- `graphrag/api/query.py`
- `graphrag/eval/runner.py`

## Suggested Evidence Assembly Logic

```text
query
  -> starting entities
  -> explored nodes
  -> collect related entities and relationships
  -> collect linked text units/chunks
  -> deduplicate chunks
  -> trim chunks by token budget
  -> format final context
  -> reasoning LLM
```

## Notes

- Reuse existing text-unit formatting utilities where possible.
- Keep chunk selection deterministic and token-budgeted.
- Preserve current ToG exploration behavior; only extend the context assembly path.
