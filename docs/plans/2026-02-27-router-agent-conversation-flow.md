# Router Agent Conversation Flow (Post-Implementation)

> Illustrates the full flow from user query to final answer after implementing the conversation history + hierarchical summarization plan.

---

## 1. Happy Path — First Query (No History)

```mermaid
sequenceDiagram
    actor User
    participant FE as Frontend
    participant EP as POST /agent
    participant RA as RouterAgent
    participant LLM as LLM (router)
    participant QS as QueryService
    participant GR as GraphRAG / Web

    User->>FE: "Tell me about Inception"
    FE->>EP: POST /agent\n{ query, history: [], summary: null }

    EP->>RA: route(query, ctx,\nhistory=[], summary=None)
    RA->>RA: _format_history_block([], None)\n→ "" (empty)
    RA->>LLM: Single prompt:\n[methods + collection + query]
    LLM-->>RA: { rewritten_query: "Tell me about Inception",\nmethod: "local", confidence: 0.91 }
    RA-->>EP: RouteDecision(method="local",\nrewritten_query="Tell me about Inception")

    EP->>QS: local_search(collection_id,\nquery="Tell me about Inception")
    QS->>GR: GraphRAG local search
    GR-->>QS: result text
    QS-->>EP: SearchResponse

    EP-->>FE: AgentSearchResponse {\n  method_used: "local",\n  rewritten_query: "Tell me about Inception",\n  response: "Inception is a 2010 film...",\n  router_reasoning: "...",\n  sources: []\n}
    FE->>FE: Append turn to conversation_history\n(stores rewritten_query + method_used)
    FE-->>User: "Inception is a 2010 film..."
```

---

## 2. Happy Path — Follow-Up Query (With History, No Summary Yet)

```mermaid
sequenceDiagram
    actor User
    participant FE as Frontend
    participant EP as POST /agent
    participant RA as RouterAgent
    participant LLM as LLM (router)
    participant QS as QueryService
    participant GR as GraphRAG / Web

    User->>FE: "Who directed it?"
    FE->>EP: POST /agent {\n  query: "Who directed it?",\n  history: [turn1_user, turn1_assistant],\n  summary: null\n}

    EP->>RA: route(query, ctx,\nhistory=[turn1_user, turn1_assistant],\nsummary=None)
    RA->>RA: _format_history_block(history, None)\n→ "[User] Tell me about Inception → method: local\n   [Assistant] Inception is a 2010 film..."
    RA->>LLM: Single prompt:\n[methods + collection +\nConversation history +\nCurrent query: "Who directed it?"]
    LLM-->>RA: { rewritten_query: "Who directed Inception?",\nmethod: "local", confidence: 0.93 }
    RA-->>EP: RouteDecision(method="local",\nrewritten_query="Who directed Inception?")

    EP->>QS: local_search(collection_id,\nquery="Who directed Inception?")
    QS->>GR: GraphRAG local search
    GR-->>QS: result text
    QS-->>EP: SearchResponse

    EP-->>FE: AgentSearchResponse {\n  method_used: "local",\n  rewritten_query: "Who directed Inception?",\n  response: "Christopher Nolan directed...",\n  sources: []\n}
    FE->>FE: Append turn 2 to conversation_history
    FE-->>User: "Christopher Nolan directed..."
```

---

## 3. Summarization Trigger — History Exceeds Threshold

> Frontend triggers this when `conversation_history` reaches the threshold (e.g. 6 user turns).

```mermaid
sequenceDiagram
    actor User
    participant FE as Frontend
    participant SE as POST /agent/summarize
    participant SS as SummarizationService
    participant LLM as LLM (summarizer)

    Note over FE: After 6 user turns,\nfrontend triggers summarization

    FE->>SE: POST /agent/summarize {\n  conversation_history: [turns 1–6],\n  existing_summary: null\n}

    SE->>SS: summarize(history, existing_summary=None)
    SS->>SS: _format_turns(history)\n→ "User: Tell me about Inception\nAssistant: ...\nUser: Who directed it?\n..."
    SS->>LLM: Summarization prompt\n[conversation_text]
    LLM-->>SS: "User explored Inception (2010),\nasked about director Christopher Nolan,\ncast, and filming locations."
    SS-->>SE: summary string

    SE->>SS: get_trimmed_history(history, keep_turns=3)
    SS-->>SE: [last 3 user turns + assistant pairs]

    SE-->>FE: SummarizeResponse {\n  summary: "User explored Inception...",\n  trimmed_history: [turns 4–6]\n}

    FE->>FE: Replace conversation_history with trimmed_history\nStore summary as conversation_summary
```

---

## 4. Query After Summarization (Summary + Recent Turns)

```mermaid
sequenceDiagram
    actor User
    participant FE as Frontend
    participant EP as POST /agent
    participant RA as RouterAgent
    participant LLM as LLM (router)
    participant QS as QueryService
    participant GR as GraphRAG / Web

    User->>FE: "What is his relationship to Emma Thomas?"
    FE->>EP: POST /agent {\n  query: "What is his relationship to Emma Thomas?",\n  summary: "User explored Inception, asked about\n            Christopher Nolan...",\n  history: [recent 3 turns]\n}

    EP->>RA: route(query, ctx,\nhistory=[recent 3 turns],\nsummary="User explored Inception...")
    RA->>RA: _format_history_block(history, summary)\n→ "Past conversation summary:\nUser explored Inception...\n\nRecent conversation:\n[User] ... → method: local\n[Assistant] ..."
    RA->>LLM: Single prompt:\n[methods + collection +\nSummary + Recent turns +\nCurrent query]
    LLM-->>RA: { rewritten_query: "What is Christopher Nolan's\nrelationship to Emma Thomas?",\nmethod: "tog", confidence: 0.89 }

    Note over RA,LLM: Router switches method to TOG\nbecause query is about a relationship

    RA-->>EP: RouteDecision(method="tog",\nrewritten_query="What is Christopher Nolan's\nrelationship to Emma Thomas?")

    EP->>QS: tog_search(collection_id,\nquery="What is Christopher Nolan's\nrelationship to Emma Thomas?")
    QS->>GR: GraphRAG ToG search\n(multi-hop entity traversal)
    GR-->>QS: result text
    QS-->>EP: SearchResponse

    EP-->>FE: AgentSearchResponse {\n  method_used: "tog",\n  rewritten_query: "What is Christopher Nolan's...",\n  response: "Christopher Nolan and Emma Thomas\nare married and long-time collaborators...",\n  sources: []\n}
    FE->>FE: Append to trimmed history
    FE-->>User: "Christopher Nolan and Emma Thomas..."
```

---

## 5. Streaming Variant Flow

```mermaid
sequenceDiagram
    actor User
    participant FE as Frontend
    participant EP as POST /agent/stream
    participant RA as RouterAgent
    participant LLM as LLM (router)
    participant QS as QueryService

    User->>FE: Query + history + summary
    FE->>EP: POST /agent/stream (SSE)

    EP-->>FE: event: status\n{ step: "routing", message: "Analyzing query..." }

    EP->>RA: route(query, ctx, history, summary)
    RA->>LLM: Single routing prompt
    LLM-->>RA: RouteDecision + rewritten_query
    RA-->>EP: RouteDecision

    EP-->>FE: event: status\n{ step: "routed", method: "local",\n  message: "Using LOCAL search" }
    EP-->>FE: event: status\n{ step: "searching",\n  message: "Searching knowledge graph..." }

    EP->>QS: <method>_search(collection_id,\nrewritten_query)
    QS-->>EP: response text (chunked)

    loop For each 50-char chunk
        EP-->>FE: event: content\n{ delta: "chunk text..." }
    end

    EP-->>FE: event: done\n{ method_used, rewritten_query,\n  sources, router_reasoning }

    FE->>FE: Append completed turn to history
    FE-->>User: Streamed response rendered
```

---

## 6. Token Budget — How Summarization Keeps Prompts Bounded

```mermaid
graph LR
    subgraph "Without Summarization (grows unbounded)"
        A1["Turn 1 ~200 tok"]
        A2["Turn 2 ~200 tok"]
        A3["Turn 3 ~200 tok"]
        A4["Turn N ~200 tok"]
        A1 --> A2 --> A3 --> A4
        A4 --> TOTAL1["Total: N × 200 tokens ⚠️"]
    end

    subgraph "With Hierarchical Summarization (bounded)"
        B1["Summary\n~150 tokens"]
        B2["Recent turn 1\n~150 tokens"]
        B3["Recent turn 2\n~150 tokens"]
        B4["Recent turn 3\n~150 tokens"]
        B5["Router base prompt\n~200 tokens"]
        B1 & B2 & B3 & B4 & B5 --> TOTAL2["Total: ~800 tokens ✅\n(always fixed)"]
    end
```

---

## 7. Component Map

```mermaid
graph TD
    subgraph Frontend["Frontend (owns session state)"]
        FE_REQ["AgentSearchRequest\n· query\n· conversation_history (recent 3 turns)\n· conversation_summary"]
        FE_STORE["Session Store\n· append rewritten_query + method_used per turn\n· trigger /summarize at threshold"]
    end

    subgraph Backend["Backend (stateless)"]
        EP_AGENT["POST /agent\nagent_search()"]
        EP_STREAM["POST /agent/stream\nagent_search_stream()"]
        EP_SUM["POST /agent/summarize\nsummarize_conversation()"]

        RA["RouterAgent\n· _format_history_block(history, summary)\n· route(query, ctx, history, summary)\n→ RouteDecision { method, confidence,\n   reasoning, rewritten_query }"]

        SS["SummarizationService\n· summarize(history, existing_summary)\n· get_trimmed_history(history, keep=3)"]

        QS["QueryService\n· local_search(id, query)\n· global_search(id, query)\n· tog_search(id, query)\n· drift_search(id, query)"]

        WS["WebSearchService\n· search(query)\n· search_streaming(query)"]
    end

    subgraph LLMs["LLM Calls"]
        LLM_R["Router LLM\nreturns: rewritten_query + method"]
        LLM_S["Summarizer LLM\nreturns: 2-4 sentence summary"]
    end

    FE_REQ --> EP_AGENT & EP_STREAM
    FE_STORE --> EP_SUM

    EP_AGENT --> RA
    EP_STREAM --> RA
    EP_SUM --> SS

    RA --> LLM_R
    SS --> LLM_S

    EP_AGENT --> QS & WS
    EP_STREAM --> QS & WS

    EP_AGENT --> FE_STORE
    EP_SUM --> FE_STORE
```
