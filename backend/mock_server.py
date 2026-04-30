"""
Mock FastAPI server for frontend development and testing.

Run with:
    cd backend
    python mock_server.py
    # or
    uvicorn mock_server:app --port 8001 --reload

Then set NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8001 in frontend/.env.local
"""

import asyncio
import random
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="GraphRAG Mock Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Mock data store
# ---------------------------------------------------------------------------

MOCK_COLLECTIONS = [
    {
        "id": "col-movie-tt0443706",
        "name": "tt0443706",
        "description": "Movie dataset – Stranger Than Fiction (2006)",
        "created_at": "2026-02-22T10:00:00Z",
        "document_count": 3,
        "indexed": True,
    },
    {
        "id": "col-wiki-vietnam",
        "name": "vietnam-history",
        "description": "Vietnamese history articles from Wikipedia",
        "created_at": "2026-03-01T08:30:00Z",
        "document_count": 12,
        "indexed": True,
    },
    {
        "id": "col-tech-docs",
        "name": "tech-docs",
        "description": "Internal technical documentation",
        "created_at": "2026-03-15T14:00:00Z",
        "document_count": 0,
        "indexed": False,
    },
]

MOCK_DOCUMENTS: dict[str, list] = {
    "col-movie-tt0443706": [
        {"name": "tt0443706.txt", "size": 45231, "uploaded_at": "2026-02-22T10:05:00Z"},
        {"name": "subtitles.txt", "size": 32100, "uploaded_at": "2026-02-22T10:06:00Z"},
        {"name": "trivia.txt", "size": 8400, "uploaded_at": "2026-02-22T10:07:00Z"},
    ],
    "col-wiki-vietnam": [
        {"name": f"article_{i:02d}.txt", "size": random.randint(5000, 80000), "uploaded_at": "2026-03-01T08:31:00Z"}
        for i in range(1, 13)
    ],
    "col-tech-docs": [],
}

# In-memory sessions
SESSIONS: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Response templates
# ---------------------------------------------------------------------------

LOCAL_RESPONSE = """Harold Crick is the central character in **Stranger Than Fiction** (2006). He is a meticulous IRS auditor whose monotonous life is upended when he begins hearing a narrator's voice describing his actions in real time.

**Key characteristics:**
- Works at the IRS for 12 years
- Lives alone in a sparse apartment
- Counts every action obsessively (brushstrokes, steps to the bus stop)
- Plays guitar but never performs for others

**Relationship arc:** Harold falls in love with Ana Pascal, a baker he is auditing. Their relationship develops slowly due to his awkward social skills.

[Data: Entities (Harold Crick, Ana Pascal, Karen Eiffel); Relationships (Harold-Ana, Harold-IRS); Sources (1, 2, 4)]"""

GLOBAL_RESPONSE = """**Stranger Than Fiction** (2006) is a meta-fictional dramedy directed by Marc Forster, written by Zach Helm. The film explores themes of fate, free will, and the relationship between author and subject.

**Main themes:**
1. **Meta-fiction and self-awareness** — Harold literally lives inside a novel
2. **The value of a single life** — Karen Eiffel must decide whether to kill her protagonist
3. **Breaking routine** — Harold's arc from rigid auditor to someone who embraces life
4. **Love as catalyst** — Ana Pascal pushes Harold to grow

**Critical reception:** The film received 73% on Rotten Tomatoes and was praised for Will Ferrell's understated performance and the script's originality.

[Data: Reports (1, 3); Entities (Karen Eiffel, Harold Crick); Relationships (Author-Character)]"""

TOG_RESPONSE = """**Think-on-Graph reasoning trace:**

→ Starting entity: **Karen Eiffel**
  → Relation: [WROTE] → **Stranger Than Fiction (novel)**
    → Relation: [FEATURES] → **Harold Crick**
      → Relation: [EMPLOYED_BY] → **IRS**
        → Relation: [AUDITS] → **Ana Pascal**
          → Relation: [OWNS] → **Ana's Bakery**

**Conclusion:**
Karen Eiffel's authorship creates a causal chain: her narrative choices directly affect Harold Crick's IRS work, which connects him to Ana Pascal. The story's central tension arises because Karen — unaware Harold is real — plans to write his death.

Karen Eiffel has written 3 prior novels, all ending with the protagonist's death. Her writer's block resolves only when Harold accepts his fate willingly.

[Data: Entities (Karen Eiffel, Harold Crick, Ana Pascal); Relationships (WROTE, FEATURES, AUDITS, OWNS)]"""

DRIFT_RESPONSE = """**DRIFT analysis – Hypothetical: What if Harold had refused to accept his fate?**

Based on the knowledge graph, if Harold had refused:

**Immediate consequences:**
- Karen Eiffel's writer's block would persist indefinitely
- Harold's relationship with Ana would continue
- The IRS audit of Ana's bakery would conclude (in Harold's favor, given his character development)

**Systemic effects:**
- Eiffel's publisher (Penny Escher) would face financial pressure
- The 12 characters from Eiffel's previous novels who all died would retroactively highlight a pattern Harold breaks

**Professor Hilbert's assessment** (as modeled from dialogue): "The book would become the first in literary history where the protagonist actively averts the author's intent — a genuinely interesting outcome."

[Data: Entities (Harold Crick, Karen Eiffel, Penny Escher, Jules Hilbert); Relationships (DEFIES, WRITES, ADVISES)]"""

WEB_RESPONSE = """Based on current web information:

**Stranger Than Fiction (2006) – Recent Coverage**

The film has seen renewed interest following its addition to streaming platforms in early 2026. Critics on Letterboxd have praised its "quiet brilliance" and Will Ferrell's career-best dramatic performance.

**Box office:** Earned $53.6M worldwide against a $30M budget — a modest success that has grown into a cult classic.

**Legacy:** The screenplay by Zach Helm is frequently cited in screenwriting courses as an example of high-concept premises executed with emotional restraint.

**Cast updates:** Will Ferrell, Maggie Gyllenhaal, Dustin Hoffman, Emma Thompson, and Queen Latifah have all continued successful careers since the film's release.

[1] [2] [3]"""

WEB_SOURCES = [
    {"id": 1, "title": "Stranger Than Fiction – Letterboxd Reviews 2026", "url": "https://letterboxd.com/film/stranger-than-fiction/"},
    {"id": 2, "title": "Box Office Mojo – Stranger Than Fiction", "url": "https://www.boxofficemojo.com/title/tt0420223/"},
    {"id": 3, "title": "The Dissolve – Cult Classic Revisit", "url": "https://thedissolve.com/reviews/stranger-than-fiction/"},
]

CONTEXT_DATA = {
    "Entities": {
        "harold crick": {"name": "Harold Crick", "description": "IRS auditor and protagonist. Meticulous, lonely, later transforms after hearing a narrator's voice."},
        "karen eiffel": {"name": "Karen Eiffel", "description": "Reclusive author who unknowingly narrates Harold's real life. Known for killing her protagonists."},
        "ana pascal": {"name": "Ana Pascal", "description": "Baker and tax evader. Harold's love interest. Principled and unconventional."},
        "jules hilbert": {"name": "Jules Hilbert", "description": "Literary professor who helps Harold determine whether he is in a comedy or tragedy."},
        "penny escher": {"name": "Penny Escher", "description": "Karen Eiffel's publisher's assistant, tasked with helping Karen finish her book."},
    },
    "Relationships": {
        "harold-ana": {"name": "Harold ↔ Ana", "description": "Romantic relationship that develops during IRS audit."},
        "harold-irs": {"name": "Harold ↔ IRS", "description": "Employment relationship; Harold is a dedicated agent."},
        "author-character": {"name": "Karen → Harold", "description": "Karen unknowingly narrates Harold's real life."},
    },
    "Reports": {
        "1": {"name": "Community Report 1", "description": "Characters and their social connections in the film."},
        "3": {"name": "Community Report 3", "description": "Thematic analysis of meta-fictional elements."},
    },
    "Sources": {
        "1": {"name": "Script excerpt", "description": "Dialogue from Harold's first narration encounter."},
        "2": {"name": "Film synopsis", "description": "Full plot summary."},
        "4": {"name": "Director's notes", "description": "Marc Forster on casting Will Ferrell."},
    },
}

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    stream: bool = False
    community_level: int = 2
    response_type: str = "Multiple Paragraphs"
    dynamic_community_selection: bool = False
    max_depth: int | None = None
    beam_width: int | None = None
    show_exploration_paths: bool = False


class AgentSearchRequest(BaseModel):
    query: str
    stream: bool = True
    session_id: str | None = None
    conversation_history: list = Field(default_factory=list)
    conversation_summary: str | None = None
    web_search_enabled: bool = False


class WebSearchRequest(BaseModel):
    query: str
    stream: bool = True


class SummarizeRequest(BaseModel):
    conversation_history: list
    existing_summary: str | None = None


class ConversationTurn(BaseModel):
    role: str
    content: str
    rewritten_query: str | None = None
    method_used: str | None = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _simulated_delay(min_ms: int = 200, max_ms: int = 1200):
    """Simulate realistic backend latency."""
    return asyncio.sleep(random.uniform(min_ms / 1000, max_ms / 1000))


def _pick_response(method: str) -> str:
    responses = {
        "local": LOCAL_RESPONSE,
        "global": GLOBAL_RESPONSE,
        "tog": TOG_RESPONSE,
        "drift": DRIFT_RESPONSE,
        "web": WEB_RESPONSE,
    }
    return responses.get(method, LOCAL_RESPONSE)


def _route_query(query: str) -> str:
    """Simple keyword-based routing for mock."""
    q = query.lower()
    if any(w in q for w in ["overview", "summary", "theme", "overall", "about"]):
        return "global"
    if any(w in q for w in ["why", "how", "if", "would", "hypothetical", "suppose"]):
        return "drift"
    if any(w in q for w in ["path", "connection", "link", "hop", "relation"]):
        return "tog"
    return "local"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "mock-1.0.0"}


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

@app.get("/api/collections")
async def list_collections():
    await _simulated_delay(100, 300)
    return {"collections": MOCK_COLLECTIONS, "total": len(MOCK_COLLECTIONS)}


@app.post("/api/collections", status_code=status.HTTP_201_CREATED)
async def create_collection(body: dict):
    await _simulated_delay(200, 500)
    new_col = {
        "id": f"col-{body.get('name', 'new')}-{int(time.time())}",
        "name": body.get("name", "new-collection"),
        "description": body.get("description"),
        "created_at": _now(),
        "document_count": 0,
        "indexed": False,
    }
    MOCK_COLLECTIONS.append(new_col)
    return new_col


@app.get("/api/collections/{collection_id}")
async def get_collection(collection_id: str):
    await _simulated_delay(80, 200)
    col = next((c for c in MOCK_COLLECTIONS if c["id"] == collection_id), None)
    if not col:
        raise HTTPException(status_code=404, detail="Collection not found")
    return col


@app.delete("/api/collections/{collection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(collection_id: str):
    await _simulated_delay(100, 300)
    global MOCK_COLLECTIONS
    MOCK_COLLECTIONS = [c for c in MOCK_COLLECTIONS if c["id"] != collection_id]


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@app.get("/api/collections/{collection_id}/documents")
async def list_documents(collection_id: str):
    await _simulated_delay(100, 250)
    docs = MOCK_DOCUMENTS.get(collection_id, [])
    return {"documents": docs, "total": len(docs)}


@app.post("/api/collections/{collection_id}/documents")
async def upload_document(collection_id: str):
    await _simulated_delay(500, 1500)
    doc = {
        "name": f"uploaded_{int(time.time())}.txt",
        "size": random.randint(5000, 100000),
        "uploaded_at": _now(),
    }
    MOCK_DOCUMENTS.setdefault(collection_id, []).append(doc)
    return doc


@app.delete("/api/collections/{collection_id}/documents/{document_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(collection_id: str, document_name: str):
    await _simulated_delay(100, 300)
    if collection_id in MOCK_DOCUMENTS:
        MOCK_DOCUMENTS[collection_id] = [
            d for d in MOCK_DOCUMENTS[collection_id] if d["name"] != document_name
        ]


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

@app.post("/api/collections/{collection_id}/index")
async def start_indexing(collection_id: str):
    await _simulated_delay(300, 700)
    return {
        "collection_id": collection_id,
        "job_id": f"job-{int(time.time())}",
        "status": "running",
        "progress": 0.0,
        "message": "Indexing started",
        "started_at": _now(),
        "completed_at": None,
        "error": None,
    }


@app.get("/api/collections/{collection_id}/index")
async def get_index_status(collection_id: str):
    await _simulated_delay(100, 200)
    col = next((c for c in MOCK_COLLECTIONS if c["id"] == collection_id), None)
    if col and col["indexed"]:
        return {
            "collection_id": collection_id,
            "job_id": f"job-mock-{collection_id}",
            "status": "completed",
            "progress": 100.0,
            "message": "Indexing completed successfully",
            "attempt": 1,
            "max_attempts": 3,
            "started_at": "2026-02-22T10:10:00Z",
            "completed_at": "2026-02-22T10:22:00Z",
            "retry_at": None,
            "lease_owner_id": None,
            "heartbeat_at": None,
            "error": None,
        }
    return {
        "collection_id": collection_id,
        "job_id": f"job-mock-{collection_id}",
        "status": "pending",
        "progress": 0.0,
        "message": "Waiting to start",
        "attempt": 0,
        "max_attempts": 3,
        "started_at": None,
        "completed_at": None,
        "retry_at": None,
        "lease_owner_id": None,
        "heartbeat_at": None,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Search – specific methods
# ---------------------------------------------------------------------------

@app.post("/api/collections/{collection_id}/search/global")
async def global_search(collection_id: str, request: SearchRequest):
    await _simulated_delay(800, 1800)
    return {
        "query": request.query,
        "response": GLOBAL_RESPONSE,
        "context_data": CONTEXT_DATA,
        "method": "global",
    }


@app.post("/api/collections/{collection_id}/search/local")
async def local_search(collection_id: str, request: SearchRequest):
    await _simulated_delay(600, 1400)
    return {
        "query": request.query,
        "response": LOCAL_RESPONSE,
        "context_data": CONTEXT_DATA,
        "method": "local",
    }


@app.post("/api/collections/{collection_id}/search/tog")
async def tog_search(collection_id: str, request: SearchRequest):
    await _simulated_delay(1200, 2500)
    return {
        "query": request.query,
        "response": TOG_RESPONSE,
        "context_data": CONTEXT_DATA,
        "method": "tog",
    }


@app.post("/api/collections/{collection_id}/search/drift")
async def drift_search(collection_id: str, request: SearchRequest):
    await _simulated_delay(1000, 2000)
    return {
        "query": request.query,
        "response": DRIFT_RESPONSE,
        "context_data": CONTEXT_DATA,
        "method": "drift",
    }


@app.post("/api/collections/{collection_id}/search/web")
async def web_search(collection_id: str, request: WebSearchRequest):
    await _simulated_delay(700, 1500)
    return {
        "query": request.query,
        "response": WEB_RESPONSE,
        "sources": WEB_SOURCES,
        "method": "web",
    }


# ---------------------------------------------------------------------------
# Search – Agent (router)
# ---------------------------------------------------------------------------

@app.post("/api/collections/{collection_id}/search/agent")
async def agent_search(collection_id: str, request: AgentSearchRequest):
    method = _route_query(request.query)
    await _simulated_delay(1000, 2200)

    doc_response = _pick_response(method)
    reasoning_map = {
        "local": "Query mentions a specific character/entity — LOCAL search is most precise.",
        "global": "Query asks for overview/themes — GLOBAL search covers community reports.",
        "tog": "Query involves multi-hop relationships — TOG reasoning traces the graph.",
        "drift": "Query is hypothetical — DRIFT handles counterfactual reasoning.",
    }

    base = {
        "method_used": method,
        "router_reasoning": reasoning_map.get(method, "LOCAL selected as default."),
        "rewritten_query": f"{request.query} (rewritten for {method} search)",
        "response": doc_response,
        "sources": [],
        "context_data": CONTEXT_DATA,
        "session_id": request.session_id,
        "web_response": None,
        "web_sources": [],
    }

    if request.web_search_enabled:
        # Simulate parallel web fetch
        await _simulated_delay(300, 700)
        base["web_response"] = WEB_RESPONSE
        base["web_sources"] = WEB_SOURCES

    return base


# ---------------------------------------------------------------------------
# Conversation – summarize
# ---------------------------------------------------------------------------

@app.post("/api/collections/{collection_id}/search/agent/summarize")
async def summarize_conversation(collection_id: str, request: SummarizeRequest):
    await _simulated_delay(400, 900)
    turn_count = len(request.conversation_history)
    summary = (
        f"[Mock summary of {turn_count} conversation turns. "
        "The user asked questions about Harold Crick, Karen Eiffel, and the themes of Stranger Than Fiction. "
        "Key topics: character motivations, film themes, and hypothetical plot outcomes.]"
    )
    # Keep last 2 turns
    trimmed = request.conversation_history[-4:] if len(request.conversation_history) > 4 else request.conversation_history
    return {"summary": summary, "trimmed_history": trimmed}


# ---------------------------------------------------------------------------
# Conversation sessions (stub)
# ---------------------------------------------------------------------------

@app.post("/api/collections/{collection_id}/conversations")
async def create_session(collection_id: str):
    await _simulated_delay(100, 300)
    session_id = f"sess-{int(time.time())}"
    SESSIONS[session_id] = {"collection_id": collection_id, "turns": [], "created_at": _now()}
    return {
        "session_id": session_id,
        "collection_id": collection_id,
        "created_at": _now(),
    }


@app.get("/api/collections/{collection_id}/conversations/{session_id}")
async def get_session(collection_id: str, session_id: str):
    await _simulated_delay(80, 200)
    sess = SESSIONS.get(session_id, {})
    return {
        "session_id": session_id,
        "collection_id": collection_id,
        "summary": sess.get("summary"),
        "turn_count": len(sess.get("turns", [])),
        "user_turn_count": sum(1 for t in sess.get("turns", []) if t.get("role") == "user"),
        "created_at": sess.get("created_at", _now()),
        "updated_at": _now(),
        "recent_turns": sess.get("turns", [])[-6:],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mock_server:app", host="127.0.0.1", port=8001, reload=True)
