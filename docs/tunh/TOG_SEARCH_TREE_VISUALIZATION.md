# ToG Search - Tree Visualization

## Full Exploration Process

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                          DEPTH 0 - STARTING ENTITIES (3)                       ║
╚════════════════════════════════════════════════════════════════════════════════╝

                        Tesla                Elon Musk              SpaceX
                       (9.0)                 (8.5)                 (7.8)
                         •                     •                     •
                         │                     │                     │
```

---

## DEPTH 1 - One Hop Away (Relations from starting entities)

### Expansion Phase

```
From TESLA:
  ├─ Martin Eberhard (8.5) ✓ TOP-1
  ├─ Marc Tarpenning (8.0) ✓ TOP-2
  ├─ Kimbal Musk (6.2)
  └─ Drew Baglino (5.8)

From ELON MUSK:
  ├─ X Corp (7.2) ✓ TOP-3
  ├─ SolarCity (6.5)
  ├─ Zip2 (4.1)
  └─ PayPal (3.9)

From SPACEX:
  ├─ Gwynne Shotwell (6.8)
  ├─ Neuralink (5.5)
  ├─ OpenAI (4.2)
  └─ Blue Origin (2.1)
```

### After Pruning (Keep Top-3 Scores)

```
                        DEPTH 1
                    (Beam Width = 3)
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    Tesla→Martin      Tesla→Marc     Elon→X Corp
    (score:8.5)      (score:8.0)    (score:7.2)
          •               •              •
```

---

## DEPTH 2 - Two Hops Away

### Expansion Phase from Top-3 Paths

```
From MARTIN EBERHARD (came from Tesla):
  ├─ UC Berkeley (7.8) ✓
  ├─ JB Straubel (6.2)
  └─ Motors (4.5)

From MARC TARPENNING (came from Tesla):
  ├─ Stanford (7.5) ✓
  ├─ PayPal (5.8)
  └─ Compaq (3.2)

From X CORP (came from Elon Musk):
  ├─ Twitter (8.2) ✓
  ├─ Emerald Street (3.1)
  └─ Thiel (2.8)
```

### After Pruning (Keep Top-3)

```
                        DEPTH 2
                   (Beam Width = 3)
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    Martin→      Marc→          X Corp→
   UC Berkeley   Stanford       Twitter
   (score:7.8)  (score:7.5)   (score:8.2)
          •               •              •
```

---

## DEPTH 3 - Three Hops Away (Final Depth)

### Expansion Phase from Top-3 Paths at Depth 2

```
From UC BERKELEY (came from Tesla→Martin):
  ├─ Elon Musk (6.5) ✓
  ├─ Faculty (4.2)
  └─ PhD Program (3.1)

From STANFORD (came from Tesla→Marc):
  ├─ David Lee (5.2) ✓
  ├─ Bill Gates (4.8)
  └─ Founders (2.1)

From TWITTER (came from Elon Musk→X Corp):
  ├─ Jack Dorsey (6.5) ✓
  ├─ Evan Williams (3.2)
  └─ Noah Glass (1.8)
```

### Final State (Depth 3 - Last Level)

```
                        DEPTH 3
                   (Final Depth)
                          │
          ┌───────────────┼───────────────┐
          │               │               │
   UC Berkeley→     Stanford→         Twitter→
     Elon Musk      David Lee      Jack Dorsey
   (score:6.5)     (score:5.2)     (score:6.5)
          •               •              •
```

---

## Complete Tree Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     DEPTH 0 (Start)                         │
│                                                             │
│   Tesla(9.0)        Elon Musk(8.5)        SpaceX(7.8)     │
│      •                   •                    •            │
└──────┬───────────────────┬────────────────────┬────────────┘
       │                   │                    │
       │ Relations         │ Relations          │ Relations
       │                   │                    │
┌──────┴──────┬─────┬──────┴────┬─────┬────────┴────┬─────────┐
│             │     │           │     │            │         │
v             v     v           v     v            v         v
Martin(8.5)✓  Marc(8.0)✓  X Corp(7.2)✓   SolarCity Gwynne  Neuralink
(8.5)        (8.0)       (7.2)         (6.5)     (6.8)     (5.5)

PRUNING: Keep Top-3 ────────────────────────────────────────→

┌─────────────────────────────────────────────────────────────┐
│                    DEPTH 1 (1 Hop)                          │
│                Beam Width = 3 Paths                         │
│                                                             │
│  Tesla→Martin    Tesla→Marc      Elon Musk→X Corp         │
│  (8.5)✓          (8.0)✓          (7.2)✓                   │
│    •               •               •                        │
└─────┬─────────────┬──────────────┬──────────────────────────┘
      │ Relations   │ Relations    │ Relations
      │             │              │
   ┌──┴─┬────┐    ┌─┴──┬───┐    ┌─┴──┬─────┐
   │    │    │    │    │   │    │    │     │
   v    v    v    v    v   v    v    v     v
 UC(7.8)✓ JB SomeEnt Stanford(7.5)✓ PayPal Compaq  Twitter(8.2)✓ Emerald Thiel
 (7.8)  (6.2) (4.5)     (7.5)     (5.8) (3.2)    (8.2)      (3.1) (2.8)

PRUNING: Keep Top-3 ────────────────────────────────────────→

┌─────────────────────────────────────────────────────────────┐
│                    DEPTH 2 (2 Hops)                         │
│                Beam Width = 3 Paths                         │
│                                                             │
│  Tesla→Martin→      Tesla→Marc→        Elon→X Corp→       │
│  UC Berkeley        Stanford           Twitter              │
│  (7.8)✓             (7.5)✓             (8.2)✓             │
│    •                 •                  •                  │
└──────┬──────────────┬──────────────────┬──────────────────┘
       │ Relations    │ Relations        │ Relations
       │              │                  │
    ┌──┴─┬──┐      ┌──┴─┬──┐         ┌──┴──┬─────┐
    │    │  │      │    │  │         │     │     │
    v    v  v      v    v  v         v     v     v
 Elon(6.5) Faculty PhD  David(5.2) Gates  Founders Dorsey(6.5) Evans Noah
 (6.5)✓    (4.2)  (3.1) (5.2)✓    (4.8)  (2.1)    (6.5)✓   (3.2) (1.8)

PRUNING: Keep Top-3 (or Max Depth Reached) ──────────────────→

┌─────────────────────────────────────────────────────────────┐
│                    DEPTH 3 (3 Hops)                         │
│              Final Level - Answer Generation                │
│                                                             │
│  Tesla→Martin→      Tesla→Marc→        Elon→X Corp→       │
│  UC Berkeley→       Stanford→          Twitter→            │
│  Elon Musk          David Lee          Jack Dorsey         │
│  (6.5)              (5.2)              (6.5)               │
│    •                 •                  •                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### Beam Width = 3
- **At each depth**, keep only **top-3 best scoring paths**
- Prevents exponential explosion (3^3 = 27 → keep only 3)

### Scores
- **Higher score = More relevant to query**
- Scores computed by LLM using relation relevance
- Decrease as depth increases (longer paths = less certain)

### Pruning Strategy
1. Get all relations from current frontier nodes
2. Score each relation (1-10 scale)
3. Keep top-K relations (num_retain_entity = 5)
4. Create child nodes
5. **Prune to beam width (keep top-3 by score)**

### Early Termination
- After each depth, check: "Can we answer the question?"
- If YES (confidence high) → return answer immediately
- If NO → continue to next depth

---

## Example Query Flow

**Question:** "Who are the key people at Tesla?"

```
DEPTH 0:
  Starting: Tesla, Elon Musk, SpaceX
  ↓
  "Elon is CEO, founders are Martin & Marc"
  ↓ (Early termination check: YES, confident enough)
  
RETURN ANSWER:
  "The key people at Tesla are:
   - Elon Musk (CEO)
   - Martin Eberhard (co-founder)
   - Marc Tarpenning (co-founder)"
```

---

## Comparison: With vs Without Pruning

### Without Pruning (Exponential Growth)
```
Depth 0: 3 entities
Depth 1: 3 × 5 = 15 entities
Depth 2: 15 × 5 = 75 entities
Depth 3: 75 × 5 = 375 entities ❌ Explosion!
```

### With Pruning (Beam Width = 3)
```
Depth 0: 3 entities
Depth 1: 3 entities (pruned)
Depth 2: 3 entities (pruned)
Depth 3: 3 entities (pruned) ✓ Controlled!
```

---

## Summary

```
ToG Search = Controlled Multi-Hop Reasoning

INPUT: Query
  ↓
[SEMANTIC ENTITY LINKING]
  ↓
DEPTH 0: Start with top-3 entities
  ↓
LOOP (for each depth 0 → max_depth):
  ├─ Get relations from current frontier
  ├─ Score relations (LLM/Semantic/BM25)
  ├─ Create child nodes (top-5 relations)
  ├─ Prune to beam width (keep top-3)
  ├─ Check early termination
  └─ If YES → return answer, else continue
  ↓
[ANSWER GENERATION]
  Collect all explored paths → LLM synthesize → Final Answer
  ↓
OUTPUT: Answer with transparent path traces
```
