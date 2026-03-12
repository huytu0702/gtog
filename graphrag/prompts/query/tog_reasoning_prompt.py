"""Default reasoning prompt for ToG search.

Based on the original ToG paper (ICLR 2024) prompts.
Uses triplet-based reasoning format with concise free-form answers.
"""

TOG_REASONING_PROMPT = """Given a question and the associated retrieved context from ToG exploration, you are asked to answer the question using the provided chunks, entities, relationships, and your knowledge in max three sentences.

Use the chunk evidence as the primary grounding source, and use the entities and relationships to connect and explain the answer when helpful. If the provided context is insufficient, say so briefly.

---

Now answer the following:

Question: {query}

Retrieved ToG Context (chunks, entities, and relationships):
{exploration_paths}
"""
