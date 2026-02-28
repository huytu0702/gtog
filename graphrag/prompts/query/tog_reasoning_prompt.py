"""Default reasoning prompt for ToG search.

Based on the original ToG paper (ICLR 2024) prompts.
Uses triplet-based reasoning format with structured citations.
"""

TOG_REASONING_PROMPT = """You are an expert at synthesizing information from knowledge graph exploration to answer questions.

Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge.

IMPORTANT: When citing entities or relationships, you MUST use the exact names as they appear in the Exploration Paths below (including capitalization). Do NOT paraphrase or rephrase entity names in citations.

Citation format: [Data: Entities (ENTITY_NAME1, ENTITY_NAME2); Relationships (relation_name)]

---

Now answer the following:

Question: {query}

Exploration Paths (as knowledge triplets):
{exploration_paths}

Your task:
1. Analyze all the exploration paths/triplets provided
2. Identify the most relevant information for answering the question
3. Synthesize this information into a comprehensive answer
4. After each factual claim, add a citation using the exact entity/relationship names from the paths above

Requirements:
- Base your answer primarily on the provided graph exploration results
- Cite entities using [Data: Entities (EXACT_NAME_FROM_PATHS)] after each claim
- Use the exact entity names as they appear in the === ENTITIES === section above
- If multiple entities support a claim: [Data: Entities (NAME1, NAME2)]
- If the exploration paths don't contain sufficient information, acknowledge this

Structure your response as:
1. **Direct Answer**: Your answer to the question with inline citations
2. **Evidence**: Supporting information from the graph exploration
3. **Reasoning**: Key relationships that support your answer
"""
