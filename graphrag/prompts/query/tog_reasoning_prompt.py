"""Default reasoning prompt for ToG search.

Based on the original ToG paper (ICLR 2024) prompts.
Uses triplet-based reasoning format.
"""

TOG_REASONING_PROMPT = """You are an expert at synthesizing information from knowledge graph exploration to answer questions.

Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge.

## Examples:

### Example 1:
Question: Find the person who said "Taste cannot be controlled by law", what did this person die from?
Knowledge Triplets:
- ("Taste cannot be controlled by law", media_common.quotation.author, "Thomas Jefferson")

Answer: Based on the given knowledge triplets, the person who said "Taste cannot be controlled by law" is Thomas Jefferson. However, the triplets don't contain information about his cause of death. From my knowledge, Thomas Jefferson died from a combination of several conditions including uremia, pneumonia, and other ailments on July 4, 1826.

### Example 2:
Question: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets:
- ("Rift Valley Province", location.administrative_division.country, "Kenya")
- ("Kenya", location.country.currency_used, "Kenyan shilling")

Answer: Based on the knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer is **Kenyan shilling**.

### Example 3:
Question: The artist nominated for The Long Winter lived where?
Knowledge Triplets:
- ("The Long Winter", book.written_work.author, "Laura Ingalls Wilder")
- ("Laura Ingalls Wilder", people.person.places_lived, "De Smet")

Answer: Based on the knowledge triplets, the author of The Long Winter is Laura Ingalls Wilder, and she lived in De Smet. Therefore, the answer is **De Smet**.

---

Now answer the following:

Question: {query}

Exploration Paths (as knowledge triplets):
{exploration_paths}

Your task:
1. Analyze all the exploration paths/triplets provided
2. Identify the most relevant information for answering the question
3. Synthesize this information into a comprehensive answer
4. Explain your reasoning, citing specific entities and relationships

Requirements:
- Base your answer primarily on the provided graph exploration results
- Cite specific entities and relationships in your answer
- If the exploration paths don't contain sufficient information, acknowledge this and explain what's missing
- Provide a clear, well-structured response

Structure your response as:
1. **Direct Answer**: Your answer to the question
2. **Evidence**: Supporting information from the graph exploration
3. **Reasoning**: Key relationships that support your answer
"""
