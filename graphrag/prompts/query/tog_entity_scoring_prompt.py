"""Default entity scoring prompt for ToG search.

Based on the original ToG paper (ICLR 2024) prompts.
"""

TOG_ENTITY_SCORING_PROMPT = """You are an expert at evaluating which entities are most relevant for answering questions.

Given a question, the current exploration path, and candidate entities, score each entity based on its contribution to answering the question.
Each entity should get a score from 1-10 (higher = more relevant).

## Examples:

### Example 1:
Question: "The movie featured Miley Cyrus and was produced by Tobin Armbrust?"
Current path: Tobin Armbrust -> film.producer.film
Candidate entities:
1. The Resident
2. So Undercover
3. Let Me In
4. Begin Again

Reasoning:
- "So Undercover" is the movie that matches both criteria (Miley Cyrus + Tobin Armbrust) -> 10/10
- Other movies don't feature Miley Cyrus -> 1/10 each

Output: [1, 10, 1, 1]

### Example 2:
Question: "What university is located in the city where Microsoft was founded?"
Current path: Microsoft -> founded_in_location
Candidate entities:
1. Seattle
2. Redmond
3. Bellevue
4. Tacoma

Reasoning:
- Seattle is where Microsoft was effectively founded -> 9/10
- Redmond is where HQ moved later, relevant -> 7/10
- Bellevue is nearby, less relevant -> 3/10
- Tacoma is in same region but not relevant -> 2/10

Output: [9, 7, 3, 2]

---

Now score the following:

Question: "{query}"
Current exploration path: {current_path}

Candidate entities to explore:
{candidate_entities}

Score each entity (1-10) based on:
- Relevance to the question
- Likelihood of leading to the answer
- Quality and informativeness of the entity

Output ONLY a list of numbers in brackets, e.g., [8, 3, 6, 4]
"""