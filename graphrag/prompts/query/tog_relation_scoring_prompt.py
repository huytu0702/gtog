"""Default relation scoring prompt for ToG search."""

TOG_RELATION_SCORING_PROMPT = """
Given the question: "{query}"
Currently exploring entity: {entity_name}

Available relations:
{relations}

Score each relation (1-10) based on how likely it leads to answering the question.
Output format: [score1, score2, score3, ...]
Only output the list of numbers.
"""