"""Default prompt loading utilities."""


def load_default_prompt_texts() -> dict[str, str]:
    """Load all default GraphRAG prompts.

    Returns:
        Dictionary mapping prompt filenames to their text content
    """
    prompts = {}

    try:
        # Import index prompts
        from graphrag.prompts.index.extract_claims import EXTRACT_CLAIMS_PROMPT
        from graphrag.prompts.index.community_report import COMMUNITY_REPORT_PROMPT
        from graphrag.prompts.index.community_report_text_units import (
            COMMUNITY_REPORT_TEXT_PROMPT,
        )
        from graphrag.prompts.index.extract_graph import GRAPH_EXTRACTION_PROMPT
        from graphrag.prompts.index.summarize_descriptions import SUMMARIZE_PROMPT

        # Import query prompts
        from graphrag.prompts.query.basic_search_system_prompt import (
            BASIC_SEARCH_SYSTEM_PROMPT,
        )
        from graphrag.prompts.query.drift_search_system_prompt import (
            DRIFT_LOCAL_SYSTEM_PROMPT,
            DRIFT_REDUCE_PROMPT,
        )
        from graphrag.prompts.query.global_search_knowledge_system_prompt import (
            GENERAL_KNOWLEDGE_INSTRUCTION,
        )
        from graphrag.prompts.query.global_search_map_system_prompt import (
            MAP_SYSTEM_PROMPT,
        )
        from graphrag.prompts.query.global_search_reduce_system_prompt import (
            REDUCE_SYSTEM_PROMPT,
        )
        from graphrag.prompts.query.local_search_system_prompt import (
            LOCAL_SEARCH_SYSTEM_PROMPT,
        )
        from graphrag.prompts.query.question_gen_system_prompt import (
            QUESTION_SYSTEM_PROMPT,
        )

        # Import ToG prompts
        from graphrag.prompts.query.tog_entity_scoring_prompt import (
            TOG_ENTITY_SCORING_PROMPT,
        )
        from graphrag.prompts.query.tog_relation_scoring_prompt import (
            TOG_RELATION_SCORING_PROMPT,
        )
        from graphrag.prompts.query.tog_reasoning_prompt import TOG_REASONING_PROMPT

        # Index prompts
        prompts["extract_graph.txt"] = GRAPH_EXTRACTION_PROMPT
        prompts["summarize_descriptions.txt"] = SUMMARIZE_PROMPT
        prompts["extract_claims.txt"] = EXTRACT_CLAIMS_PROMPT
        prompts["community_report_graph.txt"] = COMMUNITY_REPORT_PROMPT
        prompts["community_report_text.txt"] = COMMUNITY_REPORT_TEXT_PROMPT

        # Query prompts
        prompts["drift_search_system_prompt.txt"] = DRIFT_LOCAL_SYSTEM_PROMPT
        prompts["drift_search_reduce_prompt.txt"] = DRIFT_REDUCE_PROMPT
        prompts["global_search_map_system_prompt.txt"] = MAP_SYSTEM_PROMPT
        prompts["global_search_reduce_system_prompt.txt"] = REDUCE_SYSTEM_PROMPT
        prompts["global_search_knowledge_system_prompt.txt"] = GENERAL_KNOWLEDGE_INSTRUCTION
        prompts["local_search_system_prompt.txt"] = LOCAL_SEARCH_SYSTEM_PROMPT
        prompts["basic_search_system_prompt.txt"] = BASIC_SEARCH_SYSTEM_PROMPT
        prompts["question_gen_system_prompt.txt"] = QUESTION_SYSTEM_PROMPT

        # ToG prompts
        prompts["tog_entity_scoring_prompt.txt"] = TOG_ENTITY_SCORING_PROMPT
        prompts["tog_relation_scoring_prompt.txt"] = TOG_RELATION_SCORING_PROMPT
        prompts["tog_reasoning_prompt.txt"] = TOG_REASONING_PROMPT

    except Exception as e:
        import logging
        logging.warning(f"Failed to load default prompts: {e}")

    return prompts
