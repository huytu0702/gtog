from pathlib import Path
import sys

try:
    from graphrag.prompts.index.extract_claims import CLAIM_EXTRACTION_PROMPT
    from graphrag.prompts.index.community_report import COMMUNITY_REPORT_PROMPT
    from graphrag.prompts.index.community_report_text_units import COMMUNITY_REPORT_TEXT_UNITS_PROMPT
    from graphrag.prompts.index.extract_graph import GRAPH_EXTRACTION_PROMPT
    from graphrag.prompts.index.summarize_descriptions import SUMMARIZE_PROMPT
    
    print("✅ Successfully imported prompts")
    print(f"CLAIM_EXTRACTION_PROMPT length: {len(CLAIM_EXTRACTION_PROMPT)}")
    
except Exception as e:
    print(f"❌ Failed to import prompts: {e}")
    sys.exit(1)
