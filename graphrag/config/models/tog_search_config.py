from pydantic import BaseModel, Field

class ToGSearchConfig(BaseModel):
    """Configuration for ToG (Think-on-Graph) search."""

    # Model Configuration
    chat_model_id: str = Field(
        description="The model ID to use for ToG search.",
        default="default_chat_model"
    )
    embedding_model_id: str = Field(
        description="The model ID to use for embeddings.",
        default="default_embedding_model"
    )

    # Exploration Parameters
    width: int = Field(
        description="Beam width for exploration (number of entities to keep per level).",
        default=3
    )
    depth: int = Field(
        description="Maximum depth of graph traversal.",
        default=3
    )

    # Pruning Configuration
    prune_strategy: str = Field(
        description="Pruning strategy: 'llm', 'bm25', or 'semantic'.",
        default="llm"
    )
    num_retain_entity: int = Field(
        description="Number of entities to retain during pruning.",
        default=5
    )

    # Temperature Settings
    temperature_exploration: float = Field(
        description="Temperature for exploration phase.",
        default=0.4
    )
    temperature_reasoning: float = Field(
        description="Temperature for reasoning phase.",
        default=0.0
    )

    # Context and Token Limits
    max_context_tokens: int = Field(
        description="Maximum tokens for context.",
        default=8000
    )
    max_exploration_paths: int = Field(
        description="Maximum number of exploration paths to maintain.",
        default=10
    )

    # Prompts
    relation_scoring_prompt: str | None = Field(
        description="Custom prompt for relation scoring.",
        default=None
    )
    entity_scoring_prompt: str | None = Field(
        description="Custom prompt for entity scoring.",
        default=None
    )
    reasoning_prompt: str | None = Field(
        description="Custom prompt for final reasoning.",
        default=None
    )