"""Cosmos DB repository for prompts."""

from app.repositories.default_prompts import load_default_prompt_texts


class CosmosPromptRepository:
    """Repository for prompt storage in Cosmos DB."""

    def __init__(self, container):
        """Initialize with Cosmos container.

        Args:
            container: Azure Cosmos DB container instance
        """
        self._container = container

    def set_prompt(self, collection_id: str, prompt_name: str, content: str) -> None:
        """Store a prompt.

        Args:
            collection_id: ID of the collection
            prompt_name: Name of the prompt file
            content: Prompt text content
        """
        item = {
            "id": f"prompt:{collection_id}:{prompt_name}",
            "kind": "prompt",
            "collection_id": collection_id,
            "prompt_name": prompt_name,
            "content": content,
        }
        self._container.upsert_item(item)

    def get_prompt(self, collection_id: str, prompt_name: str) -> str | None:
        """Get a prompt by name.

        Args:
            collection_id: ID of the collection
            prompt_name: Name of the prompt file

        Returns:
            Prompt content or None if not found
        """
        try:
            item = self._container.read_item(
                item=f"prompt:{collection_id}:{prompt_name}",
                partition_key=f"prompt:{collection_id}:{prompt_name}",
            )
            return item["content"]
        except Exception:
            return None

    def list_prompt_names(self, collection_id: str) -> list[str]:
        """List all prompt names for a collection.

        Args:
            collection_id: ID of the collection

        Returns:
            List of prompt filenames
        """
        query = "SELECT * FROM c WHERE c.kind = 'prompt' AND c.collection_id = @collection_id"
        params = [{"name": "@collection_id", "value": collection_id}]
        items = list(self._container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True
        ))
        return [item["prompt_name"] for item in items]

    def seed_defaults(self, collection_id: str) -> None:
        """Seed default prompts for a collection.

        Args:
            collection_id: ID of the collection
        """
        default_prompts = load_default_prompt_texts()
        for prompt_name, content in default_prompts.items():
            self.set_prompt(collection_id, prompt_name, content)
