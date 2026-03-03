"""Application-specific error types for API contract mapping."""


class ServingContextUnavailableError(RuntimeError):
    """Raised when serving context dependencies are temporarily unavailable."""


class ServingContextNotReadyError(RuntimeError):
    """Raised when serving context exists but is not ready for reads."""


class ConversationStoreUnavailableError(RuntimeError):
    """Raised when server-side conversation storage cannot be reached."""


class ConversationSessionNotFoundError(ValueError):
    """Raised when a conversation session does not exist."""


class ConversationSessionMismatchError(ValueError):
    """Raised when a session does not belong to the requested collection."""
