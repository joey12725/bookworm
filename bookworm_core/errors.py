class BookwormError(Exception):
    """Base exception for predictable Bookworm errors."""


class IngestionError(BookwormError):
    """Raised when a book fails ingestion."""


class ExtractionError(BookwormError):
    """Raised when text extraction fails."""


class OpenRouterError(BookwormError):
    """Raised when OpenRouter requests fail."""
