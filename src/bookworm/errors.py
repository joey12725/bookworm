"""Exception hierarchy. Everything Bookworm raises on purpose derives from BookwormError."""


class BookwormError(Exception):
    """Base class for expected, user-facing errors."""


class ExtractionError(BookwormError):
    """A file could not be read or parsed."""


class UnsupportedFormatError(ExtractionError):
    """No extractor is registered for this file type."""


class MissingDependencyError(BookwormError):
    """An optional dependency needed for this feature is not installed."""

    def __init__(self, feature: str, extra: str) -> None:
        super().__init__(
            f"{feature} needs an optional dependency. "
            f"Install it with: pip install 'bookworm[{extra}]'"
        )


class ProviderError(BookwormError):
    """An embedding or chat provider returned an error."""


class IndexMismatchError(BookwormError):
    """The index was built with a different embedder than the one configured."""


class BookNotFoundError(BookwormError):
    """No book in the library matches the given title."""
