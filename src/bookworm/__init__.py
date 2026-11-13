"""Bookworm: a local, citable knowledge base for your ebook library."""

from bookworm.app import Bookworm
from bookworm.config import Settings
from bookworm.models import Answer, Book, Citation, Hit

__all__ = ["Answer", "Book", "Bookworm", "Citation", "Hit", "Settings", "__version__"]
__version__ = "0.2.0"
