"""
Topological & Invariant Exceptions for CoChem Core.
"""

from typing import Any, Optional


class MendeleevInvariantError(Exception):
    """Raised when an element, isotope, or chemical invariant lookup fails."""

    def __init__(self, message: str, symbol_or_query: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.symbol_or_query = symbol_or_query

    def __str__(self) -> str:
        if self.symbol_or_query is not None:
            return f"{self.message} (Query: {self.symbol_or_query})"
        return self.message
