from __future__ import annotations

__all__ = ("PangeaAIGuardBlockedError",)


class PangeaAIGuardBlockedError(Exception):
    """Raised when Pangea AI Guard returns a blocked response."""

    def __init__(self, message: str = "Pangea AI Guard returned a blocked response.") -> None:
        super().__init__(message)
