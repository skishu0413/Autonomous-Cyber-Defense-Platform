"""BaseConnector — abstract lifecycle interface for all production connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["BaseConnector"]


class BaseConnector(ABC):
    """Common lifecycle interface for all production connectors.

    Every connector must implement ``start()`` and ``stop()``; they are called
    by ``Platform.start_connectors()`` and ``Platform.stop_connectors()``
    respectively.  Both methods must be idempotent: calling them multiple times
    must not raise errors.
    """

    @abstractmethod
    async def start(self) -> None:
        """Initialize resources and begin processing. Idempotent."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Flush in-flight work, release resources, exit cleanly."""
        ...
