from abc import ABC, abstractmethod
from typing import Any

class BaseQueryProcessingClass(ABC):
    """Abstract base class for all logic components."""

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, query: str) -> Any:
        """
        Abstract method that processes a query and returns a output.

        Args:
            query: Input query to be processed

        Returns:
            Any: Output from processing the query
        """
        pass