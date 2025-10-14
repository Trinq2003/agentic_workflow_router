from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

from base_classes import BaseQueryProcessingClass


class BaseLogic(BaseQueryProcessingClass):
    """Abstract base class for logic components that process queries."""
    _context: Dict[str, Any]

    def __init__(self):
        super().__init__()
        # Shared per-query context injected by a strategy (e.g., comprehensive NLP analysis)
        self._context = {}

    def set_context(self, context: Dict[str, Any]):
        """Inject shared per-query context (e.g., {'analysis': TextAnalysis, ...})."""
        self._context = context

    def get_context(self) -> Dict[str, Any]:
        """Retrieve shared per-query context if available, else None."""
        return self._context

    @abstractmethod
    def forward(self, query: str) -> Any:
        """
        Abstract method that processes a query and returns a vector output.

        Args:
            query: Input query to be processed

        Returns:
            Any: Output vector (e.g., numpy array) from processing the query
        """
        pass