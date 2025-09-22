from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from base_classes import BaseQueryProcessingClass


class BaseLogic(BaseQueryProcessingClass):
    """Abstract base class for logic components that process queries."""

    def __init__(self):
        super().__init__()

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