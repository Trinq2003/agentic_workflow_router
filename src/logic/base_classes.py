from abc import ABC, abstractmethod
from typing import Any
import torch

from base_classes import BaseQueryProcessingClass


class BaseLogic(BaseQueryProcessingClass):
    """Abstract base class for logic components that process queries."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, query: str) -> torch.Tensor:
        """
        Abstract method that processes a query and returns a tensor output.

        Args:
            query: Input query to be processed

        Returns:
            torch.Tensor: Output tensor from processing the query
        """
        pass