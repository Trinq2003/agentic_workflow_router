from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List

from base_classes import BaseQueryProcessingClass
from logic import BaseLogic


class BaseStrategy(BaseQueryProcessingClass):
    """Abstract base class for strategy components that process queries."""
    logics: List[BaseLogic]

    def __init__(self):
        super().__init__()
        self.logics = []

    def add_logic(self, logic: BaseLogic):
        self.logics.append(logic)

    @abstractmethod
    def _load_labels(self) -> List[str]:
        """Load labels for this strategy. To be implemented by child classes."""
        pass

    @abstractmethod
    def _reduce(self, results: List[Any]) -> Any:
        pass

    def forward(self, query: str) -> List[Any]:
        """
        Process a query by running all logics in parallel, then reducing and labeling the results.

        Args:
            query: Input query to be processed

        Returns:
            List[Any]: Final labeled output from processing the query
        """
        # Run all logic forward methods in parallel
        results = []
        with ThreadPoolExecutor() as executor:
            future_to_logic = {executor.submit(logic.forward, query): logic for logic in self.logics}
            for future in as_completed(future_to_logic):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f'Logic {future_to_logic[future]} generated an exception: {exc}')

        reduced_result = self._reduce(results)
        return reduced_result
        
