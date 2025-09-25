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
        results = [None] * len(self.logics)  # Pre-allocate results list to maintain order
        with ThreadPoolExecutor() as executor:
            # Submit futures with their index to maintain order
            future_to_index_logic = {
                executor.submit(logic.forward, query): (i, logic) 
                for i, logic in enumerate(self.logics)
            }
            
            for future in as_completed(future_to_index_logic):
                index, logic = future_to_index_logic[future]
                try:
                    result = future.result()
                    results[index] = result  # Store result at the correct index
                except Exception as exc:
                    print(f'Logic {logic} generated an exception: {exc}')
                    # Store None or a default value for failed logic
                    results[index] = None

        reduced_result = self._reduce(results)
        return reduced_result
        
