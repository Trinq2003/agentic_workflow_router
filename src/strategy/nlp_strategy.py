from typing import List, Any
from strategy.base_classes import BaseStrategy
import logging
import torch
from logic.nlp import DetectSyntaxInQueryLogic, FindTimePatternInQueryLogic, DetectHumanFeatureInQueryLogic
logger = logging.getLogger(__name__)


class NLPStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.add_logic(DetectSyntaxInQueryLogic())
        self.add_logic(FindTimePatternInQueryLogic())
        self.add_logic(DetectHumanFeatureInQueryLogic())
        
    def _reduce(self, results: List[Any]) -> Any:
        """
        Sum tensors from all NLP logics element-wise.

        Args:
            results: List of tensors from each logic's forward() method

        Returns:
            Summed tensor from all logic results
        """
        if not results:
            return torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float32)
        
        try:
            summed_result = torch.zeros_like(results[0])
            for result in results:
                summed_result = summed_result + result
            return summed_result
        except Exception as e:
            logger.error(f"Error summing tensors: {e}")
            raise