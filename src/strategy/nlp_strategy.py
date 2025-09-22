from typing import List, Any
from strategy.base_classes import BaseStrategy
import logging
import torch
import yaml
from pathlib import Path
from logic.nlp import DetectSyntaxInQueryLogic, FindTimePatternInQueryLogic, DetectHumanFeatureInQueryLogic
logger = logging.getLogger(__name__)


class WorkerLabelingNLPStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.add_logic(DetectSyntaxInQueryLogic())
        self.add_logic(FindTimePatternInQueryLogic())
        self.add_logic(DetectHumanFeatureInQueryLogic())

        # Load workers configuration
        self.workers = self._load_labels()

    def _load_labels(self) -> List[str]:
        """Load worker labels from config/workers.yaml."""
        config_path = Path(__file__).parent.parent.parent / "config" / "workers.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            workers = config.get('workers', [])
            logger.info(f"Loaded {len(workers)} worker labels: {workers}")
            return workers
        except Exception as e:
            logger.error(f"Failed to load worker labels: {e}, using empty list")
            return []

    def _reduce(self, results: List[Any]) -> Any:
        """
        Implement max-vote mechanism: sum tensors, then select positions with maximum vote counts.

        Args:
            results: List of tensors from each logic's forward() method

        Returns:
            Tensor with 1s at positions with maximum vote counts, 0s elsewhere
        """
        if not results:
            return torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float32)

        try:
            # Sum all tensors to get vote counts
            summed_result = torch.zeros_like(results[0])
            for result in results:
                summed_result = summed_result + result

            # Find the maximum vote count
            max_votes = torch.max(summed_result)

            # Create output tensor: 1 where vote count equals max_votes, 0 otherwise
            vote_result = torch.where(summed_result == max_votes, torch.tensor(1.0), torch.tensor(0.0))

            logger.debug(f"Vote counts: {summed_result}, Max votes: {max_votes}, Selected workers: {vote_result}")
            return vote_result
        except Exception as e:
            logger.error(f"Error in max-vote reduction: {e}")
            raise


# Alias for backward compatibility
NLPStrategy = WorkerLabelingNLPStrategy