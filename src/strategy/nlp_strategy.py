from typing import List, Any
from strategy.base_classes import BaseStrategy
import logging
import numpy as np
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
        logger.debug(f"[WorkerLabelingNLPStrategy] Starting reduction with {len(results)} logic results")

        if not results:
            logger.debug(f"[WorkerLabelingNLPStrategy] No results to reduce, returning zero vector")
            return np.array([[0, 0, 0, 0, 0]], dtype=np.float32)

        try:
            # Log individual logic contributions
            logic_names = ["DetectSyntaxInQueryLogic", "FindTimePatternInQueryLogic", "DetectHumanFeatureInQueryLogic"]
            for i, (logic_name, result) in enumerate(zip(logic_names, results)):
                logger.debug(f"[WorkerLabelingNLPStrategy] {logic_name} contribution: {result.tolist()}")

            # Sum all vectors to get vote counts
            summed_result = np.zeros_like(results[0])
            for result in results:
                summed_result = summed_result + result

            logger.debug(f"[WorkerLabelingNLPStrategy] Summed vote counts: {summed_result.tolist()}")

            # Find the maximum vote count
            max_votes = np.max(summed_result)
            logger.debug(f"[WorkerLabelingNLPStrategy] Maximum vote count: {float(max_votes)}")

            # Create output vector: 1 where vote count equals max_votes, 0 otherwise
            if float(max_votes) == 0.0:
                logger.debug(f"[WorkerLabelingNLPStrategy] No max votes, returning zero vector")
                return np.array([[0, 0, 0, 0, 0]], dtype=np.float32)

            vote_result = (summed_result == max_votes).astype(np.float32)

            # Log which workers were selected
            selected_workers = [self.workers[i] for i in range(len(vote_result[0])) if vote_result[0][i] == 1]
            logger.debug(f"[WorkerLabelingNLPStrategy] Selected workers: {selected_workers}")
            logger.debug(f"[WorkerLabelingNLPStrategy] Final output vector: {vote_result.tolist()}")

            return vote_result
        except Exception as e:
            logger.error(f"[WorkerLabelingNLPStrategy] Error in max-vote reduction: {e}")
            raise


# Alias for backward compatibility
NLPStrategy = WorkerLabelingNLPStrategy