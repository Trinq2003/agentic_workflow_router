from typing import List, Any
from strategy.base_classes import BaseStrategy
import logging
import numpy as np
import yaml
from pathlib import Path
from logic.nlp import (
    DetectSyntaxInQueryLogic, 
    FindTimePatternInQueryLogic, 
    DetectHumanFeatureInQueryLogic, 
    FindLocationPatternInQueryLogic,
    DetectNumericalRequirementInQueryLogic,
    DetectDocSearchFeatureInQueryLogic,
)
logger = logging.getLogger(__name__)


class WorkerLabelingNLPStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.add_logic(DetectSyntaxInQueryLogic())
        self.add_logic(FindTimePatternInQueryLogic())
        self.add_logic(DetectDocSearchFeatureInQueryLogic())
        self.add_logic(DetectHumanFeatureInQueryLogic())
        self.add_logic(FindLocationPatternInQueryLogic())
        self.add_logic(DetectNumericalRequirementInQueryLogic())
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

    def _reduce(self, results: List[Any]) -> List[str]:
        """
        Implement vote-based ranking: sum tensors, then return workers sorted by vote counts
        along with their corresponding scores.

        Args:
            results: List of tensors from each logic's forward() method

        Returns:
            List of worker names sorted by vote counts in descending order
        """
        logger.debug(f"[WorkerLabelingNLPStrategy] Starting reduction with {len(results)} logic results")

        if not results:
            logger.debug(f"[WorkerLabelingNLPStrategy] No results to reduce, returning empty list")
            return []

        try:
            # Log individual logic contributions (now properly aligned)
            logic_names = [logic_.__class__.__name__ for logic_ in self.logics]
            for i, (logic_name, result) in enumerate(zip(logic_names, results)):
                if result is not None:
                    logger.debug(f"[WorkerLabelingNLPStrategy] {logic_name} contribution: {result.tolist()}")
                else:
                    logger.warning(f"[WorkerLabelingNLPStrategy] {logic_name} failed to produce a result")

            # Filter out None results before processing
            valid_results = [result for result in results if result is not None]
            
            if not valid_results:
                logger.warning(f"[WorkerLabelingNLPStrategy] No valid results from any logic")
                return []

            # Sum all vectors to get vote counts
            summed_result = np.zeros_like(valid_results[0])
            for result in valid_results:
                summed_result = summed_result + result

            logger.debug(f"[WorkerLabelingNLPStrategy] Summed vote counts: {summed_result.tolist()}")

            # Get vote counts for each worker (flatten to 1D array)
            vote_counts = summed_result.flatten()
            
            # Create list of (worker_name, vote_count) tuples
            worker_votes = [(self.workers[i], float(vote_counts[i])) for i in range(len(self.workers))]
            
            # Sort by vote count in descending order, then by worker name for consistency
            worker_votes.sort(key=lambda x: (-x[1], x[0]))
            
            # Extract just the worker names
            sorted_workers = [worker for worker, votes in worker_votes]
            
            # Log the results
            logger.debug(f"[WorkerLabelingNLPStrategy] Worker vote counts: {dict(worker_votes)}")
            logger.debug(f"[WorkerLabelingNLPStrategy] Workers sorted by votes: {sorted_workers}")

            # Also return the corresponding votes in the same order
            sorted_votes = [votes for _, votes in worker_votes]

            return {"labels": sorted_workers, "votes": sorted_votes}
        except Exception as e:
            logger.error(f"[WorkerLabelingNLPStrategy] Error in vote-based reduction: {e}")
            raise


# Alias for backward compatibility
NLPStrategy = WorkerLabelingNLPStrategy