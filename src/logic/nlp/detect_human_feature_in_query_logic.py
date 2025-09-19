from logic.base_classes import BaseLogic
from models import nlp_processor
import torch
import re
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class DetectHumanFeatureInQueryLogic(BaseLogic):
    """
    Logic class for detecting human-related features in queries.

    Detects:
    - Human names using NER
    - User IDs (6-digit numbers: 000000-999999)
    - Usernames in encoded format: {lastname}{leading_chars}{optional_number}
    """

    def __init__(self):
        super().__init__()
        # Initialize NLP processor for entity recognition
        try:
            self.nlp_processor = nlp_processor
        except Exception as e:
            logger.warning(f"Failed to initialize NLP processor: {e}")
            self.nlp_processor = None

        logger.info("Initialized DetectHumanFeatureInQueryLogic")

    def _detect_human_name(self, query: str) -> bool:
        """Detect human names using NER."""
        if not self.nlp_processor:
            return False

        try:
            # Use comprehensive analysis to detect entities
            analysis = self.nlp_processor.analyze_text_comprehensive(query)
            logger.debug(f"NLP analysis: {analysis}")

            # Check for person entities
            if analysis.entities:
                logger.debug(f"NLP entities: {analysis.entities}")
                for entity in analysis.entities:
                    entity_type = entity.get('label', '').upper()
                    if any(keyword in entity_type for keyword in ['PERSON', 'PER', 'HUMAN']):
                        logger.debug(f"Human name entity found: {entity}")
                        return True

        except Exception as e:
            logger.warning(f"NLP human name detection failed: {e}")

        return False

    def _detect_user_id(self, query: str) -> bool:
        """Detect 6-digit user IDs (000000-999999)."""
        # Pattern for exactly 6 digits
        user_id_pattern = r'\b\d{6}\b'

        matches = re.findall(user_id_pattern, query)
        for match in matches:
            # Check if it's within the valid range (000000-999999)
            user_id = int(match)
            if 0 <= user_id <= 999999:
                logger.debug(f"Valid user ID found: {match}")
                return True
        
        logger.debug(f"Valid user ID found: {match}")
        return False

    def _detect_username(self, query: str) -> bool:
        """Detect usernames in encoded format."""
        query_lower = query.lower()

        for kw in self.nlp_processor.analyze_text_comprehensive(query).keywords:
            # Check if it follows the encoding pattern
            if self._is_valid_username_encoding(kw):
                logger.debug(f"Valid username encoding found: {kw}")
                return True

        return False

    def _is_valid_username_encoding(self, username: str) -> bool:
        """Check if username follows the encoding pattern using regex."""
        # Username encoding pattern for Vietnamese names
        pattern = r'^((ch|gh|gi|kh|ng|nh|ph|qu|th|tr|[bcdfghklmnpqrstvxz]|[aeiouy])[aeiouy][mngtchpkxnhng]?|[aeiouy][mngtchpkx])[a-z]+(\d{1,3})$'

        # Convert to lowercase for matching
        username_lower = username.lower()

        # Check if username matches the pattern
        if re.match(pattern, username_lower):
            logger.debug(f"Valid username encoding found: {username}")
            return True

        logger.debug(f"Invalid username encoding: {username}")
        return False

    def _parallel_detection(self, query: str) -> List[bool]:
        """Run detection methods in parallel for optimization."""
        results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks
            future_human_name = executor.submit(self._detect_human_name, query)
            future_user_id = executor.submit(self._detect_user_id, query)
            future_username = executor.submit(self._detect_username, query)

            # Collect results
            for future in as_completed([future_human_name, future_user_id, future_username]):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.warning(f"Detection method failed: {e}")
                    results.append(False)

        logger.debug(f"Human feature detection results: {results}")
        return results

    def forward(self, query: str) -> torch.Tensor:
        """
        Detect human-related features in the query.

        Args:
            query: Input query string

        Returns:
            torch.Tensor: 2D tensor [[0,0,0,1,0]] if human features detected, [[0,0,0,0,0]] otherwise
        """
        if not query or not isinstance(query, str):
            return torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float32)

        # Use parallel detection for optimization
        detection_results = self._parallel_detection(query)

        # If any human feature detected, return 1 at position 3
        has_human_feature = any(detection_results)
        logger.debug(f"Has human feature: {has_human_feature}")

        # Return tensor with result
        if has_human_feature:
            result = torch.tensor([[0, 0, 0, 1, 0]], dtype=torch.float32)
        else:
            result = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float32)

        logger.debug(f"Query: '{query}' -> Human feature detected: {has_human_feature}")
        return result
        