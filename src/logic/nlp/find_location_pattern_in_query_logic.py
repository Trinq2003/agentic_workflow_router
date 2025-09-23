from logic.base_classes import BaseLogic
from models import nlp_processor
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from logic.nlp.utils import is_known_location

logger = logging.getLogger(__name__)


class FindLocationPatternInQueryLogic(BaseLogic):
    def forward(self, query: str):
        """
        Detect if the query contains any known Vietnam location.

        Args:
            query: Input query string

        Returns:
            Vector: [[0.5,0.5,0,0,0]] if a location is detected, [[0,0,0,0,0]] otherwise
        """
        logger.debug(f"[FindLocationPatternInQueryLogic] Processing query: '{query}'")

        if not query or not isinstance(query, str):
            logger.debug(f"[FindLocationPatternInQueryLogic] Invalid input, returning zero vector")
            return np.array([[0, 0, 0, 0, 0]], dtype=np.float32)

        # Lowercase for comparison as requested
        text = query.lower()

        # Tokenize to words and create n-grams up to length 6
        tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        max_n = min(6, len(tokens)) if tokens else 0

        contains_location = False
        for n in range(max_n, 0, -1):
            for i in range(0, len(tokens) - n + 1):
                candidate = " ".join(tokens[i:i + n])
                if is_known_location(candidate):
                    contains_location = True
                    logger.debug(f"[FindLocationPatternInQueryLogic] Matched location candidate: '{candidate}'")
                    break
            if contains_location:
                break

        if contains_location:
            result = np.array([[0.5, 0.5, 0, 0, 0]], dtype=np.float32)
            logger.debug(f"[FindLocationPatternInQueryLogic] Output vector: {result.tolist()} (location detected)")
        else:
            result = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)
            logger.debug(f"[FindLocationPatternInQueryLogic] Output vector: {result.tolist()} (no location detected)")

        return result
        