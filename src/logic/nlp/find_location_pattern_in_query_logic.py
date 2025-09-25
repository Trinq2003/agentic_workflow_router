from logic.base_classes import BaseLogic
import numpy as np
import re
import logging

from logic.nlp.utils import is_known_location

logger = logging.getLogger(__name__)


class FindLocationPatternInQueryLogic(BaseLogic):
    def forward(self, query: str):
        """
        Detect Vietnamese locations and return tailored vectors for 8 workers.

        Workers mapping:
        0: FUNCTION_CALLER_AGENT
        1: DOCS_SEARCHER_AGENT  
        2: NETMIND_INFOR_AGENT
        3: REGION_IDENTIFIER_AGENT
        4: WEBS_SEARCHER_AGENT
        5: TIME_IDENTIFIER_AGENT
        6: EMPLOYEE_INFOR_AGENT
        7: REMINDER_AGENT

        Args:
            query: Input query string

        Returns:
            Vector: Probability distribution over 8 workers based on location detection
        """
        logger.debug(f"[FindLocationPatternInQueryLogic] Processing query: '{query}'")

        if not query or not isinstance(query, str):
            logger.debug(f"[FindLocationPatternInQueryLogic] Invalid input, returning zero vector")
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        # Lowercase for comparison as requested
        text = query.lower()

        # Tokenize to words and create n-grams up to length 6
        tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        max_n = min(6, len(tokens)) if tokens else 0

        contains_location = False
        detected_locations = []
        
        for n in range(max_n, 0, -1):
            for i in range(0, len(tokens) - n + 1):
                candidate = " ".join(tokens[i:i + n])
                if is_known_location(candidate):
                    contains_location = True
                    detected_locations.append(candidate)
                    logger.debug(f"[FindLocationPatternInQueryLogic] Matched location candidate: '{candidate}'")
                    break
            if contains_location:
                break

        # Initialize result vector for 8 workers
        result = np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        if contains_location:
            # Location detected - distribute weights based on location context
            # Primary preference for REGION_IDENTIFIER_AGENT (location-specific queries)
            result[0][3] += 0.6  # REGION_IDENTIFIER_AGENT (primary for location queries)
            
            # Secondary preferences
            result[0][4] += 0.2  # WEBS_SEARCHER_AGENT (web search for location info)
            result[0][1] += 0.2  # DOCS_SEARCHER_AGENT (documentation about locations)
        else:
            # No location detected - return zero vector
            logger.debug(f"[FindLocationPatternInQueryLogic] No location detected -> zero vector")
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        # Normalize to ensure sum = 1
        if np.sum(result) > 0:
            result = result / np.sum(result)
        
        logger.debug(f"[FindLocationPatternInQueryLogic] Final contribution - Query: '{query}' -> Locations: {detected_locations} -> Vector: {result.tolist()}")
        return result
        