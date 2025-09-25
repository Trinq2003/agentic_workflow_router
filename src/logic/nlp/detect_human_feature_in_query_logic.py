from logic.base_classes import BaseLogic
from models import nlp_processor
import numpy as np
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
            logger.debug(f"[DetectHumanFeatureInQueryLogic] Analysis: {analysis}")

            # Check for person entities
            if analysis.entities:
                for entity in analysis.entities:
                    entity_type = entity.get('label', '').upper()
                    if any(keyword in entity_type for keyword in ['PERSON', 'PER', 'HUMAN']):
                        logger.debug(f"[DetectHumanFeatureInQueryLogic] Found human name: {entity_type}")
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
                return True

        return False

    def _detect_username(self, query: str) -> bool:
        """Detect usernames in encoded format."""
        query_lower = query.lower()

        for kw in self.nlp_processor.analyze_text_comprehensive(query).keywords:
            # Check if it follows the encoding pattern
            if self._is_valid_username_encoding(kw):
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
            return True
        
        return False

    def _detect_self_referential_words(self, query: str) -> bool:
        """
        Detect Vietnamese self-referential words in the query.
        
        Returns True if query contains words that reflect the user's self.
        """
        query_lower = query.lower()
        
        # Comprehensive list of Vietnamese self-referential words
        self_referential_words = [
            # Formal/polite first person pronouns
            "tôi", "tớ", "mình", "em", "con", "cháu", "anh", "chị",
            
            # Informal/casual first person pronouns
            "tao", "tau", "ta", "cha", "thằng này", "con này", "đứa này",
            
            # Possessive forms
            "của tôi", "của tớ", "của mình", "của em", "của con", "của cháu",
            "của tao", "của tau", "của ta", "của anh", "của chị",
            
            # Family/relationship context with self-reference
            "sếp tôi", "sếp tớ", "sếp em", "sếp anh", "sếp chị",
            "bố tôi", "mẹ tôi", "vợ tôi", "chồng tôi", "con tôi",
            "bố mày", "mẹ mày", "vợ mày", "chồng mày", "con mày",
            "bố tao", "mẹ tao", "vợ tao", "chồng tao", "con tao",
            "gia đình tôi", "nhà tôi", "công ty tôi", "team tôi",
            
            # Work/professional context
            "công việc của tôi", "dự án của tôi", "task của tôi",
            "lương của tôi", "thưởng của tôi", "kpi của tôi",
            
            # Personal items/belongings
            "điện thoại của tôi", "laptop của tôi", "máy tính của tôi",
            "tài khoản của tôi", "email của tôi", "password của tôi",
            
            # Actions with self-reference
            "tôi muốn", "tôi cần", "tôi làm", "tôi có", "tôi được", "tôi bị",
            "tao muốn", "tao cần", "tao làm", "tao có", "tao được", "tao bị",
            "mình muốn", "mình cần", "mình làm", "mình có", "mình được",
            
            # Question forms with self-reference
            "tôi phải", "tôi nên", "tôi có thể", "tôi được phép",
            "mình phải", "mình nên", "mình có thể", "mình được phép",
            
            # Regional/dialectal variations
            "ông", "bà", "cô", "chú", "bác" # when used as self-reference
        ]
        
        # Check for exact matches and partial matches
        for word in self_referential_words:
            # For single words, use word boundary matching to avoid false positives
            if len(word.split()) == 1 and len(word) <= 3:  # Short single words need boundary check
                pattern = r'\b' + re.escape(word) + r'\b'
                if re.search(pattern, query_lower):
                    logger.debug(f"[DetectHumanFeatureInQueryLogic] Found self-referential word: '{word}' in query")
                    return True
            else:  # Multi-word phrases or longer words can use simple substring matching
                if word in query_lower:
                    logger.debug(f"[DetectHumanFeatureInQueryLogic] Found self-referential word: '{word}' in query")
                    return True
        
        # Additional pattern matching for common Vietnamese self-reference patterns
        patterns = [
            r'\btôi\b',     # exact word "tôi"
            r'\btao\b',     # exact word "tao"  
            r'\bmình\b',    # exact word "mình"
            r'\btớ\b',      # exact word "tớ"
            r'\bta\b',      # exact word "ta" (but not "ta" in compound words)
            r'\bem\b(?!\s+(bé|nhỏ|lớn))',      # exact word "em" but not "em bé", "em nhỏ", etc.
            r'của\s+(tôi|tao|mình|tớ|ta|em)\b',  # "của" + pronoun
            r'\b(tôi|tao|mình|tớ|ta|em)\s+(muốn|cần|làm|có|được|bị|phải|nên)',  # pronoun + verb
            r'\b(sếp|bố|mẹ|vợ|chồng|con)\s+(tôi|tao|mình|tớ|mày|em)\b',  # relationship + pronoun
        ]
        
        for pattern in patterns:
            if re.search(pattern, query_lower):
                logger.debug(f"[DetectHumanFeatureInQueryLogic] Found self-referential pattern: '{pattern}' in query")
                return True
        
        return False

    def _parallel_detection(self, query: str) -> List[bool]:
        """Run detection methods in parallel for optimization."""
        results = [False] * 4  # Pre-allocate for ordered results

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit tasks with their index to maintain order
            futures = {
                executor.submit(self._detect_human_name, query): 0,
                executor.submit(self._detect_user_id, query): 1,
                executor.submit(self._detect_username, query): 2,
                executor.submit(self._detect_self_referential_words, query): 3
            }

            # Collect results in order
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.warning(f"Detection method at index {index} failed: {e}")
                    results[index] = False

        return results

    def forward(self, query: str):
        """
        Detect human-related features in the query and return tailored vectors for 8 workers.

        Workers mapping (from config/workers.yaml):
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
            Vector: Normalized probability distribution over 8 workers based on detected features
        """
        logger.debug(f"[DetectHumanFeatureInQueryLogic] Processing query: '{query}'")

        if not query or not isinstance(query, str):
            logger.debug(f"[DetectHumanFeatureInQueryLogic] Invalid input, returning zero vector")
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        # Use parallel detection for optimization
        detection_results = self._parallel_detection(query)
        
        # Break down detection results
        human_name_detected, user_id_detected, username_detected, self_ref_detected = detection_results
        logger.debug(f"[DetectHumanFeatureInQueryLogic] Human names detected: {human_name_detected}")
        logger.debug(f"[DetectHumanFeatureInQueryLogic] User IDs detected: {user_id_detected}")
        logger.debug(f"[DetectHumanFeatureInQueryLogic] Usernames detected: {username_detected}")
        logger.debug(f"[DetectHumanFeatureInQueryLogic] Self-referential words detected: {self_ref_detected}")

        # Create contribution mapping for this logic
        detected_features = []
        if human_name_detected:
            detected_features.append("human_name")
        if user_id_detected:
            detected_features.append("user_id")
        if username_detected:
            detected_features.append("username")
        if self_ref_detected:
            detected_features.append("self_referential")
        
        logger.debug(f"[DetectHumanFeatureInQueryLogic] Detected human features: {detected_features}")

        # Initialize result vector for 8 workers
        result = np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        # Determine worker preferences based on detected features
        if any(detection_results):
            # If self-referential words detected - high preference for EMPLOYEE_INFOR_AGENT and REMINDER_AGENT
            if self_ref_detected:
                result[0][6] += 0.2  # EMPLOYEE_INFOR_AGENT
                result[0][7] += 0.8  # REMINDER_AGENT
            
            # If specific user ID/username detected - focus on EMPLOYEE_INFOR_AGENT
            if user_id_detected:
                result[0][6] += 0.9  # EMPLOYEE_INFOR_AGENT
                result[0][7] += 0.1  # REMINDER_AGENT
            
            if username_detected:
                result[0][6] += 0.9  # EMPLOYEE_INFOR_AGENT
                result[0][7] += 0.1  # REMINDER_AGENT
            
            # If general human names detected - balanced approach
            if human_name_detected:
                result[0][6] += 0.5  # EMPLOYEE_INFOR_AGENT
                result[0][7] += 0.5  # REMINDER_AGENT
        else:
            # No human features detected - return zero vector
            logger.debug(f"[DetectHumanFeatureInQueryLogic] No human features detected -> zero vector")
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        
        if np.sum(result) > 0:
            result = result / np.sum(result)  # Normalize to ensure sum = 1
        
        logger.debug(f"[DetectHumanFeatureInQueryLogic] Final contribution - Query: '{query}' -> Features: {detected_features} -> Vector: {result.tolist()}")
        
        return result
        