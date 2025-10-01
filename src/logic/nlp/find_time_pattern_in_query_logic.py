from logic.base_classes import BaseLogic
from models import nlp_processor
import numpy as np
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class FindTimePatternInQueryLogic(BaseLogic):
    """
    Logic class for detecting time patterns in queries.

    Supports various time formats including:
    - Absolute dates (19/9/2025, 19-9-2025, etc.)
    - Relative time expressions (hôm nay, mai, tuần này, etc.)
    - Time mentions (17:00, buổi sáng, etc.)
    - Language-specific processing (Vietnamese or English)
    """

    def __init__(self, language: str = "vietnamese"):
        super().__init__()
        self.language = language.lower()

        if self.language not in ["vietnamese", "english"]:
            raise ValueError(f"Unsupported language: {language}. Choose 'vietnamese' or 'english'")

        # Initialize language-specific configurations
        self._setup_language_config()

    def _setup_language_config(self):
        """Setup language-specific keywords and patterns."""
        if self.language == "vietnamese":
            self.time_keywords = {
                'absolute': [
                    'ngày', 'tháng', 'năm', 'giờ', 'phút', 'giây', 'tuần', 'quý',
                    'hôm nay', 'hôm qua', 'hôm kia', 'ngày mai',
                    'tuần này', 'tuần trước', 'tuần sau', 'tháng này',
                    'tháng trước', 'tháng sau', 'năm nay', 'năm trước', 'năm sau'
                ],
                'relative': [
                    'bây giờ', 'hiện tại', 'hiện nay', 'vừa rồi', 'mới đây',
                    'sáng nay', 'chiều nay', 'tối nay', 'đêm nay',
                    'sáng qua', 'chiều qua', 'tối qua', 'đêm qua',
                    'sáng mai', 'chiều mai', 'tối mai', 'đêm mai',
                    'buổi sáng', 'buổi chiều', 'buổi tối', 'buổi trưa',
                    'nửa đầu', 'nửa cuối', 'đầu tháng', 'cuối tháng',
                    'đầu tuần', 'cuối tuần', 'giờ cao điểm', 'giữa trưa'
                ],
                'time_indicators': [
                    'vào lúc', 'lúc', 'khoảng',
                    'khoảng thời gian', 'thời điểm', 'thời gian'
                ]
            }

            # Vietnamese-specific regex patterns
            self.time_patterns = [
                # Date patterns (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, etc.)
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                # Time patterns (HH:MM, H:MM, etc.)
                r'\b\d{1,2}:\d{2}(?::\d{2})?\b',
                # Month/Day patterns (MM/DD, DD/MM)
                r'\b\d{1,2}/\d{1,2}\b',
                # Year patterns (YYYY)
                r'\b(19|20)\d{2}\b',
                # Vietnamese date patterns (ngày DD tháng MM năm YYYY)
                r'ngày\s+\d{1,2}\s+tháng\s+\d{1,2}(\s+năm\s+\d{4})?',
                # Vietnamese day patterns
                r'\b(thứ\s+(hai|ba|tư|năm|sáu|bảy|chủ nhật))\b',
            ]

            # Vietnamese temporal indicators for NLP
            self.temporal_indicators = [
                'khi nào', 'lúc nào', 'thời gian', 'lúc', 'bao lâu'
            ]

        elif self.language == "english":
            self.time_keywords = {
                'absolute': [
                    'today', 'yesterday', 'tomorrow', 'day', 'month', 'year', 'week', 'quarter',
                    'hour', 'minute', 'second', 'this week', 'last week', 'next week',
                    'this month', 'last month', 'next month', 'this year', 'last year', 'next year'
                ],
                'relative': [
                    'now', 'currently', 'recently', 'just now', 'a while ago',
                    'this morning', 'this afternoon', 'this evening', 'last night',
                    'morning', 'afternoon', 'evening', 'noon', 'night',
                    'first half', 'second half', 'early month', 'late month',
                    'early week', 'late week', 'peak hours', 'midday'
                ],
                'time_indicators': [
                    'at', 'during', 'around',
                    'time period', 'moment', 'when', 'schedule'
                ]
            }

            # English-specific regex patterns
            self.time_patterns = [
                # Date patterns (DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, etc.)
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                # Time patterns (HH:MM, H:MM, etc.)
                r'\b\d{1,2}:\d{2}(?::\d{2})?\b',
                # Month/Day patterns (MM/DD, DD/MM)
                r'\b\d{1,2}/\d{1,2}\b',
                # Year patterns (YYYY)
                r'\b(19|20)\d{2}\b',
                # English date patterns (Month DD, YYYY)
                r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b',
                # Day patterns (Monday, Tuesday, etc.)
                r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            ]

            # English temporal indicators for NLP
            self.temporal_indicators = [
                'when', 'what time', 'schedule', 'remind', 'at', 'on'
            ]

        # Initialize NLP processor for entity recognition
        try:
            self.nlp_processor = nlp_processor
        except Exception as e:
            logger.warning(f"Failed to initialize NLP processor: {e}")
            self.nlp_processor = None

        logger.info(f"Initialized {self.language} language configuration")

    def set_language(self, language: str):
        """
        Change the language configuration.

        Args:
            language: Language to set ('vietnamese' or 'english')
        """
        language = language.lower()
        if language not in ["vietnamese", "english"]:
            raise ValueError(f"Unsupported language: {language}. Choose 'vietnamese' or 'english'")

        if language != self.language:
            self.language = language
            self._setup_language_config()
            logger.info(f"Language changed to {self.language}")

    def get_language(self) -> str:
        """Get the current language setting."""
        return self.language

    def _detect_regex_patterns(self, query: str) -> bool:
        """Detect time patterns using regex."""
        query_lower = query.lower()

        # Check regex patterns
        for pattern in self.time_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.debug(f"[FindTimePatternInQueryLogic]\tFound time regex pattern: '{pattern}' in query")
                return True
        return False

    def _detect_keywords(self, query: str) -> bool:
        """Detect time-related keywords."""
        query_lower = query.lower()

        # Check language-specific keywords
        for category in self.time_keywords.values():
            for keyword in category:
                if keyword in query_lower:
                    logger.debug(f"[FindTimePatternInQueryLogic]\tFound time keyword: '{keyword}' in query")
                    return True

        return False

    def _detect_nlp_entities(self, query: str) -> bool:
        """Detect time entities using NLP."""
        if not self.nlp_processor:
            return False

        try:
            # Use comprehensive analysis to detect entities
            analysis = self.nlp_processor.analyze_text_comprehensive(query)

            # Check for date/time entities
            if analysis.entities:
                # Look for DATE, TIME, or similar entity types
                for entity in analysis.entities:
                    entity_type = entity.get('type', '').upper()
                    if any(keyword in entity_type for keyword in ['DATE', 'TIME', 'TEMPORAL', 'EVENT']):
                        logger.debug(f"[FindTimePatternInQueryLogic]\tFound NLP entity type: '{entity_type}' in query")
                        return True

            # Check if the query contains temporal keywords based on NLP analysis
            text_lower = analysis.cleaned_text.lower()

            # Use language-specific temporal indicators
            for indicator in self.temporal_indicators:
                if indicator in text_lower:
                    logger.debug(f"[FindTimePatternInQueryLogic]\tFound temporal indicator: '{indicator}' in query")
                    return True

        except Exception as e:
            logger.warning(f"NLP entity detection failed: {e}") 
        return False

    def _parallel_detection(self, query: str) -> List[bool]:
        """Run detection methods in parallel for optimization."""
        results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks
            future_regex = executor.submit(self._detect_regex_patterns, query)
            future_keywords = executor.submit(self._detect_keywords, query)
            future_nlp = executor.submit(self._detect_nlp_entities, query)

            # Collect results
            for future in as_completed([future_regex, future_keywords, future_nlp]):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.warning(f"Detection method failed: {e}")
                    results.append(False)
        return results

    def forward(self, query: str):
        """
        Detect time patterns and return tailored vectors for 8 workers.

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
            Vector: Probability distribution over 8 workers based on time pattern detection
        """
        logger.debug(f"[FindTimePatternInQueryLogic] Processing query: '{query}'")

        if not query or not isinstance(query, str):
            logger.debug(f"[FindTimePatternInQueryLogic] Invalid input, returning zero vector")
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        # Use parallel detection for optimization
        detection_results = self._parallel_detection(query)
        logger.debug(f"[FindTimePatternInQueryLogic] Parallel detection results: {detection_results}")

        # Break down detection results
        regex_detected, keywords_detected, nlp_detected = detection_results
        logger.debug(f"[FindTimePatternInQueryLogic] Regex patterns detected: {regex_detected}")
        logger.debug(f"[FindTimePatternInQueryLogic] Keywords detected: {keywords_detected}")
        logger.debug(f"[FindTimePatternInQueryLogic] NLP entities detected: {nlp_detected}")

        # Initialize result vector for 8 workers
        result = np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        # If any detection method found time patterns
        has_time_pattern = any(detection_results)
        logger.debug(f"[FindTimePatternInQueryLogic] Any time pattern detected: {has_time_pattern}")

        if has_time_pattern:
            # Time patterns detected - distribute weights based on detection methods
            detected_methods = []
            
            if regex_detected:
                detected_methods.append("regex")
                result[0][5] += 0.8  # TIME_IDENTIFIER_AGENT (primary for time patterns)
                result[0][7] += 0.2  # REMINDER_AGENT (time-based reminders)
            
            if keywords_detected:
                detected_methods.append("keywords")
                result[0][5] += 0.7  # TIME_IDENTIFIER_AGENT (time keywords)
                result[0][7] += 0.3  # REMINDER_AGENT (scheduling/reminders)
            
            if nlp_detected:
                detected_methods.append("nlp")
                result[0][5] += 0.6  # TIME_IDENTIFIER_AGENT (NLP time entities)
                result[0][1] += 0.4  # DOCS_SEARCHER_AGENT (contextual time info)

            logger.debug(f"[FindTimePatternInQueryLogic] Time patterns detected via: {detected_methods}")
        else:
            # No time patterns detected - return zero vector
            logger.debug(f"[FindTimePatternInQueryLogic] No time patterns detected -> zero vector")
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        # Normalize to ensure sum = 1
        if np.sum(result) > 0:
            result = result / np.sum(result)

        logger.debug(f"[FindTimePatternInQueryLogic] Final contribution - Query: '{query}' -> Detection methods: Regex={regex_detected}, Keywords={keywords_detected}, NLP={nlp_detected} -> Vector: {result.tolist()}")
        return result
        