from logic.base_classes import BaseLogic
from models import nlp_processor
import torch
import re
from typing import List, Dict, Any, Tuple
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
                logging.debug(f"Regex pattern {pattern} found in query: {query}")
                return True
        logging.debug(f"No regex pattern found in query: {query}")
        return False

    def _detect_keywords(self, query: str) -> bool:
        """Detect time-related keywords."""
        query_lower = query.lower()

        # Check language-specific keywords
        for category in self.time_keywords.values():
            for keyword in category:
                if keyword in query_lower:
                    logging.debug(f"Keyword \'{keyword}\' found in query: {query}")
                    return True

        logging.debug(f"No keyword found in query: {query}")
        return False

    def _detect_nlp_entities(self, query: str) -> bool:
        """Detect time entities using NLP."""
        if not self.nlp_processor:
            logging.debug(f"NLP processor not found in query: {query}")
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
                        logging.debug(f"Entity \'{entity_type}\' found in query: {query}")
                        return True

            # Check if the query contains temporal keywords based on NLP analysis
            text_lower = analysis.cleaned_text.lower()

            # Use language-specific temporal indicators
            for indicator in self.temporal_indicators:
                if indicator in text_lower:
                    logging.debug(f"Temporal indicator \'{indicator}\' found in query: {query}")
                    return True

        except Exception as e:
            logger.warning(f"NLP entity detection failed: {e}")
        logging.debug(f"No NLP entity found in query: {query}")
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
        logging.debug(f"Detection results: {results}")
        return results

    def forward(self, query: str) -> torch.Tensor:
        """
        Detect if the query contains time patterns.

        Args:
            query: Input query string

        Returns:
            torch.Tensor: 2D tensor [[1,1,0,0,0]] if time patterns detected, [[0,0,0,0,0]] otherwise
        """
        if not query or not isinstance(query, str):
            return torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float32)

        # Use parallel detection for optimization
        detection_results = self._parallel_detection(query)

        # If any detection method found time patterns, return 1
        has_time_pattern = any(detection_results)
        logging.debug(f"Has time pattern: {has_time_pattern}")
        # Return tensor with result
        if has_time_pattern:
            result = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.float32)
        else:
            result = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.float32)

        logger.debug(f"Query: '{query}' -> Time pattern detected: {has_time_pattern}")
        return result
        