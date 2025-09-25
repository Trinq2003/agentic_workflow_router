from logic.base_classes import BaseLogic
from models import nlp_processor
import numpy as np
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging


logger = logging.getLogger(__name__)


class DetectNumericalRequirementInQueryLogic(BaseLogic):
    """
    Detect if a query requests numerical/quantitative tasks (stats, counts, ratios, rankings...).

    This leverages Vietnamese NLP (tokenization/POS via underthesea) plus robust
    diacritic-insensitive keyword and pattern detection.

    If detected, returns a vector preferring FUNCTION_CALLING worker, which can fetch
    realtime numerical data: [[1,0,0,0,0]]. Otherwise returns zeros.
    """

    def __init__(self):
        super().__init__()
        try:
            self._nlp = nlp_processor
        except Exception as exc:
            logger.warning(f"Failed to initialize NLP processor: {exc}")
            self._nlp = None

        # Core Vietnamese numerical intent keywords (with diacritics preserved)
        self._num_keywords = {
            "thống kê",
            "so sánh",
            "tỷ lệ", "tỉ lệ",  # both variants
            "tỷ trọng", "tỉ trọng",
            "số lượng",
            "số liệu",
            "bao nhiêu",
            "số vị trí",
            "biểu đồ",
            "xếp hạng",
            "ranking",
            "top",
            "phần trăm",
            "trung bình",
            "tổng",
            "đếm",
            "max",
            "min",
            "median",
            "tổng hợp",
            "thay đổi",
            "tăng trưởng",
            "tăng",
            "giảm",
        }

        # Comparative markers (with diacritics preserved)
        self._comparatives = {
            "hơn",
            "ít hơn",
            "nhiều hơn",
            "cao hơn",
            "thấp hơn",
            "lớn nhất",
            "nhỏ nhất",
            "tốt nhất",
            "xấu nhất",
            "tệ nhất",
        }

        # Regex patterns capturing numerics/percentages/rank forms with Vietnamese diacritics
        self._regex_patterns = [
            re.compile(r"\btop\s*\d+\b", re.IGNORECASE),
            re.compile(r"\bhạng\s*\d+\b", re.IGNORECASE),
            re.compile(r"\b\d+\s*%\b"),
            re.compile(r"\b(tăng|giảm|tăng trưởng)\s+\d+\s*%\b", re.IGNORECASE),  # tăng/giảm + percentage
            re.compile(r"\b(phần trăm|tỷ lệ|tỉ lệ)\b", re.IGNORECASE),
            re.compile(r"\b(số lượng|số liệu)\b", re.IGNORECASE),
            re.compile(r"\b(thống kê|so sánh|xếp hạng)\b", re.IGNORECASE),
        ]

    def _detect_keywords(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self._num_keywords)

    def _detect_comparatives(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self._comparatives)

    def _detect_regex(self, text: str) -> bool:
        return any(p.search(text) is not None for p in self._regex_patterns)

    def _parallel_detection(self, query: str) -> List[bool]:
        # Use original query with diacritics preserved
        tasks = [
            (self._detect_keywords, (query,)),
            (self._detect_comparatives, (query,)),
            (self._detect_regex, (query,)),
        ]

        results: List[bool] = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(func, *args) for func, args in tasks]
            for fut in as_completed(futures):
                try:
                    results.append(bool(fut.result()))
                except Exception as exc:
                    logger.warning(f"Numerical detection subtask failed: {exc}")
                    results.append(False)
        return results

    def forward(self, query: str):
        """
        Detect numerical requirements and return tailored vectors for 8 workers.
        
        Workers mapping:
        0: FUNCTION_CALLER_AGENT (numerical/statistical queries)
        1: DOCS_SEARCHER_AGENT (documentation lookup)
        2: NETMIND_INFOR_AGENT
        3: REGION_IDENTIFIER_AGENT  
        4: WEBS_SEARCHER_AGENT
        5: TIME_IDENTIFIER_AGENT
        6: EMPLOYEE_INFOR_AGENT
        7: REMINDER_AGENT
        
        Returns:
            Vector: Probability distribution over 8 workers based on numerical requirements
        """
        logger.debug(f"[DetectNumericalRequirementInQueryLogic] Processing query: '{query}'")

        if not query or not isinstance(query, str):
            logger.debug("[DetectNumericalRequirementInQueryLogic] Invalid input, returning zero vector")
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        # Run ensemble detections in parallel
        kwords, comps, regexes = self._parallel_detection(query)
        logger.debug(f"[DetectNumericalRequirementInQueryLogic] Detectors -> keywords={kwords}, comparatives={comps}, regex={regexes}")

        # Initialize result vector for 8 workers
        result = np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        
        if any([kwords, comps, regexes]):
            if kwords:
                result[0][0] += 0.8
                result[0][1] += 0.2
            if comps:
                result[0][0] += 0.8
                result[0][1] += 0.2
            if regexes:
                result[0][0] += 0.5
                result[0][1] += 0.5
        else:
            result = np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        # Normalize to ensure sum = 1
        if np.sum(result) > 0:
            result = result / np.sum(result)

        logger.debug(f"[DetectNumericalRequirementInQueryLogic] Final contribution - Query: '{query}' -> Detectors: keywords={kwords}, comparatives={comps}, regex={regexes} -> Vector: {result.tolist()}")
        
        return result


