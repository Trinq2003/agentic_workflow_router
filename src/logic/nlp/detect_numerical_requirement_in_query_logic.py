from logic.base_classes import BaseLogic
from models import nlp_processor
import numpy as np
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from logic.nlp.utils import normalize_text

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

        # Core Vietnamese numerical intent keywords (normalized, lowercase, no diacritics)
        self._num_keywords = {
            "thong ke",
            "so sanh",
            "ti le",  # tỷ lệ / tỉ lệ -> ti le
            "ti trong",
            "so luong",
            "so lieu",
            "bao nhieu",
            "so vi tri",
            "bieu do",
            "xep hang",
            "ranking",
            "top",
            "phan tram",
            "trung binh",
            "tong",
            "dem",
            "max",
            "min",
            "median",
            "tong hop",
            "thay doi",
            "tang truong",
            "giam",
        }

        # Comparative markers (normalized)
        self._comparatives = {
            "hon",            # hơn
            "it hon",
            "nhieu hon",
            "cao hon",
            "thap hon",
            "lon nhat",
            "nho nhat",
            "tot nhat",
            "xau nhat",
        }

        # Regex patterns capturing numerics/percentages/rank forms in normalized text
        self._regex_patterns = [
            re.compile(r"\btop\s*\d+\b"),
            re.compile(r"\bhang\s*\d+\b"),
            re.compile(r"\b\d+\s*%\b"),
            re.compile(r"\b(phan tram|ti le)\b"),
            re.compile(r"\b(so luong|so lieu)\b"),
        ]

    def _detect_keywords(self, normalized_text: str) -> bool:
        return any(kw in normalized_text for kw in self._num_keywords)

    def _detect_comparatives(self, normalized_text: str) -> bool:
        return any(kw in normalized_text for kw in self._comparatives)

    def _detect_regex(self, normalized_text: str) -> bool:
        return any(p.search(normalized_text) is not None for p in self._regex_patterns)

    def _detect_pos_numeric(self, text: str) -> bool:
        """Use Vietnamese POS to detect numerals/quantifiers presence with nouns/metrics."""
        try:
            if not self._nlp:
                return False
            # Tokenization + POS
            result = self._nlp.process_text(text)
            pos_tags = result.pos_tags or []
            # Heuristic: presence of numerals with adjacent nouns or measure words
            # underthesea often tags numbers as 'M'. We'll also fall back to digits regex.
            has_digit = bool(re.search(r"\d", text))

            has_numeral_tag = any(tag.upper().startswith("M") for _, tag in pos_tags)
            has_noun = any(tag.upper().startswith("N") for _, tag in pos_tags)

            return (has_numeral_tag and has_noun) or (has_digit and has_noun)
        except Exception as exc:
            logger.warning(f"POS-based numeric detection failed: {exc}")
            return False

    def _parallel_detection(self, query: str) -> List[bool]:
        normalized = normalize_text(query)
        tasks = [
            (self._detect_keywords, (normalized,)),
            (self._detect_comparatives, (normalized,)),
            (self._detect_regex, (normalized,)),
            (self._detect_pos_numeric, (query,)),
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
        Return [[1,0,0,0,0]] if numerical requirement is detected, else zeros.
        """
        logger.debug(f"[DetectNumericalRequirementInQueryLogic] Processing query: '{query}'")

        if not query or not isinstance(query, str):
            logger.debug("[DetectNumericalRequirementInQueryLogic] Invalid input, returning zero vector")
            return np.array([[0, 0, 0, 0, 0]], dtype=np.float32)

        # Run ensemble detections in parallel
        kwords, comps, regexes, posnum = self._parallel_detection(query)
        logger.debug(f"[DetectNumericalRequirementInQueryLogic] Detectors -> keywords={kwords}, comparatives={comps}, regex={regexes}, pos_numeric={posnum}")

        # Decision rule: any signal indicates a numerical request
        is_numerical = any([kwords, comps, regexes, posnum])

        if is_numerical:
            result = np.array([[1, 0, 0, 0, 0]], dtype=np.float32)
            logger.debug(f"[DetectNumericalRequirementInQueryLogic] Output vector: {result.tolist()} (numerical detected)")
        else:
            result = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)
            logger.debug(f"[DetectNumericalRequirementInQueryLogic] Output vector: {result.tolist()} (no numerical signal)")

        return result


