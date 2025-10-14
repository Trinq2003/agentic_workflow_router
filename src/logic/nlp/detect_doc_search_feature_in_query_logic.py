from logic.base_classes import BaseLogic
import numpy as np
import re
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class DetectDocSearchFeatureInQueryLogic(BaseLogic):
    """
    Detects document-search intent and question patterns in Vietnamese queries.

    Detection focuses on:
    1) Document keywords (categorized as nouns vs verbs)
    2) Special asking keywords at the start or end of the question
    3) Short-form definition questions (e.g., "X là gì", "X như thế nào")

    Output is a vector over 8 workers (see config/workers.yaml), prioritizing
    DOCS_SEARCHER_AGENT (index 1) when any signal is detected.
    """

    def __init__(self):
        super().__init__()

        # Categorized document-related keywords
        self.doc_nouns = [
            "văn bản", "tài liệu", "quy trình", "định nghĩa", "công thức",
            "giải pháp", "hướng dẫn", "thông số", "cơ sở", "cấu trúc",
            "tần suất", "mô tả", "lưu đồ", "kiến trúc", "cơ chế",
            "quy hoạch", "quy tắc", "tính năng", "ngưỡng", "quan điểm",
            "câu lệnh", "phương pháp", "yêu cầu", "trách nhiệm", "trường hợp",
            "quyết định", "phụ lục", "công cụ", "mã hiệu", "nội dung"
        ]

        self.doc_verbs = [
            "ban hành", "phê duyệt", "xử lý", "nào để"
        ]

        # Question endings and starts
        self.question_endings = [
            "là gì", "như thế nào", "là", "bao lâu", "để làm gì", "lỗi nào", "lỗi gì"
        ]
        self.question_starts = [
            "liệt kê", "tại sao", "giải thích", "khuyến nghị", "làm thế nào",
            "các bước", "nên", "có nên", "ai", "người"
        ]

        logger.info("Initialized DetectDocSearchFeatureInQueryLogic")

    def _contains_any_phrase(self, text: str, phrases: List[str]) -> Tuple[bool, List[str]]:
        text_lower = text.lower()
        hits: List[str] = []
        for phrase in phrases:
            if phrase in text_lower:
                hits.append(phrase)
        return (len(hits) > 0, hits)

    def _detect_doc_keywords(self, query: str) -> Tuple[bool, bool, List[str], List[str]]:
        """Return (has_nouns, has_verbs, noun_hits, verb_hits)."""
        has_nouns, noun_hits = self._contains_any_phrase(query, self.doc_nouns)
        has_verbs, verb_hits = self._contains_any_phrase(query, self.doc_verbs)

        # Secondary logging for matched keywords
        if has_nouns:
            for hit in noun_hits:
                logger.debug(f"[DetectDocSearchFeatureInQueryLogic]\tFound doc noun: '{hit}' in query")
        if has_verbs:
            for hit in verb_hits:
                logger.debug(f"[DetectDocSearchFeatureInQueryLogic]\tFound doc verb: '{hit}' in query")
        return has_nouns, has_verbs, noun_hits, verb_hits

    def _detect_question_start(self, query: str) -> Tuple[bool, str]:
        q = query.strip().lower()
        for starter in self.question_starts:
            if q.startswith(starter + " ") or q == starter:
                logger.debug(f"[DetectDocSearchFeatureInQueryLogic]\tFound question start: '{starter}' in query")
                return True, starter
        return False, ""

    def _detect_question_end(self, query: str) -> Tuple[bool, str]:
        q = query.strip().lower()
        for ending in self.question_endings:
            # allow optional punctuation at the end
            pattern = rf"{re.escape(ending)}\s*[?.!]*$"
            if re.search(pattern, q):
                logger.debug(f"[DetectDocSearchFeatureInQueryLogic]\tFound question end: '{ending}' in query")
                return True, ending
        return False, ""

    def _detect_short_definition(self, query: str) -> bool:
        q = query.strip().lower()
        # Common short forms: "X là gì", "X như thế nào"
        if re.search(r"\b(là gì|như thế nào)\b", q):
            logger.debug("[DetectDocSearchFeatureInQueryLogic]\tFound short definition pattern in query")
            return True
        return False

    def _is_netmind_definition_question(self, query: str) -> bool:
        """Return True if query is of the form 'NetMind là gì' (case-insensitive)."""
        q = query.strip().lower()
        # Match both 'netmind' and 'net mind' directly preceding 'là gì'
        pattern = r"(?:\bnet\s*mind\b)"
        return re.search(pattern, q, flags=re.IGNORECASE) is not None

    def _parallel_detection(self, query: str) -> Tuple[Tuple[bool, bool, List[str], List[str]], Tuple[bool, str], Tuple[bool, str], bool]:
        results = [None, None, None, None]  # type: ignore
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._detect_doc_keywords, query): 0,
                executor.submit(self._detect_question_start, query): 1,
                executor.submit(self._detect_question_end, query): 2,
                executor.submit(self._detect_short_definition, query): 3,
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    logger.warning(f"Doc-search detection subtask failed at {idx}: {exc}")
                    # Fill with safe defaults
                    if idx == 0:
                        results[idx] = (False, False, [], [])
                    elif idx in (1, 2):
                        results[idx] = (False, "")
                    else:
                        results[idx] = False
        return results[0], results[1], results[2], results[3]

    def forward(self, query: str):
        """
        Return a normalized 1x8 vector prioritizing DOCS_SEARCHER_AGENT when document
        signals or question patterns are detected.
        """
        logger.debug(f"[DetectDocSearchFeatureInQueryLogic] Processing query: '{query}'")

        if not query or not isinstance(query, str):
            logger.debug("[DetectDocSearchFeatureInQueryLogic] Invalid input -> zero vector")
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        (has_nouns, has_verbs, noun_hits, verb_hits), (start_hit, start_kw), (end_hit, end_kw), short_def = self._parallel_detection(query)

        logger.debug(
            f"[DetectDocSearchFeatureInQueryLogic] nouns={has_nouns} {noun_hits}, verbs={has_verbs} {verb_hits}, "
            f"qstart={start_hit} '{start_kw}', qend={end_hit} '{end_kw}', short_def={short_def}"
        )

        result = np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        any_signal = has_nouns or has_verbs or start_hit or end_hit or short_def
        if not any_signal:
            logger.debug("[DetectDocSearchFeatureInQueryLogic] No doc-search signal -> zero vector")
            return result

        # Prioritize DOCS_SEARCHER_AGENT (index 1)
        if has_nouns:
            result[0][1] += 1
        if has_verbs:
            result[0][1] += 1
        if start_hit:
            result[0][1] += 1
        if end_hit or short_def:
            if not self._is_netmind_definition_question(query):
                result[0][1] += 1
            else:
                logger.debug("[DetectDocSearchFeatureInQueryLogic] Skipping DOCS weight for 'NetMind' pattern")

        # Normalize
        if np.sum(result) > 0:
            result = result / np.sum(result)

        logger.debug(
            f"[DetectDocSearchFeatureInQueryLogic] Final vector: {result.tolist()}"
        )
        return result