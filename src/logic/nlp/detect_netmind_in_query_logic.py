from logic.base_classes import BaseLogic
from models import nlp_processor, NLPTechnique
import numpy as np
import re
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class DetectNetmindInQueryLogic(BaseLogic):
    """
    Detect queries that ask about the assistant (Netmind) itself and should be
    routed to NETMIND_INFOR_AGENT (index 2).

    Criteria:
    1) Query mentions "netmind" OR uses 2nd-person pronouns aimed at the bot
       ("bạn", "mày" but not in "bố mày", "cậu", "ngươi").
    2) The question is simple/brief, typically definition/ability/intro/team.
    """

    def __init__(self):
        super().__init__()
        logger.info("Initialized DetectNetmindInQueryLogic (NETMIND_INFOR_AGENT detector)")

    def _has_netmind_keyword(self, query: str) -> bool:
        if not query or not isinstance(query, str):
            return False
        q = query.lower()
        # Match 'netmind' as a word or token in text (case-insensitive)
        return re.search(r"\bnet\s*mind\b|\bnetmind\b", q) is not None

    def _has_bot_pronoun(self, query: str) -> bool:
        if not query or not isinstance(query, str):
            return False
        q = query.lower()

        pronoun_patterns = [
            r"\bbạn\b",
            r"\bcậu\b",
            r"\bngươi\b",
            r"\bmày\b",
        ]

        basic_pronoun = any(re.search(p, q) for p in pronoun_patterns)
        if basic_pronoun:
            return True

        # Special handling: "đồng chí" counts only if NOT followed by a PERSON name
        return self._has_comrade_not_followed_by_person(query)

    def _has_comrade_not_followed_by_person(self, query: str) -> bool:
        if not query or not isinstance(query, str):
            return False

        # Find all occurrences of "đồng chí" with word boundaries
        matches = list(re.finditer(r"(?<!\w)đồng\s+chí(?!\w)", query, flags=re.IGNORECASE))
        if not matches:
            return False

        try:
            ner_result = nlp_processor.process_text(query, techniques=[NLPTechnique.NAMED_ENTITY_RECOGNITION])
            entities = ner_result.entities or []
        except Exception as e:
            logger.warning(f"NER failed while checking 'đồng chí' context: {e}")
            entities = []

        # Normalize entity labels and ensure start offsets are integers
        normalized_entities = []
        for ent in entities:
            try:
                start = int(ent.get('start')) if ent.get('start') is not None else None
                label = str(ent.get('label', '')).upper()
                if start is not None:
                    normalized_entities.append({'start': start, 'label': label})
            except Exception:
                continue

        # Helper to check if a PERSON entity begins at or immediately after index i
        def followed_by_person(start_index: int) -> bool:
            i = start_index
            text = query

            # Skip whitespace
            while i < len(text) and text[i].isspace():
                i += 1

            # Skip light punctuation then spaces (e.g., ":", ",", "-", quotes)
            while i < len(text) and text[i] in [':', ',', '-', '—', '–', '.', '…']:
                i += 1
                while i < len(text) and text[i].isspace():
                    i += 1

            # Skip opening quotes if present
            if i < len(text) and text[i] in ['"', '“', '”', "'", '‘', '’']:
                i += 1
                while i < len(text) and text[i].isspace():
                    i += 1

            for ent in normalized_entities:
                if ent['label'] in ('PER', 'PERSON') and ent['start'] >= i and ent['start'] - i <= 1:
                    return True
            return False

        # If any occurrence of "đồng chí" is NOT followed by a PERSON, treat as bot pronoun
        for m in matches:
            after = m.end()
            if not followed_by_person(after):
                return True

        return False

    def _is_simple_question(self, query: str) -> bool:
        if not query or not isinstance(query, str):
            return False

        q = query.lower().strip()

        # Quick length heuristic
        token_count = len(q.split())
        if token_count > 16:
            # Long queries are unlikely to be simple info about Netmind
            return False

        # Core simple-intent patterns
        simple_patterns = [
            r"\blà\s+gì\b",              # what is
            r"\blà\s+ai\b",              # who is
            r"\blàm\s+được\s+gì\b",     # what can do
            r"\blàm\s+gì\b",             # do what
            r"\bcó\s+thể\s+làm\s+gì\b", # what can you do
            r"\bai\s+phát\s+triển\b",   # who developed
            r"\bđội\s+ngũ\b",            # team
            r"\bphát\s+triển\b",         # develop (paired with netmind/anchor)
            r"\bgiới\s+thiệu\b",         # introduction
            r"\btính\s+năng\b",          # features
            r"\bchức\s+năng\b",          # functions
            r"\bsứ\s+mệnh\b",            # mission
            r"\bmục\s+tiêu\b",           # goal
        ]

        if any(re.search(p, q) for p in simple_patterns):
            return True

        # Fallback: short direct question with a question mark
        if "?" in q and token_count <= 12:
            return True

        return False

    def _parallel_detection(self, query: str) -> List[bool]:
        """Run detector methods in parallel and return [has_netmind, has_bot_pronoun, is_simple]."""
        results = [False, False, False]

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._has_netmind_keyword, query): 0,
                executor.submit(self._has_bot_pronoun, query): 1,
                executor.submit(self._is_simple_question, query): 2,
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Detection method at index {idx} failed: {e}")
                    results[idx] = False

        return results

    def forward(self, query: str):
        """
        Return an 8-dim vector highlighting NETMIND_INFOR_AGENT (index 2) when:
        (has_netmind_keyword OR has_bot_pronoun) AND is_simple_question.

        Workers mapping (from config/workers.yaml):
        0: FUNCTION_CALLER_AGENT
        1: DOCS_SEARCHER_AGENT
        2: NETMIND_INFOR_AGENT
        3: REGION_IDENTIFIER_AGENT
        4: WEBS_SEARCHER_AGENT
        5: TIME_IDENTIFIER_AGENT
        6: EMPLOYEE_INFOR_AGENT
        7: REMINDER_AGENT
        """
        logger.debug(f"[DetectNetmindInQueryLogic] Processing query: '{query}'")

        if not query or not isinstance(query, str):
            logger.debug("[DetectNetmindInQueryLogic] Invalid input -> zero vector")
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        has_netmind, has_pronoun, is_simple = self._parallel_detection(query)
        logger.debug(f"[DetectNetmindInQueryLogic] has_netmind={has_netmind}, has_pronoun={has_pronoun}, is_simple={is_simple}")

        result = np.array([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

        anchor_ok = has_netmind or has_pronoun
        if anchor_ok and is_simple:
            result[0][2] = 1.0
        else:
            logger.debug("[DetectNetmindInQueryLogic] Conditions not met -> zero vector")
            return result

        # Already one-hot; keep normalization step for consistency
        if np.sum(result) > 0:
            result = result / np.sum(result)

        logger.debug(f"[DetectNetmindInQueryLogic] Final vector: {result.tolist()}")
        return result
