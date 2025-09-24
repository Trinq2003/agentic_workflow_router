import logging
from typing import Dict, List, Any, Tuple

from ..base_classes import BaseNLPModel, NLPConfig

logger = logging.getLogger(__name__)

# Check library availability
try:
    from underthesea import word_tokenize as vi_word_tokenize
    from underthesea import pos_tag as vi_pos_tag
    from underthesea import ner as vi_ner
    from underthesea import sentiment as vi_sentiment
    UNDERSEA_AVAILABLE = True
except ImportError:
    UNDERSEA_AVAILABLE = False

# Try to import dependency parser separately to allow graceful fallback on older versions
try:
    from underthesea import dependency_parse as vi_dependency_parse
    UNDERSEA_DEPENDENCY_AVAILABLE = True
except Exception:
    UNDERSEA_DEPENDENCY_AVAILABLE = False


class UndertheseaModel(BaseNLPModel):
    """Underthesea NLP model implementation."""

    def __init__(self, config: NLPConfig):
        super().__init__(config)

    def _initialize(self):
        """Initialize underthesea model."""
        if not UNDERSEA_AVAILABLE:
            logger.warning("Underthesea not available")
            return

        # underthesea models are loaded on-demand
        self._is_initialized = True
        logger.info("Underthesea model initialized successfully")

    def get_name(self) -> str:
        return "Underthesea"

    def tokenize(self, text: str) -> List[str]:
        if not UNDERSEA_AVAILABLE:
            return text.split()
        return vi_word_tokenize(text)

    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        if not UNDERSEA_AVAILABLE:
            return [(token, 'UNK') for token in text.split()]
        return vi_pos_tag(text)

    def lemmatize(self, text: str) -> List[str]:
        # Vietnamese doesn't have complex lemmatization
        return self.tokenize(text)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        entities = []
        if not UNDERSEA_AVAILABLE:
            return entities

        try:
            ner_result = vi_ner(text)
            # Format: [(word, pos_tag, chunk_tag, ner_tag), ...]
            current_entity = None
            current_start = -1

            for item in ner_result:
                if isinstance(item, tuple) and len(item) == 4:
                    word, pos_tag, chunk_tag, ner_tag = item

                    # Check if this is a named entity (not 'O')
                    if ner_tag != 'O':
                        # Extract entity type from BIO tag (B-PER -> PER, I-PER -> PER)
                        entity_type = ner_tag.split('-')[-1] if '-' in ner_tag else ner_tag

                        if ner_tag.startswith('B-'):  # Beginning of entity
                            # Save previous entity if exists
                            if current_entity:
                                entities.append(current_entity)

                            # Start new entity
                            current_start = text.find(word)
                            current_entity = {
                                'text': word,
                                'label': entity_type,
                                'start': current_start,
                                'end': current_start + len(word),
                            }

                        elif ner_tag.startswith('I-') and current_entity:  # Inside entity
                            # Extend current entity
                            current_entity['text'] += ' ' + word
                            current_entity['end'] = text.find(word, current_entity['start']) + len(word)

                    else:
                        # Save previous entity when we hit non-entity
                        if current_entity:
                            entities.append(current_entity)
                            current_entity = None

            # Don't forget to save the last entity
            if current_entity:
                entities.append(current_entity)

        except Exception as e:
            logger.warning(f"Underthesea NER failed: {e}")

        return entities

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        if not UNDERSEA_AVAILABLE:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}

        try:
            sentiment_result = vi_sentiment(text)
            if sentiment_result == 'positive':
                return {'compound': 0.5, 'pos': 0.5, 'neu': 0.5, 'neg': 0.0}
            elif sentiment_result == 'negative':
                return {'compound': -0.5, 'pos': 0.0, 'neu': 0.5, 'neg': 0.5}
            else:
                return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        except Exception as e:
            logger.warning(f"Underthesea sentiment failed: {e}")
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}

    def parse_dependencies(self, text: str) -> List[Dict[str, Any]]:
        """Dependency parsing using underthesea.dependency_parse.

        Returns list of dicts with keys: text, head, dep, pos.
        - head is mapped to the head token text (root -> token itself) for
          consistency with spaCy implementation.
        - pos is filled using underthesea.pos_tag when available; otherwise 'UNK'.
        """
        # Fallback if library or feature is unavailable
        logger.debug(f"Underthesea dependency parsing result: {UNDERSEA_AVAILABLE} {UNDERSEA_DEPENDENCY_AVAILABLE}")
        if not UNDERSEA_AVAILABLE or not UNDERSEA_DEPENDENCY_AVAILABLE:
            tokens = self.tokenize(text)
            return [{
                'text': token,
                'head': token,  # mimic spaCy behavior for root/self when unknown
                'dep': 'UNK',
                'pos': 'UNK',
            } for token in tokens]

        try:
            dep_result = vi_dependency_parse(text)
            logger.debug(f"Underthesea dependency parsing result: {dep_result}")
            # Attempt to get POS tags; if it fails, we'll use UNK
            try:
                pos_tags = vi_pos_tag(text)
            except Exception:
                pos_tags = []

            # Build mapping of index->pos using sequence alignment when lengths match
            index_to_pos: List[str] = []
            if isinstance(pos_tags, list) and len(pos_tags) == len(dep_result):
                index_to_pos = [tag for (_, tag) in pos_tags]
            else:
                index_to_pos = ['UNK'] * len(dep_result)

            dependencies: List[Dict[str, Any]] = []
            # dep_result items look like: (token_text, head_index, dep_label)
            # head_index is 1-based; 0 indicates root
            for i, item in enumerate(dep_result):
                try:
                    token_text, head_index, dep_label = item
                except Exception:
                    # Unexpected format; skip item
                    continue

                if isinstance(head_index, int) and head_index > 0 and head_index <= len(dep_result):
                    head_text = dep_result[head_index - 1][0]
                else:
                    # Root or invalid -> point to itself for consistency with spaCy
                    head_text = token_text

                pos_val = index_to_pos[i] if i < len(index_to_pos) else 'UNK'

                dependencies.append({
                    'text': token_text,
                    'head': head_text,
                    'dep': dep_label,
                    'pos': pos_val,
                })

            return dependencies
        except Exception as e:
            logger.warning(f"Underthesea dependency parsing failed, falling back: {e}")
            tokens = self.tokenize(text)
            return [{
                'text': token,
                'head': token,
                'dep': 'UNK',
                'pos': 'UNK',
            } for token in tokens]