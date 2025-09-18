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
            for item in ner_result:
                if isinstance(item, tuple) and len(item) == 2:
                    word, label = item
                    if label != 'O':  # Not 'Outside'
                        entities.append({
                            'text': word,
                            'label': label,
                            'start': text.find(word),
                            'end': text.find(word) + len(word),
                        })
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
        # underthesea doesn't provide dependency parsing
        tokens = self.tokenize(text)
        return [{
            'text': token,
            'head': '',
            'dep': 'UNK',
            'pos': 'UNK',
        } for token in tokens]
