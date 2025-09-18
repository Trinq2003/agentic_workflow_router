import logging
from typing import Dict, List, Any, Tuple

from ..base_classes import BaseNLPModel, NLPConfig

logger = logging.getLogger(__name__)

# Check library availability
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class SpacyModel(BaseNLPModel):
    """spaCy NLP model implementation."""

    def __init__(self, config: NLPConfig, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        super().__init__(config)

    def _initialize(self):
        """Initialize spaCy model."""
        if not SPACY_AVAILABLE:
            logger.warning("spaCy not available")
            return

        try:
            self.nlp = spacy.load(self.model_name)
            self._is_initialized = True
            logger.info(f"spaCy model {self.model_name} loaded successfully")
        except OSError:
            logger.warning(f"spaCy model {self.model_name} not found")
        except Exception as e:
            logger.error(f"Error loading spaCy model: {e}")

    def get_name(self) -> str:
        return f"spaCy ({self.model_name})"

    def tokenize(self, text: str) -> List[str]:
        if not self.nlp:
            return text.split()
        doc = self.nlp(text)
        return [token.text for token in doc]

    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        if not self.nlp:
            return [(token, 'UNK') for token in text.split()]
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]

    def lemmatize(self, text: str) -> List[str]:
        if not self.nlp:
            return text.split()
        doc = self.nlp(text)
        return [token.lemma_ for token in doc]

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        entities = []
        if not self.nlp:
            return entities

        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
            })
        return entities

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        # spaCy doesn't have built-in sentiment analysis
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}

    def parse_dependencies(self, text: str) -> List[Dict[str, Any]]:
        dependencies = []
        if not self.nlp:
            return dependencies

        doc = self.nlp(text)
        for token in doc:
            dependencies.append({
                'text': token.text,
                'head': token.head.text,
                'dep': token.dep_,
                'pos': token.pos_,
            })
        return dependencies
