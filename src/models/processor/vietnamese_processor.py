import logging
from typing import Dict, List, Any, Tuple, Optional

from ..base_classes import BaseLanguageProcessor, NLPConfig
from ..config import NLPConfigLoader
from ..nlp_models.underthesea import UndertheseaModel

logger = logging.getLogger(__name__)


class VietnameseProcessor(BaseLanguageProcessor):
    """
    Vietnamese language processor using configurable NLP models.

    Uses the model specified in config (currently underthesea) for Vietnamese text processing.
    """

    def __init__(self, config: NLPConfig, model_config: Optional[Dict[str, Any]] = None):
        self.nlp_model = None
        self.model_config = model_config or NLPConfigLoader.load_config()
        super().__init__(config)

    def _initialize_models(self):
        """Initialize Vietnamese NLP model based on configuration."""
        try:
            model_type = self.model_config.get("models", {}).get("vietnamese", "underthesea").lower()

            if model_type == "underthesea":
                self.nlp_model = UndertheseaModel(self.config)
            else:
                logger.warning(f"Unknown model type '{model_type}' for Vietnamese, defaulting to underthesea")
                self.nlp_model = UndertheseaModel(self.config)

            if self.nlp_model and not self.nlp_model.is_available():
                logger.warning(f"Configured model {model_type} is not available for Vietnamese")

        except Exception as e:
            logger.error(f"Error initializing Vietnamese model: {e}")
            self.nlp_model = None

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Vietnamese text."""
        if self.nlp_model:
            return self.nlp_model.tokenize(text)
        return text.split()

    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        """POS tagging for Vietnamese text."""
        if self.nlp_model:
            return self.nlp_model.pos_tag(text)
        return [(token, 'UNK') for token in text.split()]

    def lemmatize(self, text: str) -> List[str]:
        """Lemmatize Vietnamese text (basic implementation)."""
        # Vietnamese doesn't have complex lemmatization like English
        return self.tokenize(text)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from Vietnamese text."""
        if self.nlp_model:
            return self.nlp_model.extract_entities(text)
        return []

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of Vietnamese text."""
        if self.nlp_model:
            return self.nlp_model.analyze_sentiment(text)
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}

    def parse_dependencies(self, text: str) -> List[Dict[str, Any]]:
        """Dependency parsing for Vietnamese text."""
        if self.nlp_model:
            return self.nlp_model.parse_dependencies(text)
        return []
