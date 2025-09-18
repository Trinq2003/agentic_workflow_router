import logging
from typing import Dict, List, Any, Tuple, Optional

from ..base_classes import BaseLanguageProcessor, NLPConfig
from ..config import NLPConfigLoader
from ..nlp_models.spacy import SpacyModel
from ..nlp_models.nltk import NLTKModel

logger = logging.getLogger(__name__)


class EnglishProcessor(BaseLanguageProcessor):
    """
    English language processor using configurable NLP models.

    Uses the model specified in config (spaCy or NLTK) for English text processing.
    """

    def __init__(self, config: NLPConfig, model_config: Optional[Dict[str, Any]] = None):
        self.nlp_model = None
        self.model_config = model_config or NLPConfigLoader.load_config()
        super().__init__(config)

    def _initialize_models(self):
        """Initialize English NLP model based on configuration."""
        try:
            model_type = self.model_config.get("models", {}).get("english", "nltk").lower()

            if model_type == "spacy":
                self.nlp_model = SpacyModel(self.config, "en_core_web_sm")
            elif model_type == "nltk":
                self.nlp_model = NLTKModel(self.config)
            else:
                logger.warning(f"Unknown model type '{model_type}' for English, defaulting to NLTK")
                self.nlp_model = NLTKModel(self.config)

            if self.nlp_model and not self.nlp_model.is_available():
                logger.warning(f"Configured model {model_type} is not available for English")

        except Exception as e:
            logger.error(f"Error initializing English model: {e}")
            # Fallback to NLTK
            try:
                self.nlp_model = NLTKModel(self.config)
            except:
                self.nlp_model = None

    def tokenize(self, text: str) -> List[str]:
        """Tokenize English text."""
        if self.nlp_model:
            return self.nlp_model.tokenize(text)
        return text.split()

    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        """POS tagging for English text."""
        if self.nlp_model:
            return self.nlp_model.pos_tag(text)
        return [(token, 'UNK') for token in text.split()]

    def lemmatize(self, text: str) -> List[str]:
        """Lemmatize English text."""
        if self.nlp_model:
            return self.nlp_model.lemmatize(text)
        return text.split()

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from English text."""
        if self.nlp_model:
            return self.nlp_model.extract_entities(text)
        return []

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of English text."""
        if self.nlp_model:
            return self.nlp_model.analyze_sentiment(text)
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}

    def parse_dependencies(self, text: str) -> List[Dict[str, Any]]:
        """Dependency parsing for English text."""
        if self.nlp_model:
            return self.nlp_model.parse_dependencies(text)
        return []
