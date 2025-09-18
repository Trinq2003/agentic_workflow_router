import logging
from typing import Dict, List, Any, Tuple

from ..base_classes import BaseNLPModel, NLPConfig

logger = logging.getLogger(__name__)

# Check library availability
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


class NLTKModel(BaseNLPModel):
    """NLTK NLP model implementation."""

    def __init__(self, config: NLPConfig):
        self.lemmatizer = None
        self.sentiment_analyzer = None
        super().__init__(config)

    def _initialize(self):
        """Initialize NLTK components."""
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available")
            return

        try:
            # Download required NLTK data
            required_packages = [
                'tokenizers/punkt',
                'corpora/wordnet',
                'taggers/averaged_perceptron_tagger'
            ]

            for package in required_packages:
                try:
                    nltk.data.find(package)
                except LookupError:
                    nltk.download(package.split('/')[-1], quiet=True)

            self.lemmatizer = WordNetLemmatizer()
            if VADER_AVAILABLE:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()

            self._is_initialized = True
            logger.info("NLTK model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing NLTK: {e}")

    def get_name(self) -> str:
        return "NLTK"

    def tokenize(self, text: str) -> List[str]:
        if not NLTK_AVAILABLE:
            return text.split()
        return word_tokenize(text)

    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        if not NLTK_AVAILABLE:
            return [(token, 'UNK') for token in text.split()]
        tokens = word_tokenize(text)
        return pos_tag(tokens)

    def lemmatize(self, text: str) -> List[str]:
        if not NLTK_AVAILABLE or not self.lemmatizer:
            return text.split()
        tokens = word_tokenize(text)
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        entities = []
        if not NLTK_AVAILABLE:
            return entities

        try:
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            tree = ne_chunk(tagged)

            for subtree in tree:
                if hasattr(subtree, 'label'):
                    entity_text = ' '.join([token for token, pos in subtree])
                    entities.append({
                        'text': entity_text,
                        'label': subtree.label(),
                        'start': text.find(entity_text),
                        'end': text.find(entity_text) + len(entity_text),
                    })
        except Exception as e:
            logger.warning(f"NLTK NER failed: {e}")

        return entities

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        if VADER_AVAILABLE and self.sentiment_analyzer:
            return self.sentiment_analyzer.polarity_scores(text)
        elif TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            return {
                'compound': blob.sentiment.polarity,
                'pos': max(0, blob.sentiment.polarity),
                'neu': 1 - abs(blob.sentiment.polarity),
                'neg': max(0, -blob.sentiment.polarity)
            }
        else:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}

    def parse_dependencies(self, text: str) -> List[Dict[str, Any]]:
        # NLTK doesn't provide dependency parsing
        tokens = self.tokenize(text)
        return [{
            'text': token,
            'head': '',
            'dep': 'UNK',
            'pos': 'UNK',
        } for token in tokens]
