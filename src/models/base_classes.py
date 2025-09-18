from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from .config import NLPConfig, Language

from langdetect import detect


class NLPTechnique(Enum):
    """Available NLP techniques."""
    TOKENIZATION = "tokenization"
    POS_TAGGING = "pos_tagging"
    LEMMATIZATION = "lemmatization"
    STEMMING = "stemming"
    NAMED_ENTITY_RECOGNITION = "ner"
    DEPENDENCY_PARSING = "dependency_parsing"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    LANGUAGE_DETECTION = "language_detection"


@dataclass
class NLPResult:
    """Result of NLP processing."""
    text: str
    language: Language
    tokens: List[str] = field(default_factory=list)
    pos_tags: List[Tuple[str, str]] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: Dict[str, float] = field(default_factory=dict)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    processing_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'text': self.text,
            'language': self.language.value,
            'tokens': self.tokens,
            'pos_tags': self.pos_tags,
            'lemmas': self.lemmas,
            'entities': self.entities,
            'sentiment': self.sentiment,
            'dependencies': self.dependencies,
            'processing_time': self.processing_time,
        }


@dataclass
class TextAnalysis:
    """Comprehensive text analysis result."""
    original_text: str
    cleaned_text: str
    language: Language
    word_count: int
    sentence_count: int
    avg_word_length: float
    sentiment_score: float
    entities: List[Dict[str, Any]]
    keywords: List[str]
    topics: List[str]
    grammar_issues: List[str]
    processing_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'original_text': self.original_text,
            'cleaned_text': self.cleaned_text,
            'language': self.language.value,
            'word_count': self.word_count,
            'sentence_count': self.sentence_count,
            'avg_word_length': self.avg_word_length,
            'sentiment_score': self.sentiment_score,
            'entities': self.entities,
            'keywords': self.keywords,
            'topics': self.topics,
            'grammar_issues': self.grammar_issues,
            'processing_time': self.processing_time,
        }


class LanguageDetector:
    """Language detection utility."""

    @staticmethod
    def detect_language(text: str) -> Language:
        """Detect the language of the given text."""
        if not text or len(text.strip()) < 3:
            return Language.UNKNOWN
        
        detected = detect(text)
        if detected == 'en':
            return Language.ENGLISH
        elif detected == 'vi':
            return Language.VIETNAMESE
        else:
            return Language.UNKNOWN


class BaseLanguageProcessor(ABC):
    """
    Abstract base class for language-specific NLP processors.

    This class defines the interface that all language processors must implement,
    ensuring consistency and enabling polymorphism.
    """

    def __init__(self, config: NLPConfig):
        self.config = config
        self._initialize_models()

    @abstractmethod
    def _initialize_models(self):
        """Initialize language-specific NLP models."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        pass

    @abstractmethod
    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        """Perform part-of-speech tagging."""
        pass

    @abstractmethod
    def lemmatize(self, text: str) -> List[str]:
        """Lemmatize words to their base forms."""
        pass

    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        pass

    @abstractmethod
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of the text."""
        pass

    @abstractmethod
    def parse_dependencies(self, text: str) -> List[Dict[str, Any]]:
        """Parse syntactic dependencies."""
        pass

    def get_supported_features(self) -> List[str]:
        """Get list of supported NLP features."""
        return [
            "tokenization",
            "pos_tagging",
            "lemmatization",
            "entity_extraction",
            "sentiment_analysis",
            "dependency_parsing"
        ]

    def is_available(self) -> bool:
        """Check if the processor is properly initialized."""
        return True

class BaseNLPModel(ABC):
    """
    Abstract base class for NLP model implementations.

    Each NLP library (spaCy, NLTK, underthesea) should have its own implementation
    inheriting from this class.
    """

    def __init__(self, config: NLPConfig):
        self.config = config
        self._is_initialized = False
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """Initialize the NLP model."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this NLP model."""
        pass

    def is_available(self) -> bool:
        """Check if this model is properly initialized and available."""
        return self._is_initialized

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        pass

    @abstractmethod
    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        """POS tagging."""
        pass

    @abstractmethod
    def lemmatize(self, text: str) -> List[str]:
        """Lemmatization."""
        pass

    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Named entity recognition."""
        pass

    @abstractmethod
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Sentiment analysis."""
        pass

    @abstractmethod
    def parse_dependencies(self, text: str) -> List[Dict[str, Any]]:
        """Dependency parsing."""
        pass