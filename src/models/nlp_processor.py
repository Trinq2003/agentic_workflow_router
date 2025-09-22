"""
Optimized NLP Wrapper for NetMind Workflow

This module provides a clean, focused NLP interface supporting:
- Lexical Analysis (tokenization, POS tagging, lemmatization)
- Syntactic Analysis (dependency parsing)
- Semantic Analysis (sentiment analysis, named entity recognition)
- Language Detection and Processing (English via spaCy/NLTK, Vietnamese via underthesea)

Built with proper OOP principles using abstract base classes and inheritance.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import base classes
from .base_classes import (
    BaseLanguageProcessor,
    NLPResult,
    NLPTechnique,
    LanguageDetector,
    TextAnalysis
)

# Import config
from .config import NLPConfig, NLPConfigLoader, Language

# Import processors
from .processor.english_processor import EnglishProcessor
from .processor.vietnamese_processor import VietnameseProcessor

# Check library availability for TextPreprocessor
try:
    import nltk
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)
class TextPreprocessor:
    """Text preprocessing utilities."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\'"]', '', text)

        return text

    @staticmethod
    def remove_stopwords(tokens: List[str], language: Language) -> List[str]:
        """Remove stopwords from token list."""
        if not NLTK_AVAILABLE:
            return tokens

        try:
            if language == Language.ENGLISH:
                stop_words = set(stopwords.words('english'))
            elif language == Language.VIETNAMESE:
                stop_words = set([
                    'là', 'và', 'của', 'có', 'được', 'cho', 'trong', 'với',
                    'lên', 'đến', 'từ', 'qua', 'theo', 'như', 'nếu', 'khi',
                    'thì', 'mà', 'để', 'ở', 'vì', 'nên', 'các', 'những'
                ])
            else:
                return tokens

            return [token for token in tokens if token.lower() not in stop_words]

        except Exception as e:
            logger.warning(f"Stopword removal failed: {e}")
            return tokens


class NLPProcessor:
    """
    Main NLP Processor supporting multiple languages and techniques.

    Uses configurable NLP models (spaCy, NLTK, underthesea) through the
    BaseLanguageProcessor interface to handle different languages uniformly.
    """

    def __init__(self, config: NLPConfig = None, model_config_path: Optional[Path] = None):
        self.config = config or NLPConfig()
        self.language_detector = LanguageDetector()
        self.text_preprocessor = TextPreprocessor()

        # Load model configuration
        self.model_config = NLPConfigLoader.load_config(model_config_path)

        # Initialize language processors with model config
        self.english_processor = EnglishProcessor(self.config, self.model_config)
        self.vietnamese_processor = VietnameseProcessor(self.config, self.model_config)

        # Cache for processed results
        self._cache: Dict[str, Any] = {}

        logger.info("NLP Processor initialized successfully")

    def _get_processor(self, language: Language) -> BaseLanguageProcessor:
        """Get the appropriate processor for the language."""
        if language == Language.ENGLISH:
            return self.english_processor
        elif language == Language.VIETNAMESE:
            return self.vietnamese_processor
        else:
            # Default to Vietnamese processor
            return self.vietnamese_processor

    def process_text(self, text: str, techniques: List[NLPTechnique] = None) -> NLPResult:
        """
        Process text using specified NLP techniques.

        Args:
            text: Input text to process
            techniques: List of NLP techniques to apply

        Returns:
            NLPResult with processed information
        """
        import time
        start_time = time.time()

        if not text:
            return NLPResult(text="", language=Language.UNKNOWN, processing_time=0.0)

        # Clean text
        cleaned_text = self.text_preprocessor.clean_text(text)

        # Detect language
        if self.config.auto_detect_language:
            language = self.language_detector.detect_language(cleaned_text)
        else:
            language = self.config.default_language

        # Limit text length
        if len(cleaned_text) > self.config.max_text_length:
            cleaned_text = cleaned_text[:self.config.max_text_length]
            logger.warning(f"Text truncated to {self.config.max_text_length} characters")

        # Get appropriate processor
        processor = self._get_processor(language)

        # Initialize result
        result = NLPResult(
            text=cleaned_text,
            language=language,
            processing_time=0.0
        )

        # Apply techniques
        if techniques is None:
            techniques = [NLPTechnique.TOKENIZATION, NLPTechnique.POS_TAGGING]

        for technique in techniques:
            try:
                if technique == NLPTechnique.TOKENIZATION:
                    result.tokens = processor.tokenize(cleaned_text)

                elif technique == NLPTechnique.POS_TAGGING:
                    result.pos_tags = processor.pos_tag(cleaned_text)

                elif technique == NLPTechnique.LEMMATIZATION:
                    result.lemmas = processor.lemmatize(cleaned_text)

                elif technique == NLPTechnique.NAMED_ENTITY_RECOGNITION:
                    result.entities = processor.extract_entities(cleaned_text)

                elif technique == NLPTechnique.SENTIMENT_ANALYSIS:
                    result.sentiment = processor.analyze_sentiment(cleaned_text)

                elif technique == NLPTechnique.DEPENDENCY_PARSING:
                    result.dependencies = processor.parse_dependencies(cleaned_text)

            except Exception as e:
                logger.warning(f"Failed to apply technique {technique.value}: {e}")

        result.processing_time = time.time() - start_time
        return result

    def analyze_text_comprehensive(self, text: str) -> TextAnalysis:
        """
        Perform comprehensive text analysis.

        Args:
            text: Input text to analyze

        Returns:
            TextAnalysis with comprehensive analysis results
        """
        import time
        start_time = time.time()

        if not text:
            return TextAnalysis(
                original_text="",
                cleaned_text="",
                language=Language.UNKNOWN,
                word_count=0,
                sentence_count=0,
                avg_word_length=0.0,
                sentiment_score=0.0,
                entities=[],
                keywords=[],
                topics=[],
                grammar_issues=[],
                processing_time=0.0
            )

        # Basic processing
        cleaned_text = self.text_preprocessor.clean_text(text)

        # Detect language
        if self.config.auto_detect_language:
            language = self.language_detector.detect_language(cleaned_text)
        else:
            language = self.config.default_language
        processor = self._get_processor(language)
        
        # Tokenization and basic metrics
        tokens = processor.tokenize(cleaned_text)
        lemmas = processor.lemmatize(cleaned_text)
        word_count = len(tokens)

        # Calculate metrics
        avg_word_length = sum(len(token) for token in tokens) / max(word_count, 1)

        # Sentiment analysis
        sentiment = processor.analyze_sentiment(cleaned_text)
        sentiment_score = sentiment.get('compound', 0.0)

        # Named entity recognition
        entities = processor.extract_entities(cleaned_text)

        # Keyword extraction
        keywords = self._extract_keywords(tokens, language)

        # Topics (basic implementation)
        topics = self._extract_topics(tokens, language)

        # Grammar issues (basic check)
        grammar_issues = self._check_grammar(cleaned_text, language)

        processing_time = time.time() - start_time

        return TextAnalysis(
            original_text=text,
            cleaned_text=cleaned_text,
            language=language,
            word_count=word_count,
            sentence_count=1,  # Simplified
            avg_word_length=avg_word_length,
            sentiment_score=sentiment_score,
            entities=entities,
            keywords=keywords,
            topics=topics,
            grammar_issues=grammar_issues,
            processing_time=processing_time
        )

    def _extract_keywords(self, tokens: List[str], language: Language) -> List[str]:
        """Extract keywords from tokens."""
        # Remove stopwords
        filtered_tokens = self.text_preprocessor.remove_stopwords(tokens, language)

        # Simple frequency-based keyword extraction
        if not filtered_tokens:
            return []

        # Count frequencies
        freq = {}
        for token in filtered_tokens:
            token = token.lower()
            if len(token) > 2:  # Skip very short words
                freq[token] = freq.get(token, 0) + 1

        # Return top keywords
        sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_keywords[:10]]

    def _extract_topics(self, tokens: List[str], language: Language) -> List[str]:
        """Extract basic topics from tokens."""
        # Simple topic extraction based on common patterns
        topics = []

        # Look for common topic indicators
        topic_keywords = {
            'technical': ['system', 'software', 'hardware', 'network', 'server', 'database', 'api'],
            'business': ['company', 'customer', 'service', 'product', 'market', 'sales', 'revenue'],
            'medical': ['patient', 'treatment', 'disease', 'health', 'medicine', 'doctor'],
            'legal': ['contract', 'agreement', 'law', 'regulation', 'policy', 'compliance'],
            'vietnamese_business': ['doanh nghiệp', 'khách hàng', 'dịch vụ', 'sản phẩm', 'thị trường']
        }

        text_lower = ' '.join(tokens).lower()

        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)

        return topics[:3]  # Return top 3 topics

    def _check_grammar(self, text: str, language: Language) -> List[str]:
        """Check for basic grammar issues."""
        issues = []

        # Basic checks
        if text.count('  ') > 0:
            issues.append("Multiple consecutive spaces found")

        if text.count('..') > 0:
            issues.append("Multiple consecutive periods found")

        # Language-specific checks
        if language == Language.ENGLISH:
            if re.search(r'\bi\s+[a-z]', text, re.IGNORECASE):
                issues.append("Incorrect capitalization after 'I'")

            if re.search(r'[.!?]\s*[a-z]', text):
                issues.append("Sentence doesn't start with capital letter")

        elif language == Language.VIETNAMESE:
            if re.search(r'[.!?]\s*[a-zàáãạảăắằẳẵặâấầẩẫậèéẹẻẽêếềểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹ]', text):
                issues.append("Câu không bắt đầu bằng chữ hoa")

        return issues

    def batch_process(self, texts: List[str], techniques: List[NLPTechnique] = None) -> List[NLPResult]:
        """
        Process multiple texts in batch.

        Args:
            texts: List of texts to process
            techniques: NLP techniques to apply

        Returns:
            List of NLPResult objects
        """
        results = []

        for text in texts:
            result = self.process_text(text, techniques)
            results.append(result)

        return results

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages with their configured models."""
        languages = []

        # Check English processor
        if self.english_processor.is_available() and self.english_processor.nlp_model:
            model_name = self.english_processor.nlp_model.get_name()
            languages.append(f"English ({model_name})")

        # Check Vietnamese processor
        if self.vietnamese_processor.is_available() and self.vietnamese_processor.nlp_model:
            model_name = self.vietnamese_processor.nlp_model.get_name()
            languages.append(f"Vietnamese ({model_name})")

        return languages

    def get_available_techniques(self) -> List[str]:
        """Get list of available NLP techniques."""
        return [technique.value for technique in NLPTechnique]