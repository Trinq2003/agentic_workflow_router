# Models package
from .nlp_processor import (
    NLPProcessor,
    TextPreprocessor,
)
from .config import NLPConfig, NLPConfigLoader, Language
from .base_classes import (
    BaseLanguageProcessor,
    BaseNLPModel,
    NLPResult,
    NLPTechnique,
    LanguageDetector,
    TextAnalysis
)

__all__ = [
    'NLPProcessor',
    'TextPreprocessor',
    'create_default_nlp_processor',
    'process_query_text',
    'analyze_query_comprehensive',
    'NLPConfig',
    'NLPConfigLoader',
    'Language',
    'BaseLanguageProcessor',
    'BaseNLPModel',
    'NLPResult',
    'NLPTechnique',
    'LanguageDetector',
    'TextAnalysis'
]

nlp_processor = NLPProcessor()