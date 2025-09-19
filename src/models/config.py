import logging
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    VIETNAMESE = "vi"
    UNKNOWN = "unknown"


@dataclass
class NLPConfig:
    """Configuration for NLP processing."""
    default_language: Language = Language.VIETNAMESE
    auto_detect_language: bool = True
    max_text_length: int = 10000
    enable_caching: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.max_text_length <= 0:
            raise ValueError("max_text_length must be > 0")


class NLPConfigLoader:
    """Load NLP configuration from YAML file."""

    @staticmethod
    def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load NLP configuration from config/nlp.yaml."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "nlp.yaml"

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {"models": {"english": "nltk", "vietnamese": "underthesea"}}
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return {"models": {"english": "nltk", "vietnamese": "underthesea"}}