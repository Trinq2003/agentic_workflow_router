#!/usr/bin/env python3
"""
Vietnamese NLP Processing Test Suite

This module provides comprehensive testing for Vietnamese language processing
capabilities of the NetMind NLP system. It focuses on:

- Language detection accuracy
- Text preprocessing for Vietnamese
- Tokenization, POS tagging, and lemmatization
- Named entity recognition
- Sentiment analysis
- Comprehensive analysis features

The test suite uses real data from the dataset.xlsx file to ensure
realistic testing scenarios.
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader, DataLoaderConfig, QueryData
from models import (
    NLPProcessor,
    NLPConfig,
    NLPResult,
    TextAnalysis,
    NLPTechnique,
    Language
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VietnameseNLPTestResults:
    """Results container for Vietnamese NLP tests."""

    total_queries_tested: int = 0
    vietnamese_queries_detected: int = 0
    language_detection_accuracy: float = 0.0

    tokenization_tests: int = 0
    tokenization_success: int = 0
    tokenization_avg_tokens: float = 0.0

    pos_tagging_tests: int = 0
    pos_tagging_success: int = 0
    pos_tagging_avg_tags: float = 0.0

    ner_tests: int = 0
    ner_success: int = 0
    total_entities_found: int = 0
    entity_types: Dict[str, int] = field(default_factory=dict)

    sentiment_tests: int = 0
    sentiment_success: int = 0
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)

    processing_times: List[float] = field(default_factory=list)
    avg_processing_time: float = 0.0

    sample_results: List[Dict[str, Any]] = field(default_factory=list)

    def calculate_metrics(self):
        """Calculate derived metrics from test results."""
        if self.total_queries_tested > 0:
            self.language_detection_accuracy = (
                self.vietnamese_queries_detected / self.total_queries_tested
            ) * 100

        if self.processing_times:
            self.avg_processing_time = sum(self.processing_times) / len(self.processing_times)

        if self.tokenization_tests > 0:
            self.tokenization_avg_tokens = (
                sum(r.get('token_count', 0) for r in self.sample_results) /
                len(self.sample_results)
            )

        if self.pos_tagging_tests > 0:
            self.pos_tagging_avg_tags = (
                sum(r.get('pos_tag_count', 0) for r in self.sample_results) /
                len(self.sample_results)
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for reporting."""
        return {
            'total_queries_tested': self.total_queries_tested,
            'vietnamese_queries_detected': self.vietnamese_queries_detected,
            'language_detection_accuracy': f"{self.language_detection_accuracy:.2f}%",
            'tokenization': {
                'tests': self.tokenization_tests,
                'success_rate': f"{(self.tokenization_success/self.tokenization_tests*100):.2f}%" if self.tokenization_tests > 0 else "N/A",
                'avg_tokens': f"{self.tokenization_avg_tokens:.1f}"
            },
            'pos_tagging': {
                'tests': self.pos_tagging_tests,
                'success_rate': f"{(self.pos_tagging_success/self.pos_tagging_tests*100):.2f}%" if self.pos_tagging_tests > 0 else "N/A",
                'avg_tags': f"{self.pos_tagging_avg_tags:.1f}"
            },
            'named_entity_recognition': {
                'tests': self.ner_tests,
                'success_rate': f"{(self.ner_success/self.ner_tests*100):.2f}%" if self.ner_tests > 0 else "N/A",
                'total_entities': self.total_entities_found,
                'entity_types': self.entity_types
            },
            'sentiment_analysis': {
                'tests': self.sentiment_tests,
                'success_rate': f"{(self.sentiment_success/self.sentiment_tests*100):.2f}%" if self.sentiment_tests > 0 else "N/A",
                'distribution': self.sentiment_distribution
            },
            'performance': {
                'avg_processing_time': f"{self.avg_processing_time:.4f}s",
                'total_processing_time': f"{sum(self.processing_times):.4f}s"
            }
        }


class VietnameseNLPTestSuite:
    """
    Comprehensive test suite for Vietnamese NLP processing.

    Tests various aspects of Vietnamese language processing including:
    - Language detection
    - Tokenization
    - POS tagging
    - Named entity recognition
    - Sentiment analysis
    - Performance metrics
    """

    def __init__(self, data_path: str = r"C:\Users\ADMIN\Code\VTNET\netmind_workflow\data\dataset.xlsx", max_samples: int = 100):
        """
        Initialize the test suite.

        Args:
            data_path: Path to the dataset Excel file
            max_samples: Maximum number of queries to test
        """
        self.data_path = Path(__file__).parent.parent.parent / data_path
        self.max_samples = max_samples

        # Initialize components
        self.data_loader = None
        self.nlp_processor = None
        self.test_results = VietnameseNLPTestResults()

        logger.info(f"Initialized VietnameseNLPTestSuite with data path: {self.data_path}")

    def setup(self):
        """Set up the test environment."""
        logger.info("Setting up test environment...")

        # Initialize data loader
        config = DataLoaderConfig(
            file_path=self.data_path,
            max_workers=4,
            chunk_size=500,
            cache_enabled=True
        )
        self.data_loader = DataLoader(config)

        # Initialize NLP processor
        nlp_config = NLPConfig(
            auto_detect_language=True,
            max_text_length=1000
        )
        self.nlp_processor = NLPProcessor(nlp_config)

        logger.info("Test environment setup complete")

    def load_test_data(self) -> List[QueryData]:
        """Load and filter test data for Vietnamese queries."""
        logger.info("Loading test data...")

        # Load all data
        all_data = self.data_loader.get_processed_data()

        # Filter for queries that might be Vietnamese or test language detection
        vietnamese_queries = []
        sample_queries = []

        # First, try to detect Vietnamese queries
        for query in all_data[:self.max_samples]:
            # Test language detection
            detected_lang = self.nlp_processor.language_detector.detect_language(query.query)

            if detected_lang == Language.VIETNAMESE:
                vietnamese_queries.append(query)
            else:
                # Include some non-Vietnamese queries for language detection testing
                sample_queries.append(query)

            if len(vietnamese_queries) + len(sample_queries) >= self.max_samples:
                break

        # If no Vietnamese detected, use a mix of queries
        if not vietnamese_queries:
            logger.warning("No Vietnamese queries detected by language detector, using sample queries")
            vietnamese_queries = all_data[:min(self.max_samples, len(all_data))]

        test_queries = vietnamese_queries[:50] + sample_queries[:50]  # Mix of queries
        logger.info(f"Loaded {len(test_queries)} test queries ({len(vietnamese_queries)} detected as Vietnamese)")

        return test_queries

    def test_language_detection(self, queries: List[QueryData]) -> Dict[str, Any]:
        """Test language detection accuracy."""
        logger.info("Testing language detection...")

        results = {
            'total': len(queries),
            'vietnamese_detected': 0,
            'english_detected': 0,
            'unknown_detected': 0,
            'detection_times': []
        }

        vietnamese_indicators = [
            'và', 'của', 'là', 'được', 'cho', 'trong', 'với', 'như',
            'từ', 'đến', 'qua', 'theo', 'nếu', 'khi', 'thì', 'mà',
            'để', 'ở', 'vì', 'nên', 'các', 'những', 'đang', 'đã',
            'sẽ', 'có', 'không', 'đây', 'đó', 'này'
        ]

        for query in queries:
            start_time = time.time()
            detected_lang = self.nlp_processor.language_detector.detect_language(query.query)
            detection_time = time.time() - start_time

            results['detection_times'].append(detection_time)

            # Heuristic check for Vietnamese (presence of Vietnamese words)
            has_vietnamese_words = any(word in query.query.lower() for word in vietnamese_indicators)

            if detected_lang == Language.VIETNAMESE:
                results['vietnamese_detected'] += 1
                if has_vietnamese_words:
                    self.test_results.vietnamese_queries_detected += 1
            elif detected_lang == Language.ENGLISH:
                results['english_detected'] += 1
            else:
                results['unknown_detected'] += 1

        results['avg_detection_time'] = sum(results['detection_times']) / len(results['detection_times'])
        logger.info(f"Language detection: {results['vietnamese_detected']}/{results['total']} detected as Vietnamese")

        return results

    def test_tokenization(self, queries: List[QueryData]) -> Dict[str, Any]:
        """Test Vietnamese tokenization."""
        logger.info("Testing Vietnamese tokenization...")

        results = {
            'total': len(queries),
            'successful': 0,
            'failed': 0,
            'token_counts': [],
            'processing_times': [],
            'samples': []
        }

        for query in queries:
            try:
                start_time = time.time()

                # Process with tokenization
                result = self.nlp_processor.process_text(
                    query.query,
                    techniques=[NLPTechnique.TOKENIZATION]
                )

                processing_time = time.time() - start_time

                if result.tokens and len(result.tokens) > 0:
                    results['successful'] += 1
                    results['token_counts'].append(len(result.tokens))
                    results['processing_times'].append(processing_time)

                    # Store sample result
                    if len(results['samples']) < 5:  # Keep 5 samples
                        results['samples'].append({
                            'query': query.query[:50] + "..." if len(query.query) > 50 else query.query,
                            'tokens': result.tokens[:10],  # First 10 tokens
                            'token_count': len(result.tokens),
                            'language': result.language.value
                        })

                    self.test_results.processing_times.append(processing_time)
                else:
                    results['failed'] += 1

                self.test_results.tokenization_tests += 1
                if result.tokens:
                    self.test_results.tokenization_success += 1

            except Exception as e:
                logger.warning(f"Tokenization failed for query: {e}")
                results['failed'] += 1
                self.test_results.tokenization_tests += 1

        if results['token_counts']:
            results['avg_tokens'] = sum(results['token_counts']) / len(results['token_counts'])
        if results['processing_times']:
            results['avg_time'] = sum(results['processing_times']) / len(results['processing_times'])

        logger.info(f"Tokenization: {results['successful']}/{results['total']} successful")
        return results

    def test_pos_tagging(self, queries: List[QueryData]) -> Dict[str, Any]:
        """Test Vietnamese POS tagging."""
        logger.info("Testing Vietnamese POS tagging...")

        results = {
            'total': len(queries),
            'successful': 0,
            'failed': 0,
            'tag_counts': [],
            'processing_times': [],
            'tag_distribution': defaultdict(int),
            'samples': []
        }

        for query in queries:
            try:
                start_time = time.time()

                # Process with POS tagging
                result = self.nlp_processor.process_text(
                    query.query,
                    techniques=[NLPTechnique.POS_TAGGING]
                )

                processing_time = time.time() - start_time

                if result.pos_tags and len(result.pos_tags) > 0:
                    results['successful'] += 1
                    results['tag_counts'].append(len(result.pos_tags))
                    results['processing_times'].append(processing_time)

                    # Count tag types
                    for _, tag in result.pos_tags:
                        results['tag_distribution'][tag] += 1

                    # Store sample result
                    if len(results['samples']) < 3:  # Keep 3 samples
                        results['samples'].append({
                            'query': query.query[:50] + "..." if len(query.query) > 50 else query.query,
                            'pos_tags': result.pos_tags[:5],  # First 5 tags
                            'tag_count': len(result.pos_tags),
                            'language': result.language.value
                        })

                    self.test_results.processing_times.append(processing_time)
                else:
                    results['failed'] += 1

                self.test_results.pos_tagging_tests += 1
                if result.pos_tags:
                    self.test_results.pos_tagging_success += 1

            except Exception as e:
                logger.warning(f"POS tagging failed for query: {e}")
                results['failed'] += 1
                self.test_results.pos_tagging_tests += 1

        if results['tag_counts']:
            results['avg_tags'] = sum(results['tag_counts']) / len(results['tag_counts'])
        if results['processing_times']:
            results['avg_time'] = sum(results['processing_times']) / len(results['processing_times'])

        logger.info(f"POS tagging: {results['successful']}/{results['total']} successful")
        return results

    def test_named_entity_recognition(self, queries: List[QueryData]) -> Dict[str, Any]:
        """Test Vietnamese named entity recognition."""
        logger.info("Testing Vietnamese NER...")

        results = {
            'total': len(queries),
            'successful': 0,
            'failed': 0,
            'total_entities': 0,
            'entity_types': defaultdict(int),
            'processing_times': [],
            'samples': []
        }

        for query in queries:
            try:
                start_time = time.time()

                # Process with NER
                result = self.nlp_processor.process_text(
                    query.query,
                    techniques=[NLPTechnique.NAMED_ENTITY_RECOGNITION]
                )

                processing_time = time.time() - start_time

                results['processing_times'].append(processing_time)

                if result.entities is not None:  # Can be empty list, which is valid
                    results['successful'] += 1
                    results['total_entities'] += len(result.entities)

                    # Count entity types
                    for entity in result.entities:
                        entity_type = entity.get('label', 'UNKNOWN')
                        results['entity_types'][entity_type] += 1

                    # Store sample result
                    if len(results['samples']) < 3 and result.entities:  # Keep 3 samples with entities
                        results['samples'].append({
                            'query': query.query[:50] + "..." if len(query.query) > 50 else query.query,
                            'entities': result.entities[:3],  # First 3 entities
                            'entity_count': len(result.entities),
                            'language': result.language.value
                        })

                    self.test_results.total_entities_found += len(result.entities)
                else:
                    results['failed'] += 1

                self.test_results.ner_tests += 1
                if result.entities is not None:
                    self.test_results.ner_success += 1

                # Update entity types in test results
                for entity_type, count in results['entity_types'].items():
                    self.test_results.entity_types[entity_type] = count

            except Exception as e:
                logger.warning(f"NER failed for query: {e}")
                results['failed'] += 1
                self.test_results.ner_tests += 1

        if results['processing_times']:
            results['avg_time'] = sum(results['processing_times']) / len(results['processing_times'])

        logger.info(f"NER: {results['successful']}/{results['total']} successful, {results['total_entities']} entities found")
        return results

    def test_sentiment_analysis(self, queries: List[QueryData]) -> Dict[str, Any]:
        """Test Vietnamese sentiment analysis."""
        logger.info("Testing Vietnamese sentiment analysis...")

        results = {
            'total': len(queries),
            'successful': 0,
            'failed': 0,
            'sentiment_distribution': defaultdict(int),
            'processing_times': [],
            'samples': []
        }

        for query in queries:
            try:
                start_time = time.time()

                # Process with sentiment analysis
                result = self.nlp_processor.process_text(
                    query.query,
                    techniques=[NLPTechnique.SENTIMENT_ANALYSIS]
                )

                processing_time = time.time() - start_time

                if result.sentiment and isinstance(result.sentiment, dict):
                    results['successful'] += 1
                    results['processing_times'].append(processing_time)

                    # Categorize sentiment
                    compound = result.sentiment.get('compound', 0)
                    if compound > 0.1:
                        sentiment_category = 'positive'
                    elif compound < -0.1:
                        sentiment_category = 'negative'
                    else:
                        sentiment_category = 'neutral'

                    results['sentiment_distribution'][sentiment_category] += 1
                    self.test_results.sentiment_distribution[sentiment_category] += 1

                    # Store sample result
                    if len(results['samples']) < 3:  # Keep 3 samples
                        results['samples'].append({
                            'query': query.query[:50] + "..." if len(query.query) > 50 else query.query,
                            'sentiment': result.sentiment,
                            'category': sentiment_category,
                            'language': result.language.value
                        })

                    self.test_results.processing_times.append(processing_time)
                else:
                    results['failed'] += 1

                self.test_results.sentiment_tests += 1
                if result.sentiment:
                    self.test_results.sentiment_success += 1

            except Exception as e:
                logger.warning(f"Sentiment analysis failed for query: {e}")
                results['failed'] += 1
                self.test_results.sentiment_tests += 1

        if results['processing_times']:
            results['avg_time'] = sum(results['processing_times']) / len(results['processing_times'])

        logger.info(f"Sentiment analysis: {results['successful']}/{results['total']} successful")
        return results

    def test_comprehensive_analysis(self, queries: List[QueryData]) -> Dict[str, Any]:
        """Test comprehensive Vietnamese text analysis."""
        logger.info("Testing comprehensive Vietnamese analysis...")

        results = {
            'total': len(queries),
            'successful': 0,
            'failed': 0,
            'processing_times': [],
            'avg_word_counts': [],
            'avg_sentiment_scores': [],
            'keyword_counts': [],
            'topic_counts': [],
            'samples': []
        }

        # Test only first 20 queries for comprehensive analysis (it's more expensive)
        test_queries = queries[:20]

        for query in test_queries:
            try:
                start_time = time.time()

                # Comprehensive analysis
                analysis = self.nlp_processor.analyze_text_comprehensive(query.query)

                processing_time = time.time() - start_time

                if analysis:
                    results['successful'] += 1
                    results['processing_times'].append(processing_time)
                    results['avg_word_counts'].append(analysis.word_count)
                    results['avg_sentiment_scores'].append(analysis.sentiment_score)
                    results['keyword_counts'].append(len(analysis.keywords))
                    results['topic_counts'].append(len(analysis.topics))

                    # Store sample result
                    if len(results['samples']) < 2:  # Keep 2 samples
                        results['samples'].append({
                            'query': query.query[:50] + "..." if len(query.query) > 50 else query.query,
                            'word_count': analysis.word_count,
                            'sentiment_score': analysis.sentiment_score,
                            'keywords': analysis.keywords[:5],
                            'topics': analysis.topics,
                            'entities': [e.get('text', '') for e in analysis.entities[:3]],
                            'language': analysis.language.value
                        })

                    self.test_results.processing_times.append(processing_time)
                else:
                    results['failed'] += 1

            except Exception as e:
                logger.warning(f"Comprehensive analysis failed for query: {e}")
                results['failed'] += 1

        # Calculate averages
        if results['processing_times']:
            results['avg_time'] = sum(results['processing_times']) / len(results['processing_times'])
        if results['avg_word_counts']:
            results['avg_words'] = sum(results['avg_word_counts']) / len(results['avg_word_counts'])
        if results['avg_sentiment_scores']:
            results['avg_sentiment'] = sum(results['avg_sentiment_scores']) / len(results['avg_sentiment_scores'])
        if results['keyword_counts']:
            results['avg_keywords'] = sum(results['keyword_counts']) / len(results['keyword_counts'])
        if results['topic_counts']:
            results['avg_topics'] = sum(results['topic_counts']) / len(results['topic_counts'])

        logger.info(f"Comprehensive analysis: {results['successful']}/{results['total']} successful")
        return results

    def run_all_tests(self) -> VietnameseNLPTestResults:
        """Run all Vietnamese NLP tests."""
        logger.info("Starting Vietnamese NLP test suite...")

        try:
            # Setup
            self.setup()

            # Load test data
            test_queries = self.load_test_data()
            self.test_results.total_queries_tested = len(test_queries)

            if not test_queries:
                logger.error("No test data available!")
                return self.test_results

            # Run individual tests
            logger.info("Running language detection tests...")
            lang_results = self.test_language_detection(test_queries)

            logger.info("Running tokenization tests...")
            token_results = self.test_tokenization(test_queries)

            logger.info("Running POS tagging tests...")
            pos_results = self.test_pos_tagging(test_queries)

            logger.info("Running NER tests...")
            ner_results = self.test_named_entity_recognition(test_queries)

            logger.info("Running sentiment analysis tests...")
            sentiment_results = self.test_sentiment_analysis(test_queries)

            logger.info("Running comprehensive analysis tests...")
            comprehensive_results = self.test_comprehensive_analysis(test_queries)

            # Store sample results for reporting
            self.test_results.sample_results = (
                token_results.get('samples', []) +
                pos_results.get('samples', []) +
                ner_results.get('samples', []) +
                sentiment_results.get('samples', [])
            )

            # Calculate final metrics
            self.test_results.calculate_metrics()

            # Print summary
            self.print_test_summary()

            logger.info("Vietnamese NLP test suite completed successfully!")
            return self.test_results

        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            raise

    def print_test_summary(self):
        """Print a comprehensive test summary."""
        print("\n" + "="*80)
        print("VIETNAMESE NLP PROCESSING TEST RESULTS")
        print("="*80)

        results_dict = self.test_results.to_dict()

        print(f"\n📊 OVERVIEW:")
        print(f"   Total queries tested: {results_dict['total_queries_tested']}")
        print(f"   Vietnamese queries detected: {results_dict['vietnamese_queries_detected']}")
        print(f"   Language detection accuracy: {results_dict['language_detection_accuracy']}")

        print(f"\n🔤 TOKENIZATION:")
        print(f"   Tests: {results_dict['tokenization']['tests']}")
        print(f"   Success rate: {results_dict['tokenization']['success_rate']}")
        print(f"   Average tokens per query: {results_dict['tokenization']['avg_tokens']}")

        print(f"\n🏷️  POS TAGGING:")
        print(f"   Tests: {results_dict['pos_tagging']['tests']}")
        print(f"   Success rate: {results_dict['pos_tagging']['success_rate']}")
        print(f"   Average tags per query: {results_dict['pos_tagging']['avg_tags']}")

        print(f"\n🎯 NAMED ENTITY RECOGNITION:")
        print(f"   Tests: {results_dict['named_entity_recognition']['tests']}")
        print(f"   Success rate: {results_dict['named_entity_recognition']['success_rate']}")
        print(f"   Total entities found: {results_dict['named_entity_recognition']['total_entities']}")
        if results_dict['named_entity_recognition']['entity_types']:
            print("   Entity types:")
            for entity_type, count in results_dict['named_entity_recognition']['entity_types'].items():
                print(f"     {entity_type}: {count}")

        print(f"\n😊 SENTIMENT ANALYSIS:")
        print(f"   Tests: {results_dict['sentiment_analysis']['tests']}")
        print(f"   Success rate: {results_dict['sentiment_analysis']['success_rate']}")
        if results_dict['sentiment_analysis']['distribution']:
            print("   Sentiment distribution:")
            for sentiment, count in results_dict['sentiment_analysis']['distribution'].items():
                print(f"     {sentiment}: {count}")

        print(f"\n⚡ PERFORMANCE:")
        print(f"   Average processing time: {results_dict['performance']['avg_processing_time']}")
        print(f"   Total processing time: {results_dict['performance']['total_processing_time']}")

        print(f"\n📝 SAMPLE RESULTS:")
        for i, sample in enumerate(self.test_results.sample_results[:3], 1):
            print(f"\n   Sample {i}:")
            print(f"   Query: {sample.get('query', 'N/A')}")
            if 'tokens' in sample:
                print(f"   Tokens: {sample['tokens']}")
            if 'pos_tags' in sample:
                print(f"   POS Tags: {sample['pos_tags']}")
            if 'entities' in sample:
                print(f"   Entities: {sample['entities']}")
            if 'sentiment' in sample:
                print(f"   Sentiment: {sample['sentiment']}")
            print(f"   Language: {sample.get('language', 'N/A')}")

        print("\n" + "="*80)


def main():
    """Main function to run the Vietnamese NLP test suite."""
    print("🚀 Starting Vietnamese NLP Processing Test Suite...")

    # Initialize test suite
    test_suite = VietnameseNLPTestSuite(
        data_path=r"C:\Users\ADMIN\Code\VTNET\netmind_workflow\data\dataset.xlsx",  # Absolute path to data file
        max_samples=50  # Test with 50 queries for reasonable runtime
    )

    try:
        # Run all tests
        results = test_suite.run_all_tests()

        # Exit with success/failure code
        success_rate = (
            results.tokenization_success + results.pos_tagging_success +
            results.ner_success + results.sentiment_success
        ) / (
            results.tokenization_tests + results.pos_tagging_tests +
            results.ner_tests + results.sentiment_tests
        ) * 100 if (
            results.tokenization_tests + results.pos_tagging_tests +
            results.ner_tests + results.sentiment_tests
        ) > 0 else 0

        print(f"Overall success rate: {success_rate:.2f}%")
        if success_rate >= 80:
            print("✅ Test suite PASSED!")
            return 0
        else:
            print("❌ Test suite FAILED!")
            return 1

    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        print(f"💥 Test suite CRASHED: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
