"""
Sentence Structure Analysis for NetMind Workflow

This module performs comprehensive sentence structure analysis on query data,
including lexical analysis, syntactic parsing, grammar checking, and semantic analysis.

Features:
- Parallel processing for performance
- Support for Vietnamese and English queries
- Comprehensive NLP analysis using multiple techniques
- Batch processing with progress tracking
- Results saved to reports folder
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
import pandas as pd

# Import from the project
from src.data.data_loader import DataLoader, DataLoaderConfig, QueryData
from src.models.nlp_processor import NLPProcessor
from src.models.base_classes import NLPTechnique, TextAnalysis
from src.models.config import Language

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SentenceAnalysisResult:
    """Result of sentence structure analysis for a single query."""
    query_id: int
    original_query: str
    knowledge_domain: str
    worker: str
    language: str
    word_count: int
    sentence_count: int
    avg_word_length: float
    tokens: List[str]
    pos_tags: List[Tuple[str, str]]
    lemmas: List[str]
    entities: List[Dict[str, Any]]
    sentiment: Dict[str, float]
    dependencies: List[Dict[str, Any]]
    keywords: List[str]
    topics: List[str]
    grammar_issues: List[str]
    processing_time: float
    analysis_timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class BatchAnalysisResult:
    """Results from processing a batch of queries."""
    batch_id: int
    worker_type: str
    total_queries: int
    processed_queries: int
    successful_analyses: int
    failed_analyses: int
    average_processing_time: float
    results: List[SentenceAnalysisResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    processing_timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['results'] = [result.to_dict() for result in self.results]
        return data


class SentenceStructureAnalyzer:
    """
    Main analyzer for sentence structure analysis.

    Performs comprehensive NLP analysis on queries including:
    - Lexical analysis (tokenization, POS tagging, lemmatization)
    - Syntactic analysis (dependency parsing)
    - Semantic analysis (sentiment, entities, keywords)
    - Grammar checking
    """

    def __init__(self, data_path: str = None, max_workers: int = 4):
        """
        Initialize the analyzer.

        Args:
            data_path: Path to the dataset Excel file
            max_workers: Maximum number of parallel workers for processing
        """
        # Set default data path if not provided
        if data_path is None:
            # Try to find data/dataset.xlsx from the current working directory
            current_dir = Path.cwd()
            if (current_dir / "data" / "dataset.xlsx").exists():
                data_path = str(current_dir / "data" / "dataset.xlsx")
            elif (current_dir.parent / "data" / "dataset.xlsx").exists():
                data_path = str(current_dir.parent / "data" / "dataset.xlsx")
            else:
                raise FileNotFoundError("Could not find dataset.xlsx in expected locations")

        self.data_path = Path(data_path)
        self.max_workers = max_workers
        self.nlp_processor = NLPProcessor()
        self.data_loader = None

        # Create reports directory if it doesn't exist
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)

        logger.info("Sentence Structure Analyzer initialized")

    def load_data(self) -> DataLoader:
        """Load and prepare the dataset."""
        logger.info(f"Loading data from {self.data_path}")

        config = DataLoaderConfig(
            file_path=self.data_path,
            query_column="query",
            knowledge_domain_column="knowledge_domain",
            worker_column="worker",
            max_workers=self.max_workers
        )

        self.data_loader = DataLoader(config)
        return self.data_loader

    def get_worker_samples(self, worker_type: str, sample_size: int = 300,
                          random_state: int = 42) -> List[QueryData]:
        """
        Get random sample of queries for a specific worker type.

        Args:
            worker_type: Type of worker ('FUNCTION_CALLING' or 'DOC_SEARCH')
            sample_size: Number of samples to get
            random_state: Random seed for reproducibility

        Returns:
            List of QueryData objects for the worker type
        """
        if self.data_loader is None:
            self.load_data()

        logger.info(f"Getting {sample_size} random samples for worker: {worker_type}")

        # Get all data as DataFrame
        df = self.data_loader.to_dataframe()

        # Filter by worker type
        worker_df = df[df['workers'] == worker_type].copy()

        if len(worker_df) == 0:
            logger.warning(f"No queries found for worker type: {worker_type}")
            return []

        # Sample the requested number of queries
        sample_df = worker_df.sample(
            n=min(sample_size, len(worker_df)),
            random_state=random_state
        )

        # Convert back to QueryData objects
        samples = []
        for idx, row in sample_df.iterrows():
            query_data = QueryData(
                query=row['query'],
                knowledge_domain=row['knowledge_domain'],
                workers=[row['workers']],  # Single worker in this case
                index=idx
            )
            samples.append(query_data)

        logger.info(f"Retrieved {len(samples)} samples for {worker_type}")
        return samples

    def analyze_single_query(self, query_data: QueryData) -> SentenceAnalysisResult:
        """
        Perform sentence structure analysis on a single query.

        Args:
            query_data: QueryData object to analyze

        Returns:
            SentenceAnalysisResult with comprehensive analysis
        """
        start_time = time.time()

        try:
            # Perform comprehensive NLP analysis
            analysis = self.nlp_processor.analyze_text_comprehensive(query_data.query)

            # Perform additional NLP processing for detailed results
            nlp_result = self.nlp_processor.process_text(
                query_data.query,
                techniques=[
                    NLPTechnique.TOKENIZATION,
                    NLPTechnique.POS_TAGGING,
                    NLPTechnique.LEMMATIZATION,
                    NLPTechnique.NAMED_ENTITY_RECOGNITION,
                    NLPTechnique.SENTIMENT_ANALYSIS,
                    NLPTechnique.DEPENDENCY_PARSING
                ]
            )

            # Create result object
            result = SentenceAnalysisResult(
                query_id=query_data.index,
                original_query=query_data.query,
                knowledge_domain=query_data.knowledge_domain,
                worker=query_data.workers[0] if query_data.workers else "UNKNOWN",
                language=analysis.language.value,
                word_count=analysis.word_count,
                sentence_count=analysis.sentence_count,
                avg_word_length=analysis.avg_word_length,
                tokens=nlp_result.tokens,
                pos_tags=nlp_result.pos_tags,
                lemmas=nlp_result.lemmas,
                entities=nlp_result.entities,
                sentiment=nlp_result.sentiment,
                dependencies=nlp_result.dependencies,
                keywords=analysis.keywords,
                topics=analysis.topics,
                grammar_issues=analysis.grammar_issues,
                processing_time=time.time() - start_time
            )

            logger.debug(f"Successfully analyzed query {query_data.index}")
            return result

        except Exception as e:
            logger.error(f"Failed to analyze query {query_data.index}: {e}")

            # Return minimal result for failed analysis
            return SentenceAnalysisResult(
                query_id=query_data.index,
                original_query=query_data.query,
                knowledge_domain=query_data.knowledge_domain,
                worker=query_data.workers[0] if query_data.workers else "UNKNOWN",
                language="UNKNOWN",
                word_count=0,
                sentence_count=0,
                avg_word_length=0.0,
                tokens=[],
                pos_tags=[],
                lemmas=[],
                entities=[],
                sentiment={},
                dependencies=[],
                keywords=[],
                topics=[],
                grammar_issues=[f"Analysis failed: {str(e)}"],
                processing_time=time.time() - start_time
            )

    def analyze_queries_parallel(self, queries: List[QueryData],
                               batch_size: int = 50) -> BatchAnalysisResult:
        """
        Analyze multiple queries in parallel with batch processing.

        Args:
            queries: List of QueryData objects to analyze
            batch_size: Size of each processing batch

        Returns:
            BatchAnalysisResult with all analysis results
        """
        if not queries:
            return BatchAnalysisResult(
                batch_id=0,
                worker_type="EMPTY",
                total_queries=0,
                processed_queries=0,
                successful_analyses=0,
                failed_analyses=0,
                average_processing_time=0.0
            )

        worker_type = queries[0].workers[0] if queries[0].workers else "UNKNOWN"
        logger.info(f"Starting parallel analysis of {len(queries)} queries for {worker_type}")

        batch_results = []
        total_processed = 0
        total_successful = 0
        total_failed = 0
        all_processing_times = []

        # Process in batches
        for batch_start in range(0, len(queries), batch_size):
            batch_end = min(batch_start + batch_size, len(queries))
            batch_queries = queries[batch_start:batch_end]
            batch_id = batch_start // batch_size + 1

            logger.info(f"Processing batch {batch_id}: queries {batch_start}-{batch_end}")

            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_query = {
                    executor.submit(self.analyze_single_query, query): query
                    for query in batch_queries
                }

                batch_start_time = time.time()
                batch_results_temp = []

                for future in as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        result = future.result()
                        batch_results_temp.append(result)
                        total_processed += 1

                        if result.grammar_issues and any("failed" in issue.lower() for issue in result.grammar_issues):
                            total_failed += 1
                        else:
                            total_successful += 1

                        all_processing_times.append(result.processing_time)

                    except Exception as e:
                        logger.error(f"Batch processing error for query {query.index}: {e}")
                        total_failed += 1

                batch_processing_time = time.time() - batch_start_time
                logger.info(f"Batch {batch_id} completed in {batch_processing_time:.2f}s")

                batch_results.extend(batch_results_temp)

                # Save intermediate results
                self.save_batch_results(batch_results_temp, batch_id, worker_type)

        # Calculate average processing time
        avg_processing_time = sum(all_processing_times) / len(all_processing_times) if all_processing_times else 0.0

        result = BatchAnalysisResult(
            batch_id=0,  # Overall result
            worker_type=worker_type,
            total_queries=len(queries),
            processed_queries=total_processed,
            successful_analyses=total_successful,
            failed_analyses=total_failed,
            average_processing_time=avg_processing_time,
            results=batch_results
        )

        logger.info(f"Completed analysis: {total_successful}/{total_processed} successful")
        return result

    def save_batch_results(self, results: List[SentenceAnalysisResult],
                          batch_id: int, worker_type: str) -> None:
        """
        Save batch results to JSON file.

        Args:
            results: List of analysis results
            batch_id: Batch identifier
            worker_type: Type of worker analyzed
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"sentence_analysis_{worker_type}_batch_{batch_id}_{timestamp}.json"
        filepath = self.reports_dir / filename

        # Convert results to dictionaries
        data = {
            "batch_id": batch_id,
            "worker_type": worker_type,
            "timestamp": timestamp,
            "results_count": len(results),
            "results": [result.to_dict() for result in results]
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved batch results to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save batch results: {e}")

    def save_final_results(self, function_calling_results: BatchAnalysisResult,
                          doc_search_results: BatchAnalysisResult) -> None:
        """
        Save final comprehensive results to reports folder.

        Args:
            function_calling_results: Results for FUNCTION_CALLING queries
            doc_search_results: Results for DOC_SEARCH queries
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save summary statistics
        summary_data = {
            "analysis_timestamp": timestamp,
            "function_calling": {
                "total_queries": function_calling_results.total_queries,
                "processed_queries": function_calling_results.processed_queries,
                "successful_analyses": function_calling_results.successful_analyses,
                "failed_analyses": function_calling_results.failed_analyses,
                "average_processing_time": function_calling_results.average_processing_time
            },
            "doc_search": {
                "total_queries": doc_search_results.total_queries,
                "processed_queries": doc_search_results.processed_queries,
                "successful_analyses": doc_search_results.successful_analyses,
                "failed_analyses": doc_search_results.failed_analyses,
                "average_processing_time": doc_search_results.average_processing_time
            },
            "overall": {
                "total_processed": (function_calling_results.processed_queries +
                                  doc_search_results.processed_queries),
                "total_successful": (function_calling_results.successful_analyses +
                                   doc_search_results.successful_analyses),
                "average_processing_time": (
                    (function_calling_results.average_processing_time +
                     doc_search_results.average_processing_time) / 2
                )
            }
        }

        summary_file = self.reports_dir / f"sentence_analysis_summary_{timestamp}.json"
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved analysis summary to {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")

        # Save detailed results as CSV for easier analysis
        self.save_results_as_csv(function_calling_results, f"function_calling_analysis_{timestamp}.csv")
        self.save_results_as_csv(doc_search_results, f"doc_search_analysis_{timestamp}.csv")

    def save_results_as_csv(self, batch_result: BatchAnalysisResult, filename: str) -> None:
        """
        Save analysis results as CSV file.

        Args:
            batch_result: Batch analysis results
            filename: Output filename
        """
        filepath = self.reports_dir / filename

        try:
            # Flatten results for CSV format
            rows = []
            for result in batch_result.results:
                row = {
                    'query_id': result.query_id,
                    'original_query': result.original_query,
                    'knowledge_domain': result.knowledge_domain,
                    'worker': result.worker,
                    'language': result.language,
                    'word_count': result.word_count,
                    'sentence_count': result.sentence_count,
                    'avg_word_length': result.avg_word_length,
                    'tokens_count': len(result.tokens),
                    'pos_tags_count': len(result.pos_tags),
                    'entities_count': len(result.entities),
                    'keywords_count': len(result.keywords),
                    'topics_count': len(result.topics),
                    'grammar_issues_count': len(result.grammar_issues),
                    'sentiment_score': result.sentiment.get('compound', 0.0),
                    'processing_time': result.processing_time,
                    'analysis_timestamp': result.analysis_timestamp,
                    'tokens': '|'.join(result.tokens[:10]),  # First 10 tokens
                    'keywords': '|'.join(result.keywords[:5]),  # Top 5 keywords
                    'topics': '|'.join(result.topics),
                    'grammar_issues': '|'.join(result.grammar_issues[:3])  # First 3 issues
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"Saved CSV results to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save CSV results: {e}")

    def run_analysis(self, sample_size: int = 300) -> None:
        """
        Run the complete sentence structure analysis.

        Args:
            sample_size: Number of queries to analyze per worker type
        """
        logger.info("Starting sentence structure analysis...")

        try:
            # Load data
            self.load_data()

            # Get samples for each worker type
            function_calling_queries = self.get_worker_samples("FUNCTION_CALLING", sample_size)
            doc_search_queries = self.get_worker_samples("DOC_SEARCH", sample_size)

            logger.info(f"Analyzing {len(function_calling_queries)} FUNCTION_CALLING and {len(doc_search_queries)} DOC_SEARCH queries")

            # Analyze queries in parallel
            function_calling_results = self.analyze_queries_parallel(function_calling_queries)
            doc_search_results = self.analyze_queries_parallel(doc_search_queries)

            # Save final results
            self.save_final_results(function_calling_results, doc_search_results)

            logger.info("Sentence structure analysis completed successfully!")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


def main():
    """Main entry point for the analysis."""
    analyzer = SentenceStructureAnalyzer(data_path="../data/dataset.xlsx")
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
