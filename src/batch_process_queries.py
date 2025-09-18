#!/usr/bin/env python3
"""
Batch Process Queries with Parallel NLP Processing

This script processes queries from the dataset using parallel processing
and saves the results to output files with tqdm progress bars.

Features:
- Loads first 100 queries from dataset.xlsx (configurable)
- Processes queries in parallel using 6 jobs
- Applies comprehensive NLP analysis to each query
- Saves results to JSON/CSV files after each batch (incremental saving)
- Memory-efficient processing with configurable batch sizes
- Beautiful tqdm progress bars for real-time progress tracking
"""

import json
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import pandas as pd
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

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
class QueryNLPResult:
    """Combined result for a query with both metadata and NLP analysis."""

    # Original query data
    query_id: int
    original_query: str
    knowledge_domain: str
    workers: List[str]

    # Language detection
    detected_language: str
    language_confidence: float

    # NLP Results
    tokens: List[str]
    pos_tags: List[List[str]]  # [[token, tag], ...]
    entities: List[Dict[str, Any]]
    sentiment: Dict[str, float]

    # Comprehensive analysis
    word_count: int
    sentence_count: int
    avg_word_length: float
    sentiment_score: float
    keywords: List[str]
    topics: List[str]
    grammar_issues: List[str]

    # Processing metadata
    processing_time: float
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class BatchQueryProcessor:
    """
    Batch processor for all queries with parallel processing.

    Processes all queries from dataset.xlsx using parallel execution
    and saves comprehensive results to output files.
    """

    def __init__(self,
                 data_path: str = r"C:\Users\ADMIN\Code\VTNET\netmind_workflow\data\dataset.xlsx",
                 output_path: str = "query_nlp_results.json",
                 max_workers: int = 6,
                 batch_size: int = 100):
        """
        Initialize the batch processor.

        Args:
            data_path: Path to the dataset Excel file
            output_path: Path to save the results
            max_workers: Number of parallel workers
            batch_size: Size of batches for processing
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.max_workers = max_workers
        self.batch_size = batch_size

        # Initialize components
        self.data_loader = None
        self.nlp_processor = None

        # Progress tracking
        self.total_queries = 0
        self.processed_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0

        logger.info(f"Initialized BatchQueryProcessor with {max_workers} workers")

    def setup(self):
        """Set up the processing environment."""
        logger.info("Setting up batch processing environment...")

        # Initialize data loader
        config = DataLoaderConfig(
            file_path=self.data_path,
            max_workers=4,
            chunk_size=1000,
            cache_enabled=True
        )
        self.data_loader = DataLoader(config)

        # Initialize NLP processor with comprehensive settings
        nlp_config = NLPConfig(
            auto_detect_language=True,
            max_text_length=2000,  # Allow longer queries
            enable_caching=False  # Disable caching for parallel processing
        )
        self.nlp_processor = NLPProcessor(nlp_config)

        logger.info("Batch processing environment setup complete")

    def load_all_queries(self, limit: int = 100) -> List[QueryData]:
        """Load queries from the dataset (limited to first N queries)."""
        logger.info(f"Loading first {limit} queries from dataset...")

        all_queries = self.data_loader.get_processed_data()
        # Limit to first N queries
        limited_queries = all_queries[:limit]
        self.total_queries = len(limited_queries)

        logger.info(f"Loaded {self.total_queries} queries for processing (limited to first {limit})")
        return limited_queries

    def process_single_query(self, query_data: QueryData) -> QueryNLPResult:
        """
        Process a single query with comprehensive NLP analysis.

        Args:
            query_data: QueryData object to process

        Returns:
            QueryNLPResult: Comprehensive processing result
        """
        start_time = time.time()

        try:
            # Basic NLP processing
            basic_result = self.nlp_processor.process_text(
                query_data.query,
                techniques=[
                    NLPTechnique.TOKENIZATION,
                    NLPTechnique.POS_TAGGING,
                    NLPTechnique.NAMED_ENTITY_RECOGNITION,
                    NLPTechnique.SENTIMENT_ANALYSIS
                ]
            )

            # Comprehensive analysis
            comprehensive_analysis = self.nlp_processor.analyze_text_comprehensive(query_data.query)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Create result object
            result = QueryNLPResult(
                query_id=query_data.index,
                original_query=query_data.query,
                knowledge_domain=query_data.knowledge_domain,
                workers=query_data.workers,
                detected_language=basic_result.language.value,
                language_confidence=1.0,  # Simplified confidence
                tokens=basic_result.tokens,
                pos_tags=[[token, tag] for token, tag in basic_result.pos_tags],
                entities=basic_result.entities,
                sentiment=basic_result.sentiment,
                word_count=comprehensive_analysis.word_count,
                sentence_count=comprehensive_analysis.sentence_count,
                avg_word_length=comprehensive_analysis.avg_word_length,
                sentiment_score=comprehensive_analysis.sentiment_score,
                keywords=comprehensive_analysis.keywords,
                topics=comprehensive_analysis.topics,
                grammar_issues=comprehensive_analysis.grammar_issues,
                processing_time=processing_time,
                success=True
            )

            return result

        except Exception as e:
            # Handle processing errors
            processing_time = time.time() - start_time
            logger.warning(f"Failed to process query {query_data.index}: {e}")

            return QueryNLPResult(
                query_id=query_data.index,
                original_query=query_data.query,
                knowledge_domain=query_data.knowledge_domain,
                workers=query_data.workers,
                detected_language="unknown",
                language_confidence=0.0,
                tokens=[],
                pos_tags=[],
                entities=[],
                sentiment={},
                word_count=0,
                sentence_count=0,
                avg_word_length=0.0,
                sentiment_score=0.0,
                keywords=[],
                topics=[],
                grammar_issues=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

    def process_batch_parallel(self, queries: List[QueryData], batch_num: int, total_batches: int) -> List[QueryNLPResult]:
        """
        Process a batch of queries using parallel execution with tqdm progress bar.

        Args:
            queries: List of QueryData objects to process
            batch_num: Current batch number
            total_batches: Total number of batches

        Returns:
            List[QueryNLPResult]: List of processing results
        """
        batch_desc = f"Batch {batch_num}/{total_batches}"
        logger.info(f"Processing {batch_desc}: {len(queries)} queries with {self.max_workers} workers...")

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_query = {
                executor.submit(self.process_single_query, query): query
                for query in queries
            }

            # Create tqdm progress bar for this batch
            with tqdm(total=len(queries), desc=batch_desc, unit="query", leave=True) as pbar:
                # Collect results as they complete
                for future in as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        result = future.result()
                        results.append(result)

                        # Update progress
                        self.processed_queries += 1
                        if result.success:
                            self.successful_queries += 1
                        else:
                            self.failed_queries += 1

                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'success': self.successful_queries,
                            'failed': self.failed_queries
                        })

                    except Exception as e:
                        logger.error(f"Unexpected error processing query {query.index}: {e}")
                        self.processed_queries += 1
                        self.failed_queries += 1
                        pbar.update(1)

        logger.info(f"{batch_desc} processing complete: {len(results)} results")
        return results

    def save_results_json(self, results: List[QueryNLPResult], append: bool = False):
        """Save results to JSON file. Can append to existing file or create new one."""
        if append:
            # Load existing data and append new results
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                existing_results = existing_data.get("results", [])
                existing_results.extend([result.to_dict() for result in results])
                existing_data["results"] = existing_results
                # Update metadata
                existing_data["metadata"].update({
                    "processed_queries": self.processed_queries,
                    "successful_queries": self.successful_queries,
                    "failed_queries": self.failed_queries,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                results_dict = existing_data
            except (FileNotFoundError, json.JSONDecodeError):
                # Create new file if it doesn't exist or is corrupted
                append = False

        if not append:
            # Create new results dict
            results_dict = {
                "metadata": {
                    "total_queries": self.total_queries,
                    "processed_queries": self.processed_queries,
                    "successful_queries": self.successful_queries,
                    "failed_queries": self.failed_queries,
                    "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "parallel_workers": self.max_workers,
                    "batch_size": self.batch_size
                },
                "results": [result.to_dict() for result in results]
            }

        # Save to JSON file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(results)} results to {self.output_path} (total processed: {self.processed_queries})")

    def save_results_csv(self, results: List[QueryNLPResult], csv_path: Optional[Path] = None, append: bool = False):
        """Save results to CSV file for easier analysis. Can append to existing file."""
        if csv_path is None:
            csv_path = self.output_path.with_suffix('.csv')

        # Prepare CSV data
        csv_data = []
        for result in results:
            row = {
                'query_id': result.query_id,
                'original_query': result.original_query,
                'knowledge_domain': result.knowledge_domain,
                'workers': ', '.join(result.workers),
                'detected_language': result.detected_language,
                'language_confidence': result.language_confidence,
                'token_count': len(result.tokens),
                'entity_count': len(result.entities),
                'sentiment_score': result.sentiment_score,
                'word_count': result.word_count,
                'keyword_count': len(result.keywords),
                'topic_count': len(result.topics),
                'processing_time': result.processing_time,
                'success': result.success,
                'error_message': result.error_message or ''
            }
            csv_data.append(row)

        # Save to CSV
        df = pd.DataFrame(csv_data)
        if append and csv_path.exists():
            # Append to existing CSV
            df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            # Create new CSV
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        logger.info(f"Saved {len(results)} results to CSV {csv_path} (mode: {'append' if append else 'create'})")

    def print_summary(self, total_time: float):
        """Print processing summary."""
        print("\n" + "="*80)
        print("BATCH QUERY PROCESSING SUMMARY")
        print("="*80)

        print(f"\n📊 OVERVIEW:")
        print(f"   Total queries: {self.total_queries}")
        print(f"   Processed queries: {self.processed_queries}")
        print(f"   Successful: {self.successful_queries}")
        print(f"   Failed: {self.failed_queries}")
        print(".2f")

        success_rate = (self.successful_queries / self.total_queries) * 100 if self.total_queries > 0 else 0
        print(".2f")

        print(f"\n⚡ PERFORMANCE:")
        print(".2f")
        print(".4f")

        throughput = self.total_queries / total_time if total_time > 0 else 0
        print(".2f")

        print(f"\n💾 OUTPUT FILES:")
        print(f"   JSON results: {self.output_path}")
        csv_path = self.output_path.with_suffix('.csv')
        print(f"   CSV summary: {csv_path}")

        print("\n" + "="*80)

    def run_batch_processing(self):
        """Run the complete batch processing pipeline."""
        logger.info("Starting batch query processing...")
        start_time = time.time()

        try:
            # Setup
            self.setup()

            # Load all queries (limited to first 100)
            all_queries = self.load_all_queries(limit=100)

            # Calculate total batches
            total_batches = (len(all_queries) + self.batch_size - 1) // self.batch_size

            # Process queries in batches for memory efficiency
            all_results = []
            batch_num = 0

            for i in range(0, len(all_queries), self.batch_size):
                batch_num += 1
                batch_queries = all_queries[i:i + self.batch_size]

                logger.info(f"Starting batch {batch_num}/{total_batches}: queries {i}-{min(i + self.batch_size, len(all_queries))}")

                # Process batch with tqdm progress bar
                batch_results = self.process_batch_parallel(batch_queries, batch_num, total_batches)
                all_results.extend(batch_results)

                # Save results after each batch (append mode for subsequent batches)
                append_mode = batch_num > 1
                self.save_results_json(batch_results, append=append_mode)
                self.save_results_csv(batch_results, append=append_mode)

            # Calculate total time
            total_time = time.time() - start_time

            # Print summary
            self.print_summary(total_time)

            logger.info("Batch processing completed successfully!")

            return all_results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise


def main():
    """Main function to run batch processing."""
    print("🚀 Starting Batch Query Processing...")

    # Initialize batch processor
    processor = BatchQueryProcessor(
        data_path=r"C:\Users\ADMIN\Code\VTNET\netmind_workflow\data\dataset.xlsx",
        output_path=r"C:\Users\ADMIN\Code\VTNET\netmind_workflow\reports\query_nlp_results_100.json",
        max_workers=6,  # 6 parallel jobs as requested
        batch_size=50  # Process in batches of 50 for 100 total queries
    )

    try:
        # Run batch processing
        results = processor.run_batch_processing()

        print(f"\n✅ Batch processing completed successfully!")
        print(f"   Processed {len(results)} queries")
        print(f"   Results saved to: {processor.output_path}")

        return 0

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        print(f"\n💥 Batch processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
