"""
Parallel Processor for NetMind Workflow

This module provides parallel processing capabilities for data operations,
optimized for CPU-bound and I/O-bound tasks.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Iterator
import multiprocessing
import threading

import sys
from pathlib import Path

# Add src to path for imports
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from data.data_loader import DataLoader, QueryData

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""

    max_workers: int = min(4, multiprocessing.cpu_count())
    chunk_size: int = 1000
    use_processes: bool = False  # True for CPU-bound, False for I/O-bound
    timeout: Optional[float] = None
    enable_progress: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")


@dataclass
class ProcessingResult:
    """Result of parallel processing operation."""

    operation: str
    total_items: int
    processed_items: int
    failed_items: int
    duration: float
    results: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParallelProcessor:
    """
    Parallel processor for data operations.

    Provides optimized parallel processing for various data operations
    with support for both thread-based and process-based execution.
    """

    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def process_in_parallel(self,
                           items: List[Any],
                           process_func: Callable[[Any], Any],
                           operation_name: str = "parallel_processing") -> ProcessingResult:
        """
        Process items in parallel using the configured execution mode.

        Args:
            items: List of items to process
            process_func: Function to apply to each item
            operation_name: Name of the operation for logging

        Returns:
            ProcessingResult with results and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting parallel processing: {operation_name} with {len(items)} items")

        # Choose executor based on configuration
        executor_class = ProcessPoolExecutor if self.config.use_processes else ThreadPoolExecutor
        max_workers = min(self.config.max_workers, len(items))

        results = []
        errors = []

        try:
            with executor_class(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_item = {
                    executor.submit(self._safe_process_item, process_func, item, idx): (item, idx)
                    for idx, item in enumerate(items)
                }

                # Process results as they complete
                processed_count = 0
                for future in as_completed(future_to_item):
                    item, idx = future_to_item[future]

                    try:
                        result = future.result(timeout=self.config.timeout)
                        results.append(result)
                        processed_count += 1

                        if self.config.enable_progress and processed_count % 100 == 0:
                            self.logger.info(f"Processed {processed_count}/{len(items)} items")

                    except Exception as e:
                        error_msg = f"Failed to process item {idx}: {e}"
                        errors.append(error_msg)
                        self.logger.warning(error_msg)

        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            raise

        duration = time.time() - start_time

        result = ProcessingResult(
            operation=operation_name,
            total_items=len(items),
            processed_items=len(results),
            failed_items=len(errors),
            duration=duration,
            results=results,
            errors=errors,
            metadata={
                'max_workers': max_workers,
                'executor_type': 'process' if self.config.use_processes else 'thread',
                'success_rate': len(results) / len(items) if items else 0,
                'items_per_second': len(results) / duration if duration > 0 else 0,
            }
        )

        self.logger.info(
            f"Parallel processing completed: {result.processed_items}/{result.total_items} "
            ".2f"
        )

        return result

    def _safe_process_item(self, process_func: Callable, item: Any, idx: int) -> Any:
        """Safely process a single item with error handling."""
        try:
            return process_func(item)
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Processing failed for item {idx}: {e}") from e

    def process_data_chunks(self,
                           data_loader: DataLoader,
                           process_func: Callable[[List[QueryData]], Any],
                           operation_name: str = "chunk_processing") -> ProcessingResult:
        """
        Process data in chunks for memory efficiency.

        Args:
            data_loader: DataLoader instance
            process_func: Function to process each chunk
            operation_name: Name of the operation

        Returns:
            ProcessingResult with chunked processing results
        """
        self.logger.info(f"Starting chunked processing: {operation_name}")

        # Get processed data
        processed_data = data_loader.get_processed_data()

        # Create chunks
        chunks = []
        for i in range(0, len(processed_data), self.config.chunk_size):
            chunk = processed_data[i:i + self.config.chunk_size]
            chunks.append(chunk)

        self.logger.info(f"Created {len(chunks)} chunks of size ~{self.config.chunk_size}")

        # Process chunks in parallel
        def process_chunk_wrapper(chunk: List[QueryData]) -> Any:
            return process_func(chunk)

        return self.process_in_parallel(chunks, process_chunk_wrapper, operation_name)

    def batch_process_dataframe(self,
                               data_loader: DataLoader,
                               process_func: Callable[[Any], Any],
                               batch_column: str = None,
                               operation_name: str = "dataframe_processing") -> ProcessingResult:
        """
        Process pandas DataFrame in batches.

        Args:
            data_loader: DataLoader instance
            process_func: Function to process each batch
            batch_column: Column to group by for batching (optional)
            operation_name: Name of the operation

        Returns:
            ProcessingResult with batch processing results
        """
        self.logger.info(f"Starting DataFrame batch processing: {operation_name}")

        # Get DataFrame
        df = data_loader.to_dataframe()

        if batch_column and batch_column in df.columns:
            # Group by column and process each group
            batches = [group for _, group in df.groupby(batch_column)]
            self.logger.info(f"Processing {len(batches)} batches by {batch_column}")
        else:
            # Create batches of fixed size
            batches = []
            for i in range(0, len(df), self.config.chunk_size):
                batch = df.iloc[i:i + self.config.chunk_size]
                batches.append(batch)
            self.logger.info(f"Processing {len(batches)} batches of size ~{self.config.chunk_size}")

        return self.process_in_parallel(batches, process_func, operation_name)


# Utility functions for common parallel operations

def parallel_data_validation(data_loader: DataLoader,
                           config: ParallelConfig = None) -> ProcessingResult:
    """
    Validate data in parallel.

    Args:
        data_loader: DataLoader instance
        config: Parallel processing configuration

    Returns:
        ProcessingResult with validation results
    """
    if config is None:
        config = ParallelConfig(use_processes=False)  # I/O bound

    processor = ParallelProcessor(config)

    def validate_query_data(query_data: QueryData) -> Dict[str, Any]:
        """Validate a single QueryData object."""
        validation_result = {
            'query_id': query_data.index,
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        # Basic validation checks
        if not query_data.query or len(query_data.query.strip()) == 0:
            validation_result['is_valid'] = False
            validation_result['errors'].append('Empty query')

        if not query_data.knowledge_domain or len(query_data.knowledge_domain.strip()) == 0:
            validation_result['is_valid'] = False
            validation_result['errors'].append('Empty knowledge domain')

        if not query_data.workers:
            validation_result['is_valid'] = False
            validation_result['errors'].append('No workers assigned')

        # Additional checks
        if len(query_data.query) > 10000:  # Arbitrary limit
            validation_result['warnings'].append('Very long query text')

        if len(query_data.workers) > 10:  # Arbitrary limit
            validation_result['warnings'].append('Many workers assigned')

        return validation_result

    # Get data to validate
    data = data_loader.get_processed_data()

    return processor.process_in_parallel(
        data,
        validate_query_data,
        "data_validation"
    )


def parallel_data_analysis(data_loader: DataLoader,
                          config: ParallelConfig = None) -> ProcessingResult:
    """
    Perform data analysis in parallel.

    Args:
        data_loader: DataLoader instance
        config: Parallel processing configuration

    Returns:
        ProcessingResult with analysis results
    """
    if config is None:
        config = ParallelConfig(use_processes=True)  # CPU bound analysis

    processor = ParallelProcessor(config)

    def analyze_data_chunk(chunk: List[QueryData]) -> Dict[str, Any]:
        """Analyze a chunk of data."""
        analysis = {
            'chunk_size': len(chunk),
            'avg_query_length': sum(len(q.query) for q in chunk) / len(chunk),
            'unique_domains': len(set(q.knowledge_domain for q in chunk)),
            'unique_workers': len(set(w for q in chunk for w in q.workers)),
            'total_assignments': sum(len(q.workers) for q in chunk),
        }

        # Domain distribution in chunk
        domain_counts = {}
        for q in chunk:
            domain_counts[q.knowledge_domain] = domain_counts.get(q.knowledge_domain, 0) + 1
        analysis['domain_distribution'] = domain_counts

        return analysis

    return processor.process_data_chunks(
        data_loader,
        analyze_data_chunk,
        "data_analysis"
    )


def parallel_data_export(data_loader: DataLoader,
                        export_formats: List[str] = None,
                        output_dir: str = "outputs",
                        config: ParallelConfig = None) -> ProcessingResult:
    """
    Export data in parallel to multiple formats.

    Args:
        data_loader: DataLoader instance
        export_formats: List of export formats
        output_dir: Output directory
        config: Parallel processing configuration

    Returns:
        ProcessingResult with export results
    """
    if config is None:
        config = ParallelConfig(use_processes=False)  # I/O bound

    if export_formats is None:
        export_formats = ['csv', 'json']

    processor = ParallelProcessor(config)

    def export_data_chunk(chunk_df: Any) -> Dict[str, Any]:
        """Export a chunk of data."""
        import pandas as pd
        from pathlib import Path

        export_results = {}
        timestamp = int(time.time())

        for fmt in export_formats:
            try:
                output_path = Path(output_dir) / f"chunk_{timestamp}.{fmt}"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if fmt == 'csv':
                    chunk_df.to_csv(output_path, index=False)
                elif fmt == 'json':
                    chunk_df.to_json(output_path, orient='records', indent=2)
                elif fmt == 'parquet':
                    chunk_df.to_parquet(output_path, index=False)
                else:
                    export_results[f"{fmt}_error"] = f"Unsupported format: {fmt}"
                    continue

                export_results[fmt] = str(output_path)

            except Exception as e:
                export_results[f"{fmt}_error"] = str(e)

        return export_results

    return processor.batch_process_dataframe(
        data_loader,
        export_data_chunk,
        operation_name="data_export"
    )
