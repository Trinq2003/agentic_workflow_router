"""
Sequential Processor for NetMind Workflow

This module provides sequential processing capabilities for data operations,
optimized for memory efficiency and predictable execution.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Iterator
from pathlib import Path
import pandas as pd

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
class SequentialConfig:
    """Configuration for sequential processing."""

    chunk_size: int = 1000
    enable_progress: bool = True
    progress_interval: int = 100
    enable_caching: bool = True
    timeout: Optional[float] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.progress_interval < 1:
            raise ValueError("progress_interval must be >= 1")


@dataclass
class SequentialResult:
    """Result of sequential processing operation."""

    operation: str
    total_items: int
    processed_items: int
    failed_items: int
    duration: float
    results: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SequentialProcessor:
    """
    Sequential processor for data operations.

    Provides reliable, memory-efficient sequential processing with
    comprehensive progress tracking and error handling.
    """

    def __init__(self, config: SequentialConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._cache: Dict[str, Any] = {}

    def process_in_sequence(self,
                           items: List[Any],
                           process_func: Callable[[Any], Any],
                           operation_name: str = "sequential_processing") -> SequentialResult:
        """
        Process items sequentially with progress tracking.

        Args:
            items: List of items to process
            process_func: Function to apply to each item
            operation_name: Name of the operation for logging

        Returns:
            SequentialResult with results and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting sequential processing: {operation_name} with {len(items)} items")

        results = []
        errors = []
        processed_count = 0

        for idx, item in enumerate(items):
            try:
                if self.config.timeout:
                    # Timeout handling could be implemented with threading
                    result = process_func(item)
                else:
                    result = process_func(item)

                results.append(result)
                processed_count += 1

                # Progress reporting
                if (self.config.enable_progress and
                    (idx + 1) % self.config.progress_interval == 0):
                    progress = (idx + 1) / len(items) * 100
                    self.logger.info(
                        ".1f"
                    )

            except Exception as e:
                error_msg = f"Failed to process item {idx}: {e}"
                errors.append(error_msg)
                self.logger.warning(error_msg)

                # Continue processing other items unless critical error
                if "critical" in str(e).lower():
                    self.logger.error("Critical error encountered, stopping processing")
                    break

        duration = time.time() - start_time

        result = SequentialResult(
            operation=operation_name,
            total_items=len(items),
            processed_items=len(results),
            failed_items=len(errors),
            duration=duration,
            results=results,
            errors=errors,
            metadata={
                'success_rate': len(results) / len(items) if items else 0,
                'items_per_second': len(results) / duration if duration > 0 else 0,
                'processing_efficiency': len(results) / len(items) if items else 0,
            }
        )

        self.logger.info(
            f"Sequential processing completed: {result.processed_items}/{result.total_items} "
            ".2f"
        )

        return result

    def process_data_chunks(self,
                           data_loader: DataLoader,
                           process_func: Callable[[List[QueryData]], Any],
                           operation_name: str = "chunk_processing") -> SequentialResult:
        """
        Process data in chunks sequentially for memory efficiency.

        Args:
            data_loader: DataLoader instance
            process_func: Function to process each chunk
            operation_name: Name of the operation

        Returns:
            SequentialResult with chunked processing results
        """
        self.logger.info(f"Starting sequential chunked processing: {operation_name}")

        # Get processed data
        processed_data = data_loader.get_processed_data()

        # Create chunks
        chunks = []
        for i in range(0, len(processed_data), self.config.chunk_size):
            chunk = processed_data[i:i + self.config.chunk_size]
            chunks.append(chunk)

        self.logger.info(f"Created {len(chunks)} chunks of size ~{self.config.chunk_size}")

        # Process chunks sequentially
        def process_chunk_wrapper(chunk: List[QueryData]) -> Any:
            return process_func(chunk)

        return self.process_in_sequence(chunks, process_chunk_wrapper, operation_name)

    def process_dataframe_batches(self,
                                data_loader: DataLoader,
                                process_func: Callable[[pd.DataFrame], Any],
                                batch_column: str = None,
                                operation_name: str = "dataframe_processing") -> SequentialResult:
        """
        Process pandas DataFrame in batches sequentially.

        Args:
            data_loader: DataLoader instance
            process_func: Function to process each batch
            batch_column: Column to group by for batching (optional)
            operation_name: Name of the operation

        Returns:
            SequentialResult with batch processing results
        """
        self.logger.info(f"Starting sequential DataFrame batch processing: {operation_name}")

        # Get DataFrame
        df = data_loader.to_dataframe()

        if batch_column and batch_column in df.columns:
            # Group by column and process each group
            batches = []
            for group_name, group_df in df.groupby(batch_column):
                batches.append((group_name, group_df))
            self.logger.info(f"Processing {len(batches)} batches by {batch_column}")
        else:
            # Create batches of fixed size
            batches = []
            for i in range(0, len(df), self.config.chunk_size):
                batch_df = df.iloc[i:i + self.config.chunk_size]
                batches.append((f"batch_{i//self.config.chunk_size}", batch_df))
            self.logger.info(f"Processing {len(batches)} batches of size ~{self.config.chunk_size}")

        def process_batch_wrapper(batch_item: tuple) -> Any:
            batch_name, batch_df = batch_item
            return process_func(batch_df)

        return self.process_in_sequence(batches, process_batch_wrapper, operation_name)

    def stream_process_data(self,
                           data_loader: DataLoader,
                           process_func: Callable[[QueryData], Any],
                           operation_name: str = "stream_processing") -> Iterator[Any]:
        """
        Stream process data one item at a time for memory efficiency.

        Args:
            data_loader: DataLoader instance
            process_func: Function to process each item
            operation_name: Name of the operation

        Yields:
            Processed results one at a time
        """
        self.logger.info(f"Starting streaming processing: {operation_name}")

        # Get processed data
        processed_data = data_loader.get_processed_data()

        processed_count = 0
        for item in processed_data:
            try:
                result = process_func(item)
                processed_count += 1

                # Progress reporting
                if (self.config.enable_progress and
                    processed_count % self.config.progress_interval == 0):
                    progress = processed_count / len(processed_data) * 100
                    self.logger.info(
                        ".1f"
                    )

                yield result

            except Exception as e:
                self.logger.warning(f"Failed to process item {processed_count}: {e}")
                # Continue with next item

        self.logger.info(f"Streaming processing completed: {processed_count} items processed")

    def cached_process(self,
                      cache_key: str,
                      process_func: Callable[[], Any],
                      force_refresh: bool = False) -> Any:
        """
        Process with caching for expensive operations.

        Args:
            cache_key: Unique key for caching
            process_func: Function to execute if cache miss
            force_refresh: Force cache refresh

        Returns:
            Cached or newly computed result
        """
        if not self.config.enable_caching:
            return process_func()

        if not force_refresh and cache_key in self._cache:
            self.logger.debug(f"Using cached result for: {cache_key}")
            return self._cache[cache_key]

        self.logger.debug(f"Computing new result for: {cache_key}")
        result = process_func()

        if self.config.enable_caching:
            self._cache[cache_key] = result

        return result

    def clear_cache(self):
        """Clear all cached results."""
        cache_size = len(self._cache)
        self._cache.clear()
        self.logger.info(f"Cleared cache: {cache_size} items removed")


# Utility functions for common sequential operations

def sequential_data_validation(data_loader: DataLoader,
                             config: SequentialConfig = None) -> SequentialResult:
    """
    Validate data sequentially with detailed progress tracking.

    Args:
        data_loader: DataLoader instance
        config: Sequential processing configuration

    Returns:
        SequentialResult with validation results
    """
    if config is None:
        config = SequentialConfig(enable_progress=True, progress_interval=500)

    processor = SequentialProcessor(config)

    def validate_query_data(query_data: QueryData) -> Dict[str, Any]:
        """Validate a single QueryData object."""
        validation_result = {
            'query_id': query_data.index,
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        # Comprehensive validation checks
        if not query_data.query or len(query_data.query.strip()) == 0:
            validation_result['is_valid'] = False
            validation_result['errors'].append('Empty query')

        if not query_data.knowledge_domain or len(query_data.knowledge_domain.strip()) == 0:
            validation_result['is_valid'] = False
            validation_result['errors'].append('Empty knowledge domain')

        if not query_data.workers:
            validation_result['is_valid'] = False
            validation_result['errors'].append('No workers assigned')

        # Length checks
        if len(query_data.query) > 10000:
            validation_result['warnings'].append('Very long query text')

        if len(query_data.workers) > 10:
            validation_result['warnings'].append('Many workers assigned')

        # Domain format validation
        if not query_data.knowledge_domain.replace('_', '').replace('-', '').isalnum():
            validation_result['warnings'].append('Domain contains special characters')

        # Worker name validation
        for worker in query_data.workers:
            if not worker.replace('_', '').replace('-', '').isalnum():
                validation_result['warnings'].append(f'Worker name contains special characters: {worker}')

        return validation_result

    # Get data to validate
    data = data_loader.get_processed_data()

    return processor.process_in_sequence(
        data,
        validate_query_data,
        "data_validation"
    )


def sequential_data_analysis(data_loader: DataLoader,
                           config: SequentialConfig = None) -> SequentialResult:
    """
    Perform comprehensive data analysis sequentially.

    Args:
        data_loader: DataLoader instance
        config: Sequential processing configuration

    Returns:
        SequentialResult with analysis results
    """
    if config is None:
        config = SequentialConfig(enable_progress=True, progress_interval=1000)

    processor = SequentialProcessor(config)

    def analyze_data_chunk(chunk: List[QueryData]) -> Dict[str, Any]:
        """Analyze a chunk of data comprehensively."""
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

        # Worker distribution in chunk
        worker_counts = {}
        for q in chunk:
            for w in q.workers:
                worker_counts[w] = worker_counts.get(w, 0) + 1
        analysis['worker_distribution'] = worker_counts

        # Query length statistics
        query_lengths = [len(q.query) for q in chunk]
        analysis['query_length_stats'] = {
            'min': min(query_lengths),
            'max': max(query_lengths),
            'avg': sum(query_lengths) / len(query_lengths),
        }

        # Assignment statistics
        assignments_per_query = [len(q.workers) for q in chunk]
        analysis['assignment_stats'] = {
            'min': min(assignments_per_query),
            'max': max(assignments_per_query),
            'avg': sum(assignments_per_query) / len(assignments_per_query),
        }

        return analysis

    return processor.process_data_chunks(
        data_loader,
        analyze_data_chunk,
        "comprehensive_data_analysis"
    )


def sequential_data_export(data_loader: DataLoader,
                         export_formats: List[str] = None,
                         output_dir: str = "outputs",
                         config: SequentialConfig = None) -> SequentialResult:
    """
    Export data sequentially to multiple formats with progress tracking.

    Args:
        data_loader: DataLoader instance
        export_formats: List of export formats
        output_dir: Output directory
        config: Sequential processing configuration

    Returns:
        SequentialResult with export results
    """
    if config is None:
        config = SequentialConfig(enable_progress=True, progress_interval=10)

    if export_formats is None:
        export_formats = ['csv', 'json']

    processor = SequentialProcessor(config)

    def export_data_batch(batch_df: pd.DataFrame) -> Dict[str, Any]:
        """Export a batch of data."""
        from pathlib import Path
        import time

        export_results = {}
        timestamp = int(time.time())
        batch_id = f"batch_{timestamp}"

        for fmt in export_formats:
            try:
                output_path = Path(output_dir) / f"{batch_id}.{fmt}"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                if fmt == 'csv':
                    batch_df.to_csv(output_path, index=False)
                    file_size = output_path.stat().st_size
                elif fmt == 'json':
                    batch_df.to_json(output_path, orient='records', indent=2)
                    file_size = output_path.stat().st_size
                elif fmt == 'parquet':
                    batch_df.to_parquet(output_path, index=False)
                    file_size = output_path.stat().st_size
                elif fmt == 'excel':
                    batch_df.to_excel(output_path, index=False)
                    file_size = output_path.stat().st_size
                else:
                    export_results[f"{fmt}_error"] = f"Unsupported format: {fmt}"
                    continue

                export_results[fmt] = {
                    'path': str(output_path),
                    'size_bytes': file_size,
                    'records': len(batch_df),
                }

            except Exception as e:
                export_results[f"{fmt}_error"] = str(e)

        return export_results

    return processor.process_dataframe_batches(
        data_loader,
        export_data_batch,
        operation_name="data_export"
    )
