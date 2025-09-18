"""
Data Loader Module for NetMind Workflow

This module provides a professional data loading solution for processing
evaluation queries with knowledge domain classification and worker assignments.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Iterator
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    """Configuration class for DataLoader settings."""

    file_path: Union[str, Path]
    sheet_name: Optional[str] = None
    query_column: str = "query"
    knowledge_domain_column: str = "knowledge_domain"
    worker_column: str = "worker"
    worker_separator: str = ","
    cache_enabled: bool = True
    validate_data: bool = True
    max_workers: int = 4
    chunk_size: int = 1000

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.file_path = Path(self.file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")


@dataclass
class QueryData:
    """Data structure representing a single query entry."""

    query: str
    knowledge_domain: str
    workers: List[str]
    index: int = -1

    def __post_init__(self):
        """Validate data after initialization."""
        if not self.query.strip():
            raise ValueError("Query cannot be empty")
        if not self.knowledge_domain.strip():
            raise ValueError("Knowledge domain cannot be empty")
        if not self.workers:
            raise ValueError("Workers list cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query": self.query,
            "knowledge_domain": self.knowledge_domain,
            "workers": self.workers,
            "index": self.index
        }


class DataLoader:
    """
    Professional data loader class for processing evaluation queries.

    Features:
    - Excel file loading with robust error handling
    - Parallel processing capabilities
    - Data validation and cleaning
    - Caching for performance optimization
    - Memory-efficient chunked processing
    - Comprehensive logging and monitoring
    """

    def __init__(self, config: DataLoaderConfig):
        """
        Initialize the DataLoader with configuration.

        Args:
            config: DataLoaderConfig object containing all settings
        """
        self.config = config
        self._data: Optional[pd.DataFrame] = None
        self._processed_data: Optional[List[QueryData]] = None
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info(f"Initialized DataLoader with config: {config}")

    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load data from Excel file with caching.

        Args:
            force_reload: If True, bypass cache and reload data

        Returns:
            pandas.DataFrame: Raw loaded data
        """
        cache_key = f"raw_data_{self.config.file_path}_{self.config.sheet_name}"

        if not force_reload and self.config.cache_enabled and cache_key in self._cache:
            self.logger.info("Using cached raw data")
            return self._cache[cache_key]

        try:
            self.logger.info(f"Loading data from {self.config.file_path}")

            # Load Excel file
            read_params = {
                'engine': 'openpyxl',
                'dtype': str  # Read all as strings to handle mixed data types
            }

            if self.config.sheet_name:
                read_params['sheet_name'] = self.config.sheet_name

            self._data = pd.read_excel(self.config.file_path, **read_params)

            # Cache the data
            if self.config.cache_enabled:
                self._cache[cache_key] = self._data.copy()

            self.logger.info(f"Successfully loaded {len(self._data)} rows")
            return self._data

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns exist in the DataFrame.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = [
            self.config.query_column,
            self.config.knowledge_domain_column,
            self.config.worker_column
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.logger.info("Column validation passed")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the raw data.

        Args:
            df: Raw DataFrame to clean

        Returns:
            pandas.DataFrame: Cleaned DataFrame
        """
        self.logger.info("Starting data cleaning process")

        # Create a copy to avoid modifying original
        cleaned_df = df.copy()

        # Remove rows with all NaN values
        cleaned_df = cleaned_df.dropna(how='all')

        # Fill NaN values with empty strings for string columns
        string_columns = [self.config.query_column, self.config.knowledge_domain_column]
        for col in string_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna('').astype(str).str.strip()

        # Handle worker column specifically
        if self.config.worker_column in cleaned_df.columns:
            cleaned_df[self.config.worker_column] = (
                cleaned_df[self.config.worker_column]
                .fillna('')
                .astype(str)
                .str.strip()
            )

        # Remove completely empty rows after cleaning
        mask = (
            (cleaned_df[self.config.query_column] != '') |
            (cleaned_df[self.config.knowledge_domain_column] != '') |
            (cleaned_df[self.config.worker_column] != '')
        )
        cleaned_df = cleaned_df[mask]

        self.logger.info(f"Data cleaning completed. Rows: {len(cleaned_df)}")
        return cleaned_df

    def parse_workers(self, worker_str: str) -> List[str]:
        """
        Parse comma-separated worker string into a list.

        Args:
            worker_str: Comma-separated string of workers

        Returns:
            List[str]: List of worker names
        """
        if not worker_str or worker_str.strip() == '':
            return []

        # Split by comma and clean up whitespace
        workers = [w.strip() for w in worker_str.split(self.config.worker_separator)]
        # Remove empty strings
        workers = [w for w in workers if w]

        return workers

    def _process_single_row(self, row: pd.Series, index: int) -> QueryData:
        """
        Process a single row into QueryData object.

        Args:
            row: pandas Series representing a single row
            index: Row index

        Returns:
            QueryData: Processed query data object
        """
        try:
            workers = self.parse_workers(row[self.config.worker_column])

            return QueryData(
                query=row[self.config.query_column],
                knowledge_domain=row[self.config.knowledge_domain_column],
                workers=workers,
                index=index
            )
        except Exception as e:
            self.logger.warning(f"Error processing row {index}: {e}")
            raise

    def process_data_parallel(self, df: pd.DataFrame) -> List[QueryData]:
        """
        Process DataFrame into QueryData objects using parallel processing.

        Args:
            df: Cleaned DataFrame to process

        Returns:
            List[QueryData]: List of processed QueryData objects
        """
        self.logger.info(f"Processing {len(df)} rows using parallel processing")

        processed_data = []

        # Process in chunks for memory efficiency
        for chunk_start in range(0, len(df), self.config.chunk_size):
            chunk_end = min(chunk_start + self.config.chunk_size, len(df))
            chunk = df.iloc[chunk_start:chunk_end]

            self.logger.debug(f"Processing chunk {chunk_start}-{chunk_end}")

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks for this chunk
                future_to_index = {
                    executor.submit(self._process_single_row, row, idx): idx
                    for idx, row in chunk.iterrows()
                }

                # Collect results as they complete
                for future in as_completed(future_to_index):
                    try:
                        result = future.result()
                        processed_data.append(result)
                    except Exception as e:
                        index = future_to_index[future]
                        self.logger.error(f"Failed to process row {index}: {e}")
                        raise

        # Sort by index to maintain original order
        processed_data.sort(key=lambda x: x.index)

        self.logger.info(f"Successfully processed {len(processed_data)} query data objects")
        return processed_data

    def process_data_sequential(self, df: pd.DataFrame) -> List[QueryData]:
        """
        Process DataFrame into QueryData objects sequentially.

        Args:
            df: Cleaned DataFrame to process

        Returns:
            List[QueryData]: List of processed QueryData objects
        """
        self.logger.info(f"Processing {len(df)} rows sequentially")

        processed_data = []
        for idx, row in df.iterrows():
            try:
                query_data = self._process_single_row(row, idx)
                processed_data.append(query_data)
            except Exception as e:
                self.logger.error(f"Failed to process row {idx}: {e}")
                raise

        self.logger.info(f"Successfully processed {len(processed_data)} query data objects")
        return processed_data

    def get_processed_data(self, force_reprocess: bool = False) -> List[QueryData]:
        """
        Get processed QueryData objects with optional caching.

        Args:
            force_reprocess: If True, bypass cache and reprocess data

        Returns:
            List[QueryData]: List of processed QueryData objects
        """
        cache_key = f"processed_data_{self.config.file_path}"

        if not force_reprocess and self.config.cache_enabled and cache_key in self._cache:
            self.logger.info("Using cached processed data")
            return self._cache[cache_key]

        # Load and clean raw data
        raw_data = self.load_data(force_reload=force_reprocess)

        if self.config.validate_data:
            self.validate_columns(raw_data)

        cleaned_data = self.clean_data(raw_data)

        # Process data (choose parallel or sequential based on data size)
        if len(cleaned_data) > 100:  # Use parallel processing for larger datasets
            processed_data = self.process_data_parallel(cleaned_data)
        else:
            processed_data = self.process_data_sequential(cleaned_data)

        # Cache processed data
        if self.config.cache_enabled:
            self._cache[cache_key] = processed_data

        self._processed_data = processed_data
        return processed_data

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded data.

        Returns:
            Dict[str, Any]: Dictionary containing various statistics
        """
        if self._processed_data is None:
            self.get_processed_data()

        if not self._processed_data:
            return {"error": "No data available"}

        # Calculate statistics
        stats = {
            "total_queries": len(self._processed_data),
            "unique_knowledge_domains": len(set(q.knowledge_domain for q in self._processed_data)),
            "unique_workers": len(set(w for q in self._processed_data for w in q.workers)),
            "knowledge_domain_distribution": {},
            "worker_distribution": {},
            "queries_per_domain": {},
            "workers_per_query": {}
        }

        # Knowledge domain distribution
        for query in self._processed_data:
            domain = query.knowledge_domain
            stats["knowledge_domain_distribution"][domain] = (
                stats["knowledge_domain_distribution"].get(domain, 0) + 1
            )

        # Worker distribution
        for query in self._processed_data:
            for worker in query.workers:
                stats["worker_distribution"][worker] = (
                    stats["worker_distribution"].get(worker, 0) + 1
                )

        # Additional metrics
        worker_counts = [len(q.workers) for q in self._processed_data]
        stats["workers_per_query"] = {
            "min": min(worker_counts),
            "max": max(worker_counts),
            "avg": np.mean(worker_counts),
            "median": np.median(worker_counts)
        }

        return stats

    def iterate_batches(self, batch_size: int = 100) -> Iterator[List[QueryData]]:
        """
        Iterate over data in batches for memory-efficient processing.

        Args:
            batch_size: Size of each batch

        Yields:
            List[QueryData]: Batch of QueryData objects
        """
        if self._processed_data is None:
            self.get_processed_data()

        for i in range(0, len(self._processed_data), batch_size):
            yield self._processed_data[i:i + batch_size]

    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._data = None
            self._processed_data = None
        self.logger.info("Cache cleared")

    def __len__(self) -> int:
        """Return the number of processed data items."""
        if self._processed_data is None:
            self.get_processed_data()
        return len(self._processed_data) if self._processed_data else 0

    def __getitem__(self, index: int) -> QueryData:
        """Get a specific QueryData item by index."""
        if self._processed_data is None:
            self.get_processed_data()
        return self._processed_data[index]

    def __iter__(self) -> Iterator[QueryData]:
        """Iterate over all QueryData items."""
        if self._processed_data is None:
            self.get_processed_data()
        return iter(self._processed_data)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert processed data to pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns [query, knowledge_domain, workers, index]
        """
        if self._processed_data is None:
            self.get_processed_data()

        data = []
        for query_data in self._processed_data:
            row = {
                'query': query_data.query,
                'knowledge_domain': query_data.knowledge_domain,
                'workers': ', '.join(query_data.workers),  # Join workers back to string
                'workers_list': query_data.workers,  # Keep as list for analysis
                'index': query_data.index
            }
            data.append(row)

        return pd.DataFrame(data)

    def get_raw_dataframe(self) -> pd.DataFrame:
        """
        Get the raw pandas DataFrame after cleaning.

        Returns:
            pd.DataFrame: Raw cleaned DataFrame
        """
        if self._data is None:
            self.load_data()

        return self.clean_data(self._data)

    def get_domain_dataframe(self, domain: str) -> pd.DataFrame:
        """
        Get DataFrame filtered by knowledge domain.

        Args:
            domain: Knowledge domain to filter by

        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        df = self.to_dataframe()
        return df[df['knowledge_domain'] == domain].copy()

    def get_worker_dataframe(self, worker: str) -> pd.DataFrame:
        """
        Get DataFrame filtered by worker.

        Args:
            worker: Worker name to filter by

        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        df = self.to_dataframe()
        return df[df['workers_list'].apply(lambda x: worker in x)].copy()

    def group_by_domain(self) -> pd.DataFrame:
        """
        Group data by knowledge domain and return aggregated statistics.

        Returns:
            pd.DataFrame: Aggregated statistics by domain
        """
        df = self.to_dataframe()

        # Group by domain and calculate statistics
        grouped = df.groupby('knowledge_domain').agg({
            'query': 'count',
            'workers': lambda x: len(set(', '.join(x).split(', '))),  # Unique workers per domain
        }).rename(columns={
            'query': 'query_count',
            'workers': 'unique_workers_count'
        })

        # Add worker lists per domain
        worker_lists = df.groupby('knowledge_domain')['workers_list'].apply(list)
        grouped['worker_lists'] = worker_lists

        return grouped.reset_index()

    def group_by_worker(self) -> pd.DataFrame:
        """
        Group data by worker and return aggregated statistics.

        Returns:
            pd.DataFrame: Aggregated statistics by worker
        """
        df = self.to_dataframe()

        # Explode workers list to get one row per worker-query combination
        exploded = df.explode('workers_list')

        # Group by worker and calculate statistics
        grouped = exploded.groupby('workers_list').agg({
            'query': 'count',
            'knowledge_domain': lambda x: len(set(x)),  # Unique domains per worker
        }).rename(columns={
            'query': 'query_count',
            'knowledge_domain': 'unique_domains_count'
        })

        # Add domain lists per worker
        domain_lists = exploded.groupby('workers_list')['knowledge_domain'].apply(list)
        grouped['domain_lists'] = domain_lists

        return grouped.reset_index().rename(columns={'workers_list': 'worker'})

    def sample_dataframe(self, n: int = 100, random_state: int = 42) -> pd.DataFrame:
        """
        Get a random sample of the data as DataFrame.

        Args:
            n: Number of samples to return
            random_state: Random seed for reproducibility

        Returns:
            pd.DataFrame: Random sample of data
        """
        df = self.to_dataframe()
        return df.sample(n=min(n, len(df)), random_state=random_state).copy()

    def filter_dataframe(self, query_filter: Optional[str] = None,
                        domain_filter: Optional[str] = None,
                        worker_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Filter DataFrame by multiple criteria.

        Args:
            query_filter: String to search in query text (case-insensitive)
            domain_filter: Knowledge domain to filter by
            worker_filter: Worker name to filter by

        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        df = self.to_dataframe()

        if query_filter:
            df = df[df['query'].str.contains(query_filter, case=False, na=False)]

        if domain_filter:
            df = df[df['knowledge_domain'] == domain_filter]

        if worker_filter:
            df = df[df['workers_list'].apply(lambda x: worker_filter in x)]

        return df.copy()


def create_default_dataloader(data_path: Union[str, Path] = "data/dataset.xlsx") -> DataLoader:
    """
    Create a DataLoader with default configuration.

    Args:
        data_path: Path to the Excel data file

    Returns:
        DataLoader: Configured DataLoader instance
    """
    config = DataLoaderConfig(file_path=data_path)
    return DataLoader(config)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create data loader
    loader = create_default_dataloader()

    try:
        # Load and process data
        data = loader.get_processed_data()

        # Print statistics
        stats = loader.get_statistics()
        print(f"Loaded {stats['total_queries']} queries")
        print(f"Knowledge domains: {stats['unique_knowledge_domains']}")
        print(f"Unique workers: {stats['unique_workers']}")

        # Print first few examples
        print("\nFirst 3 queries:")
        for i, query in enumerate(data[:3]):
            print(f"{i+1}. Query: {query.query[:50]}...")
            print(f"   Domain: {query.knowledge_domain}")
            print(f"   Workers: {', '.join(query.workers)}")
            print()

        # Demonstrate pandas DataFrame functionality
        print("\n=== Pandas DataFrame Examples ===")

        # Get full DataFrame
        df = loader.to_dataframe()
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {list(df.columns)}")
        print(f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

        # Show domain distribution
        print("\nKnowledge Domain Distribution:")
        domain_counts = df['knowledge_domain'].value_counts().head(10)
        for domain, count in domain_counts.items():
            print(f"  {domain}: {count} queries")

        # Show worker distribution
        print("\nWorker Distribution:")
        worker_counts = df['workers_list'].explode().value_counts().head(10)
        for worker, count in worker_counts.items():
            print(f"  {worker}: {count} queries")

        # Group by domain analysis
        print("\n=== Group by Domain Analysis ===")
        domain_stats = loader.group_by_domain()
        print("Domain statistics:")
        print(domain_stats.to_string(index=False))

        # Filter examples
        print("\n=== Filtering Examples ===")
        vo_tuyen_df = loader.get_domain_dataframe('VO_TUYEN')
        print(f"VO_TUYEN domain has {len(vo_tuyen_df)} queries")

        # Search for specific terms
        search_df = loader.filter_dataframe(query_filter='Vinaphone')
        print(f"Found {len(search_df)} queries containing 'Vinaphone'")

        # Sample data
        sample_df = loader.sample_dataframe(5)
        print(f"\nRandom sample of {len(sample_df)} queries:")
        for idx, row in sample_df.iterrows():
            print(f"  Query: {row['query'][:60]}...")
            print(f"  Domain: {row['knowledge_domain']}")
            print(f"  Workers: {row['workers']}")
            print()

    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        raise
