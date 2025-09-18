"""
Dagster Assets for Data Processing

This module defines Dagster assets for loading, processing, and analyzing
the NetMind dataset with pandas DataFrames.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

from dagster import asset, AssetIn, AssetOut, multi_asset, Output, MetadataValue
from dagster_pandas import PandasColumn, create_dagster_pandas_dataframe_type

from ..data.data_loader import DataLoader, DataLoaderConfig
from ..models.nlp_models import NLPProcessor, NLPConfig, TextAnalysis, NLPResult

# Configure logging
logger = logging.getLogger(__name__)

# Define custom pandas dataframe types for better type checking
NetMindDataFrame = create_dagster_pandas_dataframe_type(
    name="NetMindDataFrame",
    columns=[
        PandasColumn.string_column("query", non_nullable=True),
        PandasColumn.string_column("knowledge_domain", non_nullable=True),
        PandasColumn.string_column("workers"),
        PandasColumn.integer_column("index"),
    ],
)

DomainStatsDataFrame = create_dagster_pandas_dataframe_type(
    name="DomainStatsDataFrame",
    columns=[
        PandasColumn.string_column("knowledge_domain", non_nullable=True),
        PandasColumn.integer_column("query_count"),
        PandasColumn.integer_column("unique_workers_count"),
    ],
)


@asset(
    name="raw_netmind_data",
    description="Load raw NetMind dataset from Excel file",
    io_manager_key="pandas_io_manager",
    metadata={
        "source": "data/dataset.xlsx",
        "format": "Excel",
    }
)
def raw_netmind_data_asset() -> Output[pd.DataFrame]:
    """
    Asset for loading raw NetMind data from Excel file.

    Returns:
        Raw pandas DataFrame from the Excel file
    """
    logger.info("Loading raw NetMind data from Excel file")

    # Create data loader with minimal configuration
    config = DataLoaderConfig(
        file_path="data/dataset.xlsx",
        cache_enabled=False,  # Disable caching for assets
        validate_data=True,
    )
    loader = DataLoader(config)

    # Load raw data
    raw_df = loader.load_data()

    # Get file metadata
    file_path = Path("data/dataset.xlsx")
    file_size = file_path.stat().st_size if file_path.exists() else 0

    logger.info(f"Loaded raw data: {len(raw_df)} rows, {len(raw_df.columns)} columns")

    return Output(
        value=raw_df,
        metadata={
            "row_count": len(raw_df),
            "column_count": len(raw_df.columns),
            "columns": list(raw_df.columns),
            "file_size_kb": file_size / 1024,
            "memory_usage_mb": raw_df.memory_usage(deep=True).sum() / (1024 * 1024),
        }
    )


@asset(
    name="cleaned_netmind_data",
    description="Clean and preprocess NetMind dataset",
    io_manager_key="pandas_io_manager",
    ins={"raw_data": AssetIn("raw_netmind_data")},
    metadata={
        "cleaning_steps": ["remove_null_rows", "strip_whitespace", "validate_columns"],
    }
)
def cleaned_netmind_data_asset(raw_data: pd.DataFrame) -> Output[pd.DataFrame]:
    """
    Asset for cleaning and preprocessing the raw NetMind data.

    Args:
        raw_data: Raw DataFrame from raw_netmind_data asset

    Returns:
        Cleaned pandas DataFrame
    """
    logger.info("Cleaning NetMind data")

    # Create data loader and apply cleaning
    config = DataLoaderConfig(
        file_path="data/dataset.xlsx",
        cache_enabled=False,
        validate_data=True,
    )
    loader = DataLoader(config)

    # Apply cleaning to the raw data
    cleaned_df = loader.clean_data(raw_data)

    logger.info(f"Cleaned data: {len(cleaned_df)} rows (from {len(raw_data)} raw rows)")

    return Output(
        value=cleaned_df,
        metadata={
            "original_row_count": len(raw_data),
            "cleaned_row_count": len(cleaned_df),
            "rows_removed": len(raw_data) - len(cleaned_df),
            "cleaning_efficiency": len(cleaned_df) / len(raw_data) if len(raw_data) > 0 else 0,
        }
    )


@asset(
    name="processed_netmind_data",
    description="Process NetMind data into structured format",
    io_manager_key="pandas_io_manager",
    ins={"cleaned_data": AssetIn("cleaned_netmind_data")},
    metadata={
        "processing_steps": ["parse_workers", "validate_structure", "add_metadata"],
    }
)
def processed_netmind_data_asset(cleaned_data: pd.DataFrame) -> Output[NetMindDataFrame]:
    """
    Asset for processing cleaned data into structured format with proper types.

    Args:
        cleaned_data: Cleaned DataFrame from cleaned_netmind_data asset

    Returns:
        Processed DataFrame with proper structure
    """
    logger.info("Processing NetMind data into structured format")

    # Create data loader and process data
    config = DataLoaderConfig(
        file_path="data/dataset.xlsx",
        cache_enabled=False,
        validate_data=True,
    )
    loader = DataLoader(config)

    # Process the cleaned data
    processed_objects = loader.process_data_sequential(cleaned_data)

    # Convert to DataFrame
    data = []
    for query_data in processed_objects:
        row = {
            'query': query_data.query,
            'knowledge_domain': query_data.knowledge_domain,
            'workers': ', '.join(query_data.workers),  # Join workers back to string
            'index': query_data.index
        }
        data.append(row)

    processed_df = pd.DataFrame(data)

    # Add additional computed columns
    processed_df['query_length'] = processed_df['query'].str.len()
    processed_df['worker_count'] = processed_df['workers'].str.split(',').str.len()
    processed_df['processed_at'] = pd.Timestamp.now()

    logger.info(f"Processed data: {len(processed_df)} rows with {len(processed_df.columns)} columns")

    return Output(
        value=processed_df,
        metadata={
            "row_count": len(processed_df),
            "column_count": len(processed_df.columns),
            "columns": list(processed_df.columns),
            "avg_query_length": processed_df['query_length'].mean(),
            "avg_workers_per_query": processed_df['worker_count'].mean(),
            "unique_domains": processed_df['knowledge_domain'].nunique(),
            "unique_workers": processed_df['workers'].str.split(',').explode().nunique(),
        }
    )


@asset(
    name="domain_statistics",
    description="Compute statistics grouped by knowledge domain",
    io_manager_key="pandas_io_manager",
    ins={"processed_data": AssetIn("processed_netmind_data")},
    metadata={
        "grouping_column": "knowledge_domain",
        "metrics": ["query_count", "worker_count", "avg_query_length"],
    }
)
def domain_statistics_asset(processed_data: NetMindDataFrame) -> Output[DomainStatsDataFrame]:
    """
    Asset for computing statistics grouped by knowledge domain.

    Args:
        processed_data: Processed DataFrame from processed_netmind_data asset

    Returns:
        DataFrame with domain-level statistics
    """
    logger.info("Computing domain statistics")

    # Group by domain and calculate statistics
    domain_stats = processed_data.groupby('knowledge_domain').agg({
        'query': 'count',
        'workers': lambda x: len(set(', '.join(x).split(', '))),  # Unique workers per domain
        'query_length': 'mean',
        'worker_count': 'mean',
    }).rename(columns={
        'query': 'query_count',
        'workers': 'unique_workers_count',
        'query_length': 'avg_query_length',
        'worker_count': 'avg_workers_per_query'
    })

    # Add percentage columns
    total_queries = domain_stats['query_count'].sum()
    domain_stats['query_percentage'] = (domain_stats['query_count'] / total_queries * 100).round(2)

    # Sort by query count descending
    domain_stats = domain_stats.sort_values('query_count', ascending=False).reset_index()

    logger.info(f"Generated statistics for {len(domain_stats)} knowledge domains")

    return Output(
        value=domain_stats,
        metadata={
            "domain_count": len(domain_stats),
            "total_queries": int(total_queries),
            "top_domain": domain_stats.iloc[0]['knowledge_domain'] if len(domain_stats) > 0 else None,
            "top_domain_queries": int(domain_stats.iloc[0]['query_count']) if len(domain_stats) > 0 else 0,
        }
    )


@asset(
    name="worker_statistics",
    description="Compute statistics grouped by worker",
    io_manager_key="pandas_io_manager",
    ins={"processed_data": AssetIn("processed_netmind_data")},
    metadata={
        "grouping_column": "worker",
        "metrics": ["query_count", "domain_count", "avg_query_length"],
    }
)
def worker_statistics_asset(processed_data: NetMindDataFrame) -> Output[pd.DataFrame]:
    """
    Asset for computing statistics grouped by worker.

    Args:
        processed_data: Processed DataFrame from processed_netmind_data asset

    Returns:
        DataFrame with worker-level statistics
    """
    logger.info("Computing worker statistics")

    # Explode workers to get one row per worker-query combination
    exploded_df = processed_data.copy()
    exploded_df['worker_list'] = exploded_df['workers'].str.split(', ')
    exploded_df = exploded_df.explode('worker_list')

    # Group by worker and calculate statistics
    worker_stats = exploded_df.groupby('worker_list').agg({
        'query': 'count',
        'knowledge_domain': lambda x: len(set(x)),  # Unique domains per worker
        'query_length': 'mean',
        'index': 'count',  # Same as query count
    }).rename(columns={
        'query': 'query_count',
        'knowledge_domain': 'unique_domains_count',
        'query_length': 'avg_query_length',
        'index': 'assignment_count'
    })

    # Add percentage columns
    total_assignments = worker_stats['assignment_count'].sum()
    worker_stats['assignment_percentage'] = (worker_stats['assignment_count'] / total_assignments * 100).round(2)

    # Sort by assignment count descending
    worker_stats = worker_stats.sort_values('assignment_count', ascending=False).reset_index().rename(
        columns={'worker_list': 'worker'}
    )

    logger.info(f"Generated statistics for {len(worker_stats)} workers")

    return Output(
        value=worker_stats,
        metadata={
            "worker_count": len(worker_stats),
            "total_assignments": int(total_assignments),
            "top_worker": worker_stats.iloc[0]['worker'] if len(worker_stats) > 0 else None,
            "top_worker_assignments": int(worker_stats.iloc[0]['assignment_count']) if len(worker_stats) > 0 else 0,
        }
    )


@multi_asset(
    name="data_quality_report",
    description="Generate comprehensive data quality report",
    ins={
        "raw_data": AssetIn("raw_netmind_data"),
        "cleaned_data": AssetIn("cleaned_netmind_data"),
        "processed_data": AssetIn("processed_netmind_data"),
    },
    outs={
        "quality_report": AssetOut(
            io_manager_key="pandas_io_manager",
            metadata={"report_type": "data_quality"}
        ),
        "data_summary": AssetOut(
            io_manager_key="pandas_io_manager",
            metadata={"report_type": "summary"}
        ),
    },
)
def data_quality_report_assets(raw_data: pd.DataFrame, cleaned_data: pd.DataFrame, processed_data: NetMindDataFrame):
    """
    Multi-asset for generating data quality reports.

    Args:
        raw_data: Raw DataFrame
        cleaned_data: Cleaned DataFrame
        processed_data: Processed DataFrame

    Returns:
        Quality report and data summary DataFrames
    """
    logger.info("Generating data quality report")

    # Quality metrics
    quality_report = pd.DataFrame({
        'metric': [
            'total_rows_raw',
            'total_rows_cleaned',
            'total_rows_processed',
            'cleaning_efficiency',
            'processing_efficiency',
            'duplicate_queries',
            'empty_queries',
            'empty_domains',
            'empty_workers',
            'avg_query_length',
            'max_query_length',
            'min_query_length',
            'unique_domains',
            'unique_workers',
            'avg_workers_per_query',
        ],
        'value': [
            len(raw_data),
            len(cleaned_data),
            len(processed_data),
            len(cleaned_data) / len(raw_data) if len(raw_data) > 0 else 0,
            len(processed_data) / len(raw_data) if len(raw_data) > 0 else 0,
            processed_data['query'].duplicated().sum(),
            (processed_data['query'] == '').sum(),
            (processed_data['knowledge_domain'] == '').sum(),
            (processed_data['workers'] == '').sum(),
            processed_data['query_length'].mean(),
            processed_data['query_length'].max(),
            processed_data['query_length'].min(),
            processed_data['knowledge_domain'].nunique(),
            processed_data['workers'].str.split(', ').explode().nunique(),
            processed_data['worker_count'].mean(),
        ]
    })

    # Data summary by domain
    data_summary = processed_data.groupby('knowledge_domain').agg({
        'query': 'count',
        'query_length': ['mean', 'min', 'max'],
        'worker_count': ['mean', 'min', 'max'],
    }).round(2)

    # Flatten column names
    data_summary.columns = ['_'.join(col).strip() for col in data_summary.columns.values]
    data_summary = data_summary.reset_index().sort_values('query_count', ascending=False)

    logger.info("Generated comprehensive data quality report")

    return (
        Output(
            value=quality_report,
            metadata={
                "report_rows": len(quality_report),
                "generated_at": pd.Timestamp.now().isoformat(),
            }
        ),
        Output(
            value=data_summary,
            metadata={
                "summary_rows": len(data_summary),
                "domains_covered": len(data_summary),
                "generated_at": pd.Timestamp.now().isoformat(),
            }
        )
    )


@asset(
    name="data_sample",
    description="Generate random sample of processed data for testing",
    io_manager_key="pandas_io_manager",
    ins={"processed_data": AssetIn("processed_netmind_data")},
    metadata={
        "sample_type": "random",
        "sample_size": 100,
    }
)
def data_sample_asset(processed_data: NetMindDataFrame) -> Output[NetMindDataFrame]:
    """
    Asset for generating a random sample of processed data for testing.

    Args:
        processed_data: Processed DataFrame from processed_netmind_data asset

    Returns:
        Random sample DataFrame
    """
    sample_size = min(100, len(processed_data))
    logger.info(f"Generating random sample of {sample_size} rows")

    sample_df = processed_data.sample(n=sample_size, random_state=42).copy()

    return         Output(
            value=sample_df,
            metadata={
                "sample_size": len(sample_df),
                "total_population": len(processed_data),
                "sampling_ratio": len(sample_df) / len(processed_data),
                "random_state": 42,
            }
        )


# NLP Processing Assets

@asset(
    name="nlp_processor",
    description="Initialize NLP processor with configuration",
    compute_kind="nlp",
    metadata={
        "supported_languages": ["English", "Vietnamese"],
        "techniques": ["tokenization", "sentiment", "ner", "parsing"]
    }
)
def nlp_processor_asset() -> Output[NLPProcessor]:
    """
    Asset for initializing the NLP processor.

    Returns:
        Configured NLPProcessor instance
    """
    logger.info("Initializing NLP processor")

    config = NLPConfig(
        auto_detect_language=True,
        enable_caching=True
    )

    processor = NLPProcessor(config)

    return Output(
        value=processor,
        metadata={
            "supported_languages": processor.get_supported_languages(),
            "available_techniques": processor.get_available_techniques(),
            "auto_language_detection": config.auto_detect_language,
        }
    )


@asset(
    name="nlp_text_analysis",
    description="Perform comprehensive NLP analysis on processed text data",
    io_manager_key="pandas_io_manager",
    ins={"processed_data": AssetIn("processed_netmind_data"), "nlp_processor": AssetIn("nlp_processor")},
    metadata={
        "analysis_type": "comprehensive",
        "metrics": ["sentiment", "entities", "keywords", "topics"]
    }
)
def nlp_text_analysis_asset(processed_data: NetMindDataFrame, nlp_processor: NLPProcessor) -> Output[pd.DataFrame]:
    """
    Asset for performing comprehensive NLP analysis on processed text data.

    Args:
        processed_data: Processed DataFrame from processed_netmind_data asset
        nlp_processor: NLPProcessor instance

    Returns:
        DataFrame with NLP analysis results
    """
    logger.info("Performing comprehensive NLP analysis on processed data")

    analysis_results = []

    # Process each query in the dataset
    for idx, row in processed_data.iterrows():
        query_text = row['query']

        try:
            # Perform comprehensive analysis
            analysis = nlp_processor.analyze_text_comprehensive(query_text)

            # Convert to dictionary and add metadata
            analysis_dict = analysis.to_dict()
            analysis_dict['original_index'] = idx
            analysis_dict['knowledge_domain'] = row.get('knowledge_domain', '')

            analysis_results.append(analysis_dict)

        except Exception as e:
            logger.warning(f"Failed to analyze query at index {idx}: {e}")
            # Add error entry
            analysis_results.append({
                'original_index': idx,
                'query': query_text,
                'error': str(e),
                'language': 'unknown',
                'word_count': 0,
                'sentiment_score': 0.0,
                'entities': [],
                'keywords': [],
                'topics': [],
                'processing_time': 0.0
            })

    # Convert to DataFrame
    analysis_df = pd.DataFrame(analysis_results)

    logger.info(f"NLP analysis completed for {len(analysis_df)} queries")

    return Output(
        value=analysis_df,
        metadata={
            "total_queries_analyzed": len(analysis_df),
            "successful_analyses": len(analysis_df[analysis_df.get('error', '').isna()]),
            "failed_analyses": len(analysis_df[analysis_df.get('error', '').notna()]),
            "avg_processing_time": analysis_df['processing_time'].mean(),
            "unique_languages": analysis_df['language'].nunique() if 'language' in analysis_df.columns else 0,
            "avg_sentiment": analysis_df['sentiment_score'].mean() if 'sentiment_score' in analysis_df.columns else 0.0,
        }
    )


@asset(
    name="nlp_sentiment_analysis",
    description="Analyze sentiment of queries and generate sentiment reports",
    io_manager_key="pandas_io_manager",
    ins={"processed_data": AssetIn("processed_netmind_data"), "nlp_processor": AssetIn("nlp_processor")},
    metadata={
        "analysis_type": "sentiment",
        "sentiment_metrics": ["compound", "positive", "negative", "neutral"]
    }
)
def nlp_sentiment_analysis_asset(processed_data: NetMindDataFrame, nlp_processor: NLPProcessor) -> Output[pd.DataFrame]:
    """
    Asset for sentiment analysis of queries.

    Args:
        processed_data: Processed DataFrame from processed_netmind_data asset
        nlp_processor: NLPProcessor instance

    Returns:
        DataFrame with sentiment analysis results
    """
    logger.info("Performing sentiment analysis on queries")

    sentiment_results = []

    for idx, row in processed_data.iterrows():
        query_text = row['query']

        try:
            # Basic NLP processing
            result = nlp_processor.process_text(
                query_text,
                techniques=["tokenization", "sentiment_analysis", "named_entity_recognition"]
            )

            sentiment_results.append({
                'original_index': idx,
                'query': query_text,
                'language': result.language.value,
                'sentiment_compound': result.sentiment.get('compound', 0.0),
                'sentiment_positive': result.sentiment.get('pos', 0.0),
                'sentiment_negative': result.sentiment.get('neg', 0.0),
                'sentiment_neutral': result.sentiment.get('neu', 0.0),
                'entities': result.entities,
                'tokens': result.tokens,
                'processing_time': result.processing_time
            })

        except Exception as e:
            logger.warning(f"Failed to analyze sentiment for query at index {idx}: {e}")
            sentiment_results.append({
                'original_index': idx,
                'query': query_text,
                'error': str(e),
                'sentiment_compound': 0.0,
                'sentiment_positive': 0.0,
                'sentiment_negative': 0.0,
                'sentiment_neutral': 1.0,
                'entities': [],
                'tokens': [],
                'processing_time': 0.0
            })

    sentiment_df = pd.DataFrame(sentiment_results)

    # Add sentiment classification
    sentiment_df['sentiment_class'] = pd.cut(
        sentiment_df['sentiment_compound'],
        bins=[-1, -0.5, -0.1, 0.1, 0.5, 1],
        labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    )

    logger.info("Sentiment analysis completed")

    return Output(
        value=sentiment_df,
        metadata={
            "total_queries": len(sentiment_df),
            "sentiment_distribution": sentiment_df['sentiment_class'].value_counts().to_dict(),
            "avg_sentiment": sentiment_df['sentiment_compound'].mean(),
            "positive_queries": len(sentiment_df[sentiment_df['sentiment_compound'] > 0.1]),
            "negative_queries": len(sentiment_df[sentiment_df['sentiment_compound'] < -0.1]),
            "neutral_queries": len(sentiment_df[
                (sentiment_df['sentiment_compound'] >= -0.1) &
                (sentiment_df['sentiment_compound'] <= 0.1)
            ]),
        }
    )


@asset(
    name="nlp_entity_extraction",
    description="Extract and analyze named entities from queries",
    io_manager_key="pandas_io_manager",
    ins={"processed_data": AssetIn("processed_netmind_data"), "nlp_processor": AssetIn("nlp_processor")},
    metadata={
        "analysis_type": "entity_extraction",
        "entity_types": ["PERSON", "ORG", "GPE", "MONEY", "DATE", "PERCENT"]
    }
)
def nlp_entity_extraction_asset(processed_data: NetMindDataFrame, nlp_processor: NLPProcessor) -> Output[pd.DataFrame]:
    """
    Asset for named entity extraction and analysis.

    Args:
        processed_data: Processed DataFrame from processed_netmind_data asset
        nlp_processor: NLPProcessor instance

    Returns:
        DataFrame with entity extraction results
    """
    logger.info("Extracting named entities from queries")

    entity_results = []

    for idx, row in processed_data.iterrows():
        query_text = row['query']

        try:
            # Basic NLP processing with NER
            result = nlp_processor.process_text(
                query_text,
                techniques=["tokenization", "named_entity_recognition"]
            )

            entity_results.append({
                'original_index': idx,
                'query': query_text,
                'language': result.language.value,
                'entities': result.entities,
                'entity_count': len(result.entities),
                'processing_time': result.processing_time
            })

        except Exception as e:
            logger.warning(f"Failed to extract entities for query at index {idx}: {e}")
            entity_results.append({
                'original_index': idx,
                'query': query_text,
                'error': str(e),
                'entities': [],
                'entity_count': 0,
                'processing_time': 0.0
            })

    entity_df = pd.DataFrame(entity_results)

    # Flatten entities for analysis
    all_entities = []
    for _, row in entity_df.iterrows():
        for entity in row['entities']:
            all_entities.append({
                'query_index': row['original_index'],
                'entity_text': entity['text'],
                'entity_label': entity['label'],
                'confidence': entity.get('confidence', 1.0),
                'language': row['language']
            })

    entity_analysis_df = pd.DataFrame(all_entities)

    logger.info("Entity extraction completed")

    return Output(
        value=entity_df,
        metadata={
            "total_queries": len(entity_df),
            "queries_with_entities": len(entity_df[entity_df['entity_count'] > 0]),
            "total_entities": len(entity_analysis_df) if len(entity_analysis_df) > 0 else 0,
            "entity_types": entity_analysis_df['entity_label'].value_counts().to_dict() if len(entity_analysis_df) > 0 else {},
            "avg_entities_per_query": entity_df['entity_count'].mean(),
        }
    )


@asset(
    name="nlp_comprehensive_report",
    description="Generate comprehensive NLP analysis report",
    io_manager_key="pandas_io_manager",
    ins={
        "text_analysis": AssetIn("nlp_text_analysis"),
        "sentiment_analysis": AssetIn("nlp_sentiment_analysis"),
        "entity_analysis": AssetIn("nlp_entity_extraction"),
        "domain_stats": AssetIn("domain_statistics")
    },
    metadata={"report_type": "nlp_summary"}
)
def nlp_comprehensive_report_asset(text_analysis, sentiment_analysis, entity_analysis, domain_stats) -> Output[pd.DataFrame]:
    """
    Asset for generating comprehensive NLP analysis report.

    Args:
        text_analysis: Text analysis results
        sentiment_analysis: Sentiment analysis results
        entity_analysis: Entity extraction results
        domain_stats: Domain statistics

    Returns:
        DataFrame with comprehensive NLP report
    """
    logger.info("Generating comprehensive NLP report")

    # Summary statistics
    summary_stats = {
        'total_queries': len(text_analysis),
        'avg_word_count': text_analysis['word_count'].mean(),
        'avg_sentiment': sentiment_analysis['sentiment_compound'].mean(),
        'total_entities': entity_analysis['entity_count'].sum(),
        'unique_languages': text_analysis['language'].nunique(),
        'avg_processing_time': text_analysis['processing_time'].mean(),
        'sentiment_distribution': sentiment_analysis['sentiment_class'].value_counts().to_dict(),
        'top_keywords': text_analysis['keywords'].explode().value_counts().head(10).to_dict(),
        'top_topics': text_analysis['topics'].explode().value_counts().head(5).to_dict(),
    }

    summary_df = pd.DataFrame([summary_stats])

    logger.info("Comprehensive NLP report generated")

    return Output(
        value=summary_df,
        metadata={
            "report_generated_at": pd.Timestamp.now().isoformat(),
            "total_domains_analyzed": len(domain_stats),
            "nlp_metrics_covered": ["sentiment", "entities", "keywords", "topics"],
        }
    )
