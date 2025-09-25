#!/usr/bin/env python3
"""
Full Workflow Test Script for NetMind API

This script tests the complete workflow by:
1. Loading all queries from dataset.xlsx using DataLoader
2. Dividing queries into batches of 100 and processing batches in parallel
3. Within each batch, sequentially calling the API at localhost:8000 for each query
4. Storing results in a CSV file with ground truth vs predicted comparisons
5. Generating detailed debug logs and test summary focusing on accuracy and execution time
"""

import logging
import sys
import csv
import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader, DataLoaderConfig, QueryData

# Configure logging with file output for detailed debugging
logs_dir = Path("reports/test_result/logs")
logs_dir.mkdir(parents=True, exist_ok=True)

# Generate log filename with timestamp
log_filename = f"full_workflow_test_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = logs_dir / log_filename

# Configure logging with both console and file output using fixed-width columns
old_factory = logging.getLogRecordFactory()
def _record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    short = record.name.replace('test_full_workflow', 'full_workflow')
    record.logger_padded = short[:30].ljust(30)
    return record
logging.setLogRecordFactory(_record_factory)

fmt = "%(asctime)s | %(levelname)-8s | %(logger_padded)s | %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"

formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Clear existing handlers (if any) and re-add ours
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)
logger = logging.getLogger(__name__)


class APIClient:
    """Client for making requests to the NetMind API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        
    def label_query(self, query: str) -> Dict[str, Any]:
        """
        Send a query to the /label endpoint and return the response.
        
        Args:
            query: The query string to label
            
        Returns:
            Dictionary containing the API response or error information
        """
        url = f"{self.base_url}/label"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "query": query
        }
        
        try:
            logger.debug(f"Sending request to {url} with query: '{query[:50]}...'")
            start_time = time.time()
            
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"API response received in {execution_time:.4f}s: {result}")
                return {
                    "success": True,
                    "data": result,
                    "execution_time": execution_time,
                    "status_code": response.status_code
                }
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "execution_time": execution_time,
                    "status_code": response.status_code
                }
                
        except requests.exceptions.Timeout:
            execution_time = time.time() - start_time
            logger.error(f"API request timed out after {self.timeout}s")
            return {
                "success": False,
                "error": f"Request timeout after {self.timeout}s",
                "execution_time": execution_time,
                "status_code": None
            }
            
        except requests.exceptions.ConnectionError:
            execution_time = time.time() - start_time
            logger.error(f"Failed to connect to API at {url}")
            return {
                "success": False,
                "error": f"Connection error to {url}",
                "execution_time": execution_time,
                "status_code": None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Unexpected error during API call: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "execution_time": execution_time,
                "status_code": None
            }


def setup_data_loader() -> DataLoader:
    """Setup DataLoader with configuration for dataset.xlsx."""
    config = DataLoaderConfig(
        file_path="data/dataset.xlsx",
        query_column="query",
        knowledge_domain_column="knowledge_domain",
        worker_column="worker",
        worker_separator=",",
        cache_enabled=True,
        validate_data=True,
        max_workers=4
    )
    return DataLoader(config)


def load_all_queries(data_loader: DataLoader) -> List[QueryData]:
    """Load all queries from the dataset."""
    logger.info("Loading all queries from dataset...")
    all_queries = data_loader.get_processed_data()
    logger.info(f"Loaded {len(all_queries)} queries from dataset")
    return all_queries


def categorize_query(query_data: QueryData) -> str:
    """
    Categorize a query based on its expected workers using the same mapping rules.
    
    Returns: 'DOC_SEARCH', 'FUNCTION_CALLING', 'REMINDER', or 'OTHER'
    """
    # Apply the same mapping to expected workers
    mapped_workers = map_worker_labels(query_data.workers)
    
    # Determine primary category based on mapped workers
    if 'DOC_SEARCH' in mapped_workers:
        return 'DOC_SEARCH'
    elif 'FUNCTION_CALLING' in mapped_workers:
        return 'FUNCTION_CALLING'
    elif 'REMINDER' in mapped_workers or 'MEETING' in mapped_workers:
        return 'REMINDER'
    else:
        return 'OTHER'


def create_balanced_sample(queries: List[QueryData], 
                          doc_search_count: int = 400,
                          function_calling_count: int = 400, 
                          reminder_count: int = 200) -> List[QueryData]:
    """
    Create a balanced sample from the queries.
    
    Args:
        queries: All available queries
        doc_search_count: Number of DOC_SEARCH queries to sample
        function_calling_count: Number of FUNCTION_CALLING queries to sample
        reminder_count: Number of REMINDER/MEETING queries to sample
        
    Returns:
        Balanced list of queries
    """
    import random
    
    # Categorize all queries
    categorized = {
        'DOC_SEARCH': [],
        'FUNCTION_CALLING': [],
        'REMINDER': [],
        'OTHER': []
    }
    
    for query in queries:
        category = categorize_query(query)
        categorized[category].append(query)
    
    logger.info(f"Query distribution: DOC_SEARCH={len(categorized['DOC_SEARCH'])}, "
                f"FUNCTION_CALLING={len(categorized['FUNCTION_CALLING'])}, "
                f"REMINDER={len(categorized['REMINDER'])}, "
                f"OTHER={len(categorized['OTHER'])}")
    
    # Sample from each category
    balanced_queries = []
    
    # Sample DOC_SEARCH
    if len(categorized['DOC_SEARCH']) >= doc_search_count:
        sampled_doc = random.sample(categorized['DOC_SEARCH'], doc_search_count)
        logger.info(f"Sampled {len(sampled_doc)} DOC_SEARCH queries")
    else:
        sampled_doc = categorized['DOC_SEARCH']
        logger.warning(f"Only {len(sampled_doc)} DOC_SEARCH queries available (requested {doc_search_count})")
    balanced_queries.extend(sampled_doc)
    
    # Sample FUNCTION_CALLING
    if len(categorized['FUNCTION_CALLING']) >= function_calling_count:
        sampled_func = random.sample(categorized['FUNCTION_CALLING'], function_calling_count)
        logger.info(f"Sampled {len(sampled_func)} FUNCTION_CALLING queries")
    else:
        sampled_func = categorized['FUNCTION_CALLING']
        logger.warning(f"Only {len(sampled_func)} FUNCTION_CALLING queries available (requested {function_calling_count})")
    balanced_queries.extend(sampled_func)
    
    # Sample REMINDER
    if len(categorized['REMINDER']) >= reminder_count:
        sampled_reminder = random.sample(categorized['REMINDER'], reminder_count)
        logger.info(f"Sampled {len(sampled_reminder)} REMINDER queries")
    else:
        sampled_reminder = categorized['REMINDER']
        logger.warning(f"Only {len(sampled_reminder)} REMINDER queries available (requested {reminder_count})")
    balanced_queries.extend(sampled_reminder)
    
    # Shuffle the final list to mix categories
    random.shuffle(balanced_queries)
    
    logger.info(f"Created balanced sample with {len(balanced_queries)} total queries")
    return balanced_queries


def create_query_batches(queries: List[QueryData], batch_size: int = 100) -> List[List[QueryData]]:
    """Divide queries into batches of specified size."""
    batches = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        batches.append(batch)
    
    logger.info(f"Created {len(batches)} batches of up to {batch_size} queries each")
    return batches


def map_worker_labels(workers: List[str]) -> List[str]:
    """
    Map specific worker labels to consolidated categories.
    
    Mapping rules:
    - REGION_IDENTIFIER_AGENT, TIME_IDENTIFIER_AGENT, FUNCTION_CALLER_AGENT -> FUNCTION_CALLING
    - DOCS_SEARCHER_AGENT -> DOC_SEARCH
    - REMINDER_AGENT -> REMINDER (also matches MEETING)
    """
    mapping = {
        "REGION_IDENTIFIER_AGENT": "FUNCTION_CALLING",
        "TIME_IDENTIFIER_AGENT": "FUNCTION_CALLING", 
        "FUNCTION_CALLER_AGENT": "FUNCTION_CALLING",
        "DOCS_SEARCHER_AGENT": "DOC_SEARCH",
        "REMINDER_AGENT": "REMINDER"
    }
    
    mapped_workers = []
    for worker in workers:
        mapped_worker = mapping.get(worker, worker)
        if mapped_worker not in mapped_workers:  # Avoid duplicates
            mapped_workers.append(mapped_worker)
    
    return mapped_workers


def extract_workers_and_scores_from_api_response(api_response: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    """
    Extract the top 2 predicted workers and their confidence scores from the API response.

    Preference order (new _reduce output first):
    1) labels + votes: take first 2 from both
    2) labels/workers/predicted_workers/result: list[str] sorted by votes (take first 2), scores unknown
    3) prediction_vector/output_vector: numeric list (derive top 2 by votes)
    """
    if not api_response.get("success", False):
        return [], []

    data = api_response.get("data", {})

    # Prefer new dict form labels + votes
    labels = data.get("labels")
    votes = data.get("votes")
    if isinstance(labels, list) and labels:
        top_labels = labels[:2]
        if isinstance(votes, list) and votes:
            try:
                top_votes = [float(v) for v in votes[:2]]
            except Exception:
                top_votes = []
        else:
            top_votes = []
        mapped_workers = map_worker_labels(top_labels)
        logger.debug(f"Labels+votes extraction: {top_labels} {top_votes} -> {mapped_workers}")
        return mapped_workers, top_votes

    # Fallback: label lists only (unknown scores)
    possible_keys = ["workers", "predicted_workers", "result"]
    for key in possible_keys:
        if key in data:
            workers = data[key]
            if isinstance(workers, list):
                top_workers = workers[:2]
                mapped_workers = map_worker_labels(top_workers)
                logger.debug(f"Label-list extraction: {workers[:3]} -> {top_workers} -> {mapped_workers}")
                return mapped_workers, []
            if isinstance(workers, str):
                mapped_workers = map_worker_labels([workers])
                logger.debug(f"Single worker extraction: {workers} -> {mapped_workers}")
                return mapped_workers, []

    # Fallback: derive from numeric prediction vectors
    if "prediction_vector" in data or "output_vector" in data:
        vector_key = "prediction_vector" if "prediction_vector" in data else "output_vector"
        vector = data[vector_key]

        # Assume standard worker order from strategy
        worker_names = [
            "REGION_IDENTIFIER_AGENT", "TIME_IDENTIFIER_AGENT", "FUNCTION_CALLER_AGENT",
            "DOCS_SEARCHER_AGENT", "REMINDER_AGENT", "MEETING_AGENT", "GENERAL_AGENT", "UNKNOWN_AGENT"
        ]

        if isinstance(vector, list) and len(vector) > 0:
            pairs = [(worker_names[i], vector[i]) for i in range(min(len(worker_names), len(vector)))]
            try:
                pairs.sort(key=lambda x: float(x[1]), reverse=True)
            except Exception:
                logger.debug("Vector values not numeric; skipping vector-based extraction")
            else:
                top_pairs = [(name, float(val)) for name, val in pairs[:2] if float(val) > 0]
                top_workers = [name for name, _ in top_pairs]
                top_scores = [score for _, score in top_pairs]
                mapped_workers = map_worker_labels(top_workers)
                logger.debug(f"Vector-based extraction: {pairs[:3]} -> {top_pairs} -> {mapped_workers}")
                return mapped_workers, top_scores

    logger.warning(f"Could not extract workers from API response: {data}")
    return [], []


def calculate_accuracy_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive accuracy metrics from test results.
    
    Uses mapped worker labels and evaluates based on top 2 predicted workers.
    """
    total_queries = len(results)
    successful_requests = sum(1 for r in results if r["api_success"])
    
    if successful_requests == 0:
        return {
            "total_queries": total_queries,
            "successful_requests": 0,
            "success_rate": 0.0,
            "exact_match_accuracy": 0.0,
            "partial_match_accuracy": 0.0,
            "worker_precision": {},
            "worker_recall": {},
            "worker_f1": {},
            "average_precision": 0.0,
            "average_recall": 0.0,
            "average_f1": 0.0,
            "category_distribution": {}
        }
    
    # Calculate accuracy metrics
    exact_matches = 0
    partial_matches = 0
    worker_stats = {}
    category_stats = {}
    
    for result in results:
        if not result["api_success"]:
            continue
        
        # Apply mapping to expected workers
        expected_raw = result["expected_workers"]
        expected_mapped = map_worker_labels(expected_raw)
        expected = set(expected_mapped)
        
        # Predicted workers (top-2) are already mapped in extract function
        predicted_list = result.get("predicted_workers", [])
        predicted = set(predicted_list)
        
        # Determine query category for analysis
        query_category = 'DOC_SEARCH' if 'DOC_SEARCH' in expected else \
                        'FUNCTION_CALLING' if 'FUNCTION_CALLING' in expected else \
                        'REMINDER' if ('REMINDER' in expected or 'MEETING' in expected) else 'OTHER'
        
        if query_category not in category_stats:
            category_stats[query_category] = {"total": 0, "exact": 0, "partial": 0}
        category_stats[query_category]["total"] += 1
        
        # New matching criteria:
        # - Exact: first predicted label equals ground truth (primary expected)
        # - Partial: ground truth is in top-2 predicted labels
        ground_truth = expected_mapped[0] if expected_mapped else None
        is_exact = bool(predicted_list) and ground_truth is not None and predicted_list[0] == ground_truth
        is_partial = ground_truth is not None and ground_truth in predicted_list[:2]

        if is_exact:
            exact_matches += 1
            category_stats[query_category]["exact"] += 1

        if is_partial:
            partial_matches += 1
            category_stats[query_category]["partial"] += 1
        
        # Per-worker statistics
        all_workers = expected.union(predicted)
        for worker in all_workers:
            if worker not in worker_stats:
                worker_stats[worker] = {
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "true_negative": 0
                }
            
            if worker in expected and worker in predicted:
                worker_stats[worker]["true_positive"] += 1
            elif worker not in expected and worker in predicted:
                worker_stats[worker]["false_positive"] += 1
            elif worker in expected and worker not in predicted:
                worker_stats[worker]["false_negative"] += 1
    
    # Calculate precision, recall, F1 for each worker
    worker_precision = {}
    worker_recall = {}
    worker_f1 = {}
    
    for worker, stats in worker_stats.items():
        tp = stats["true_positive"]
        fp = stats["false_positive"]
        fn = stats["false_negative"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        worker_precision[worker] = precision
        worker_recall[worker] = recall
        worker_f1[worker] = f1
    
    # Calculate averages
    avg_precision = sum(worker_precision.values()) / len(worker_precision) if worker_precision else 0.0
    avg_recall = sum(worker_recall.values()) / len(worker_recall) if worker_recall else 0.0
    avg_f1 = sum(worker_f1.values()) / len(worker_f1) if worker_f1 else 0.0
    
    # Calculate category accuracies
    category_accuracies = {}
    for category, stats in category_stats.items():
        category_accuracies[category] = {
            "total": stats["total"],
            "exact_accuracy": stats["exact"] / stats["total"] if stats["total"] > 0 else 0.0,
            "partial_accuracy": stats["partial"] / stats["total"] if stats["total"] > 0 else 0.0
        }
    
    return {
        "total_queries": total_queries,
        "successful_requests": successful_requests,
        "success_rate": successful_requests / total_queries,
        "exact_match_accuracy": exact_matches / successful_requests,
        "partial_match_accuracy": partial_matches / successful_requests,
        "worker_precision": worker_precision,
        "worker_recall": worker_recall,
        "worker_f1": worker_f1,
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1": avg_f1,
        "category_distribution": category_accuracies
    }


def process_query_batch(batch_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a batch of queries sequentially within a single worker.
    This function is designed to be called by multiprocessing workers.
    
    Args:
        batch_data: Dictionary containing batch info and queries
        
    Returns:
        Dictionary with batch results
    """
    batch_id = batch_data["batch_id"]
    queries = batch_data["queries"]
    base_url = batch_data.get("base_url", "http://localhost:8000")
    timeout = batch_data.get("timeout", 30)
    
    # Create a separate logger for this worker process
    worker_logger = logging.getLogger(f"worker_{batch_id}")
    
    # Create API client for this worker
    api_client = APIClient(base_url=base_url, timeout=timeout)
    
    worker_logger.info(f"Worker {batch_id}: Processing batch with {len(queries)} queries")
    
    results = []
    total_time = 0
    total_api_time = 0
    
    for i, query_data in enumerate(queries):
        worker_logger.debug(f"Worker {batch_id}: Processing query {i+1}/{len(queries)}: '{query_data.query[:50]}...'")
        
        start_time = time.time()
        
        # Call the API
        api_response = api_client.label_query(query_data.query)
        api_time = api_response.get("execution_time", 0)
        total_api_time += api_time
        
        # Extract predicted workers and scores
        predicted_workers, predicted_scores = extract_workers_and_scores_from_api_response(api_response)
        
        end_time = time.time()
        total_execution_time = end_time - start_time
        total_time += total_execution_time
        
        result = {
            "query_index": batch_data["start_index"] + i,  # Global index
            "query": query_data.query,
            "knowledge_domain": query_data.knowledge_domain,
            "expected_workers": query_data.workers,
            "predicted_workers": predicted_workers,
            "predicted_workers_scores": predicted_scores,
            "api_success": api_response.get("success", False),
            "api_status_code": api_response.get("status_code"),
            "api_error": api_response.get("error"),
            "api_execution_time": api_time,
            "total_execution_time": total_execution_time,
            "batch_id": batch_id
        }
        
        results.append(result)
        
        # Log result
        if api_response.get("success", False):
            worker_logger.debug(f"Worker {batch_id}: Result: {predicted_workers} (expected: {query_data.workers})")
        else:
            worker_logger.error(f"Worker {batch_id}: API Error: {api_response.get('error', 'Unknown error')}")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    worker_logger.info(f"Worker {batch_id}: Completed batch in {total_time:.2f}s (API time: {total_api_time:.2f}s)")
    
    return {
        "batch_id": batch_id,
        "results": results,
        "batch_stats": {
            "total_queries": len(queries),
            "total_time": total_time,
            "total_api_time": total_api_time,
            "successful_requests": sum(1 for r in results if r["api_success"])
        }
    }


def run_full_workflow_test_parallel(queries: List[QueryData], batch_size: int = 100, max_workers: int = None) -> Dict[str, Any]:
    """
    Run the full workflow test using parallel batch processing.
    
    Args:
        queries: List of all queries to process
        batch_size: Number of queries per batch (default: 100)
        max_workers: Maximum number of parallel workers (default: calculated to match dataset_size/batch_size, capped at 25)
        
    Returns:
        Dictionary containing all test results and statistics
    """
    if max_workers is None:
        # Calculate workers so that num_workers * batch_size = dataset_size
        optimal_workers = max(1, len(queries) // batch_size)
        if len(queries) % batch_size != 0:
            optimal_workers += 1  # Add one more worker for the remainder
        
        # Apply system limitation (max 25 workers)
        max_workers = min(optimal_workers, 25)
        
        if optimal_workers > 25:
            logger.warning(f"Optimal workers ({optimal_workers}) exceeds system limit. Using 25 workers instead.")
        
        logger.info(f"Auto-calculated workers: {len(queries)} queries ÷ {batch_size} batch_size = {optimal_workers} optimal, using {max_workers} workers")
    
    logger.info(f"Starting parallel workflow test with {len(queries)} queries")
    logger.info(f"Using {max_workers} workers with batch size {batch_size}")
    
    # Create batches
    batches = create_query_batches(queries, batch_size)
    
    # Prepare batch data for workers
    batch_data_list = []
    start_index = 0
    for i, batch in enumerate(batches):
        batch_data = {
            "batch_id": i,
            "queries": batch,
            "start_index": start_index,
            "base_url": "http://localhost:8000",
            "timeout": 30
        }
        batch_data_list.append(batch_data)
        start_index += len(batch)
    
    # Process batches in parallel
    all_results = []
    total_time = time.time()
    
    logger.info(f"Submitting {len(batches)} batches to {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batch jobs
        future_to_batch = {
            executor.submit(process_query_batch, batch_data): batch_data["batch_id"]
            for batch_data in batch_data_list
        }
        
        # Collect results as they complete
        completed_batches = 0
        for future in as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                batch_result = future.result()
                all_results.extend(batch_result["results"])
                completed_batches += 1
                
                logger.info(f"Completed batch {batch_id} ({completed_batches}/{len(batches)})")
                logger.info(f"Batch {batch_id} stats: {batch_result['batch_stats']['successful_requests']}/{batch_result['batch_stats']['total_queries']} successful")
                
            except Exception as e:
                logger.error(f"Batch {batch_id} failed with error: {e}")
                # Add failed results for this batch
                batch_data = batch_data_list[batch_id]
                for j, query_data in enumerate(batch_data["queries"]):
                    failed_result = {
                        "query_index": batch_data["start_index"] + j,
                        "query": query_data.query,
                        "knowledge_domain": query_data.knowledge_domain,
                        "expected_workers": query_data.workers,
                        "predicted_workers": [],
                        "api_success": False,
                        "api_status_code": None,
                        "api_error": f"Batch processing failed: {str(e)}",
                        "api_execution_time": 0.0,
                        "total_execution_time": 0.0,
                        "batch_id": batch_id
                    }
                    all_results.append(failed_result)
    
    total_time = time.time() - total_time
    
    # Sort results by query index to maintain order
    all_results.sort(key=lambda x: x["query_index"])
    
    # Calculate aggregate statistics
    total_api_time = sum(r["api_execution_time"] for r in all_results)
    accuracy_metrics = calculate_accuracy_metrics(all_results)
    
    logger.info(f"Parallel processing completed in {total_time:.2f}s")
    logger.info(f"Total API time across all workers: {total_api_time:.2f}s")
    
    return {
        "results": all_results,
        "statistics": {
            "total_queries": len(queries),
            "total_batches": len(batches),
            "batch_size": batch_size,
            "max_workers": max_workers,
            "total_time": total_time,
            "total_api_time": total_api_time,
            "avg_time_per_query": total_time / len(queries) if queries else 0,
            "avg_api_time_per_query": total_api_time / len(queries) if queries else 0,
            "accuracy_metrics": accuracy_metrics
        }
    }


def run_full_workflow_test(queries: List[QueryData], api_client: APIClient) -> Dict[str, Any]:
    """Run the full workflow test by calling the API for each query."""
    logger.info(f"Starting full workflow test with {len(queries)} queries...")
    
    results = []
    total_time = 0
    total_api_time = 0
    
    for i, query_data in enumerate(queries):
        logger.info(f"Processing query {i+1}/{len(queries)}: '{query_data.query[:50]}...'")
        
        start_time = time.time()
        
        # Call the API
        api_response = api_client.label_query(query_data.query)
        api_time = api_response.get("execution_time", 0)
        total_api_time += api_time
        
        # Extract predicted workers and scores
        predicted_workers, predicted_scores = extract_workers_and_scores_from_api_response(api_response)
        
        end_time = time.time()
        total_execution_time = end_time - start_time
        total_time += total_execution_time
        
        result = {
            "query_index": i,
            "query": query_data.query,
            "knowledge_domain": query_data.knowledge_domain,
            "expected_workers": query_data.workers,
            "predicted_workers": predicted_workers,
            "predicted_workers_scores": predicted_scores,
            "api_success": api_response.get("success", False),
            "api_status_code": api_response.get("status_code"),
            "api_error": api_response.get("error"),
            "api_execution_time": api_time,
            "total_execution_time": total_execution_time
        }
        
        results.append(result)
        
        # Log result
        if api_response.get("success", False):
            logger.info(f"  Result: {predicted_workers} (expected: {query_data.workers})")
        else:
            logger.error(f"  API Error: {api_response.get('error', 'Unknown error')}")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    # Calculate metrics
    accuracy_metrics = calculate_accuracy_metrics(results)
    
    return {
        "results": results,
        "statistics": {
            "total_queries": len(queries),
            "total_time": total_time,
            "total_api_time": total_api_time,
            "avg_time_per_query": total_time / len(queries) if queries else 0,
            "avg_api_time_per_query": total_api_time / len(queries) if queries else 0,
            "accuracy_metrics": accuracy_metrics
        }
    }


def save_results_to_csv(test_results: Dict[str, Any], base_filename: str = "full_workflow_test") -> str:
    """Save test results to a CSV file with ground truth vs predicted comparisons."""
    reports_dir = Path("reports/test_result")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{base_filename}_{timestamp}.csv"
    csv_path = reports_dir / csv_filename
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'query_index',
            'query',
            'knowledge_domain',
            'expected_workers_raw',
            'expected_workers_mapped',
            'predicted_workers',
            'predicted_workers_scores',
            'exact_match',
            'partial_match',
            'query_category',
            'api_success',
            'api_status_code',
            'api_error',
            'api_execution_time',
            'total_execution_time',
            'batch_id'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in test_results["results"]:
            # Apply mapping to expected workers for comparison
            expected_raw = result["expected_workers"]
            expected_mapped = map_worker_labels(expected_raw)
            predicted_list = result.get("predicted_workers", [])

            # New definitions for matches
            gt = expected_mapped[0] if expected_mapped else None
            exact_match = bool(predicted_list) and gt is not None and predicted_list[0] == gt
            partial_match = gt is not None and gt in predicted_list[:2]
            
            # Determine query category
            query_category = 'DOC_SEARCH' if 'DOC_SEARCH' in expected_mapped else \
                            'FUNCTION_CALLING' if 'FUNCTION_CALLING' in expected_mapped else \
                            'REMINDER' if ('REMINDER' in expected_mapped or 'MEETING' in expected_mapped) else 'OTHER'
            
            row = {
                'query_index': result["query_index"] + 1,
                'query': result["query"],
                'knowledge_domain': result["knowledge_domain"],
                'expected_workers_raw': "; ".join(expected_raw),
                'expected_workers_mapped': "; ".join(expected_mapped),
                'predicted_workers': "; ".join(result["predicted_workers"]),
                'predicted_workers_scores': "; ".join(
                    [f"{float(s):.6f}" for s in (result.get("predicted_workers_scores") or [])]
                ),
                'exact_match': exact_match,
                'partial_match': partial_match,
                'query_category': query_category,
                'api_success': result["api_success"],
                'api_status_code': result["api_status_code"],
                'api_error': result["api_error"] if result["api_error"] else "",
                'api_execution_time': f"{result['api_execution_time']:.4f}",
                'total_execution_time': f"{result['total_execution_time']:.4f}",
                'batch_id': result.get("batch_id", "N/A")
            }
            writer.writerow(row)
    
    logger.info(f"Saved CSV results to {csv_path}")
    return str(csv_path)


def save_test_summary(test_results: Dict[str, Any], base_filename: str = "full_workflow_test_summary"):
    """Save comprehensive test summary focusing on accuracy and execution time."""
    reports_dir = Path("reports/test_result")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON summary
    json_filename = f"{base_filename}_{timestamp}.json"
    json_path = reports_dir / json_filename
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    # Save text summary
    text_filename = f"{base_filename}_{timestamp}.txt"
    text_path = reports_dir / text_filename
    
    stats = test_results["statistics"]
    accuracy = stats["accuracy_metrics"]
    
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FULL WORKFLOW API TEST RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Test executed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write(f"  Total queries tested: {stats['total_queries']}\n")
        f.write(f"  Successful API requests: {accuracy['successful_requests']}\n")
        f.write(f"  API success rate: {accuracy['success_rate']:.1%}\n")
        f.write(f"  Total execution time: {stats['total_time']:.2f}s\n")
        f.write(f"  Total API time: {stats['total_api_time']:.2f}s\n")
        f.write(f"  Average time per query: {stats['avg_time_per_query']:.4f}s\n")
        f.write(f"  Average API time per query: {stats['avg_api_time_per_query']:.4f}s\n")
        
        # Add batch processing info if available
        if 'total_batches' in stats:
            f.write(f"  Total batches: {stats['total_batches']}\n")
            f.write(f"  Batch size: {stats['batch_size']}\n")
            f.write(f"  Parallel workers: {stats['max_workers']}\n")
        f.write("\n")
        
        f.write("ACCURACY METRICS:\n")
        f.write(f"  Exact match accuracy: {accuracy['exact_match_accuracy']:.2%}\n")
        f.write(f"  Partial match accuracy: {accuracy['partial_match_accuracy']:.2%}\n")
        f.write(f"  Average precision: {accuracy['average_precision']:.2%}\n")
        f.write(f"  Average recall: {accuracy['average_recall']:.2%}\n")
        f.write(f"  Average F1-score: {accuracy['average_f1']:.2%}\n\n")
        
        f.write("CATEGORY PERFORMANCE:\n")
        if accuracy["category_distribution"]:
            for category, stats in accuracy["category_distribution"].items():
                f.write(f"  {category}:\n")
                f.write(f"    Total queries: {stats['total']}\n")
                f.write(f"    Exact accuracy: {stats['exact_accuracy']:.2%}\n")
                f.write(f"    Partial accuracy: {stats['partial_accuracy']:.2%}\n")
        else:
            f.write("  No category performance data available\n")
        f.write("\n")
        
        f.write("PER-WORKER METRICS:\n")
        if accuracy["worker_precision"]:
            f.write("Worker Precision:\n")
            for worker, precision in sorted(accuracy["worker_precision"].items()):
                f.write(f"    {worker}: {precision:.2%}\n")
            
            f.write("\nWorker Recall:\n")
            for worker, recall in sorted(accuracy["worker_recall"].items()):
                f.write(f"    {worker}: {recall:.2%}\n")
            
            f.write("\nWorker F1-Score:\n")
            for worker, f1 in sorted(accuracy["worker_f1"].items()):
                f.write(f"    {worker}: {f1:.2%}\n")
        else:
            f.write("  No worker metrics available\n")
        
        f.write("\nFAILED REQUESTS:\n")
        failed_requests = [r for r in test_results["results"] if not r["api_success"]]
        if failed_requests:
            for result in failed_requests[:10]:  # Show first 10 failures
                f.write(f"  Query {result['query_index'] + 1}: {result['api_error']}\n")
            if len(failed_requests) > 10:
                f.write(f"  ... and {len(failed_requests) - 10} more failures\n")
        else:
            f.write("  No failed requests\n")
    
    logger.info(f"Saved JSON summary to {json_path}")
    logger.info(f"Saved text summary to {text_path}")
    
    return {
        "json_path": str(json_path),
        "text_path": str(text_path)
    }


def print_test_summary(test_results: Dict[str, Any]):
    """Print formatted test summary to console."""
    print("\n" + "="*80)
    print("FULL WORKFLOW API TEST SUMMARY")
    print("="*80)
    
    stats = test_results["statistics"]
    accuracy = stats["accuracy_metrics"]
    
    print("\nOVERALL STATISTICS:")
    print(f"  Total queries tested: {stats['total_queries']}")
    print(f"  Successful API requests: {accuracy['successful_requests']}")
    print(f"  API success rate: {accuracy['success_rate']:.1%}")
    print(f"  Total execution time: {stats['total_time']:.2f}s")
    print(f"  Total API time: {stats['total_api_time']:.2f}s")
    print(f"  Average time per query: {stats['avg_time_per_query']:.4f}s")
    print(f"  Average API time per query: {stats['avg_api_time_per_query']:.4f}s")
    
    # Add batch processing info if available
    if 'total_batches' in stats:
        print(f"  Total batches: {stats['total_batches']}")
        print(f"  Batch size: {stats['batch_size']}")
        print(f"  Parallel workers: {stats['max_workers']}")
    
    print("\nACCURACY METRICS:")
    print(f"  Exact match accuracy: {accuracy['exact_match_accuracy']:.2%}")
    print(f"  Partial match accuracy: {accuracy['partial_match_accuracy']:.2%}")
    print(f"  Average precision: {accuracy['average_precision']:.2%}")
    print(f"  Average recall: {accuracy['average_recall']:.2%}")
    print(f"  Average F1-score: {accuracy['average_f1']:.2%}")
    
    print("\nCATEGORY PERFORMANCE:")
    if accuracy["category_distribution"]:
        for category, stats in accuracy["category_distribution"].items():
            print(f"  {category}: {stats['total']} queries, Exact: {stats['exact_accuracy']:.1%}, Partial: {stats['partial_accuracy']:.1%}")
    else:
        print("  No category performance data available")
    
    print("\nTOP WORKER PERFORMANCE (by F1-score):")
    if accuracy["worker_f1"]:
        sorted_workers = sorted(accuracy["worker_f1"].items(), key=lambda x: x[1], reverse=True)
        for worker, f1 in sorted_workers[:5]:
            precision = accuracy["worker_precision"].get(worker, 0)
            recall = accuracy["worker_recall"].get(worker, 0)
            print(f"  {worker}: P={precision:.2%}, R={recall:.2%}, F1={f1:.2%}")
    else:
        print("  No worker performance data available")


def main():
    """Main test execution function."""
    try:
        logger.info("Starting Full Workflow API tests with parallel batch processing...")
        
        # Setup components
        logger.info("Setting up DataLoader...")
        data_loader = setup_data_loader()
        
        # Load all test queries
        all_queries = load_all_queries(data_loader)
        
        # Create balanced sample: 400 DOC_SEARCH, 400 FUNCTION_CALLING, 200 REMINDER
        test_queries = create_balanced_sample(all_queries, 
                                            doc_search_count=5000,
                                            function_calling_count=400, 
                                            reminder_count=200)
        logger.info(f"Using balanced sample with {len(test_queries)} queries")
        
        # Run the full workflow test with parallel batch processing
        batch_size = 100
        # Calculate workers so that num_workers * batch_size = dataset_size
        optimal_workers = max(1, len(test_queries) // batch_size)
        if len(test_queries) % batch_size != 0:
            optimal_workers += 1  # Add one more worker for the remainder
        
        # Apply system limitation (max 25 workers)
        max_workers = min(optimal_workers, 25)
        
        if optimal_workers > 25:
            logger.warning(f"Optimal workers ({optimal_workers}) exceeds system limit. Using 25 workers instead.")
        
        logger.info(f"Calculated workers: {len(test_queries)} queries ÷ {batch_size} batch_size = {optimal_workers} optimal, using {max_workers} workers")
        
        logger.info(f"Using batch size: {batch_size}, max workers: {max_workers}")
        test_results = run_full_workflow_test_parallel(
            test_queries, 
            batch_size=batch_size, 
            max_workers=max_workers
        )
        
        # Save results
        csv_path = save_results_to_csv(test_results)
        summary_files = save_test_summary(test_results)
        
        # Print summary
        print_test_summary(test_results)
        
        print(f"\nResults saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  JSON Summary: {summary_files['json_path']}")
        print(f"  Text Summary: {summary_files['text_path']}")
        print(f"  Debug Logs: {log_filepath}")
        
        logger.info("Full workflow test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    mp.freeze_support()
    main()
