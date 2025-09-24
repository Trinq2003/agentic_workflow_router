#!/usr/bin/env python3
"""
Test script for WorkerLabelingNLPStrategy

This script tests the WorkerLabelingNLPStrategy class by:
1. Loading data from dataset.xlsx using DataLoader
2. Using the first 20 queries for testing
3. Running the strategy on each query
4. Displaying results and analysis
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import time
import json
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_loader import DataLoader, DataLoaderConfig, QueryData
from strategy import NLPStrategy
import numpy as np

# Configure logging with file output for detailed debugging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = Path("reports/test_result/logs")
logs_dir.mkdir(exist_ok=True)

# Generate log filename with timestamp
log_filename = f"worker_labeling_strategy_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = logs_dir / log_filename

# Configure logging with both console and file output using fixed-width columns
old_factory = logging.getLogRecordFactory()
def _record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    short = record.name.replace('logic.nlp.', '').replace('strategy.', '')
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


def get_first_n_queries(data_loader: DataLoader, n: int = 20) -> List[QueryData]:
    """Get the first N queries from the dataset."""
    logger.info(f"Loading first {n} queries from dataset...")
    all_queries = data_loader.get_processed_data()
    return all_queries[:n]


def analyze_strategy_output(output, workers: List[str]) -> Dict[str, Any]:
    """Analyze the strategy output tensor."""
    # Convert vector to list for easier processing
    output_list = np.squeeze(output).tolist()

    selected_workers = []
    selected_indices = []

    for i, value in enumerate(output_list):
        if value == 1.0:
            selected_workers.append(workers[i])
            selected_indices.append(i)

    return {
        "selected_workers": selected_workers,
        "selected_indices": selected_indices,
        "output_vector": output_list,
        "num_selected": len(selected_workers)
    }


def run_strategy_tests(queries: List[QueryData], strategy: NLPStrategy) -> Dict[str, Any]:
    """Run the strategy on all test queries and collect results."""
    logger.info(f"Running strategy tests on {len(queries)} queries...")

    results = []
    total_time = 0

    for i, query_data in enumerate(queries):
        logger.info(f"Testing query {i+1}/{len(queries)}: '{query_data.query[:50]}...'")

        # Time the strategy execution
        start_time = time.time()
        try:
            output = strategy.forward(query_data.query)
            execution_time = time.time() - start_time

            # Analyze the output
            analysis = analyze_strategy_output(output, strategy.workers)

            result = {
                "query_index": i,
                "query": query_data.query,
                "expected_workers": query_data.workers,
                "predicted_workers": analysis["selected_workers"],
                "output_vector": analysis["output_vector"],
                "execution_time": execution_time,
                "success": True
            }

            results.append(result)
            total_time += execution_time

            # Log result
            logger.info(f"  Result: {analysis['selected_workers']} (expected: {query_data.workers})")

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"  Failed to process query {i}: {e}")

            result = {
                "query_index": i,
                "query": query_data.query,
                "expected_workers": query_data.workers,
                "predicted_workers": [],
                "output_vector": [],
                "execution_time": execution_time,
                "success": False,
                "error": str(e)
            }
            results.append(result)
            total_time += execution_time

    # Calculate statistics
    successful_tests = [r for r in results if r["success"]]
    accuracy_stats = calculate_accuracy_stats(results)

    return {
        "results": results,
        "statistics": {
            "total_queries": len(queries),
            "successful_tests": len(successful_tests),
            "total_time": total_time,
            "avg_time_per_query": total_time / len(queries) if queries else 0,
            "accuracy": accuracy_stats
        }
    }


def calculate_accuracy_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate accuracy statistics for the test results."""
    total_predictions = 0
    correct_predictions = 0
    worker_accuracy = {}

    for result in results:
        if not result["success"]:
            continue

        expected = set(result["expected_workers"])
        predicted = set(result["predicted_workers"])

        # Exact match accuracy
        if expected == predicted:
            correct_predictions += 1

        total_predictions += 1

        # Per-worker accuracy
        for worker in expected:
            if worker not in worker_accuracy:
                worker_accuracy[worker] = {"total": 0, "correct": 0}
            worker_accuracy[worker]["total"] += 1
            if worker in predicted:
                worker_accuracy[worker]["correct"] += 1

        for worker in predicted:
            if worker not in expected:
                if worker not in worker_accuracy:
                    worker_accuracy[worker] = {"total": 0, "correct": 0}
                # Don't increment total for false positives in this simple metric

    exact_match_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    worker_accuracies = {}
    for worker, stats in worker_accuracy.items():
        if stats["total"] > 0:
            worker_accuracies[worker] = stats["correct"] / stats["total"]

    return {
        "exact_match_accuracy": exact_match_accuracy,
        "worker_accuracies": worker_accuracies,
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions
    }


def save_test_results(test_results: Dict[str, Any], base_filename: str = "worker_labeling_strategy_test"):
    """Save test results to reports folder in both JSON and text formats."""
    # Create reports/test_result directory if it doesn't exist
    reports_dir = Path("reports/test_result")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON results
    json_filename = f"{base_filename}_{timestamp}.json"
    json_path = reports_dir / json_filename

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved JSON results to {json_path}")

    # Save text summary
    text_filename = f"{base_filename}_{timestamp}.txt"
    text_path = reports_dir / text_filename

    with open(text_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("WORKER LABELING NLP STRATEGY TEST RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Test executed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        stats = test_results["statistics"]

        f.write("OVERALL STATISTICS:\n")
        f.write(f"  Total queries tested: {stats['total_queries']}\n")
        f.write(f"  Successful tests: {stats['successful_tests']}\n")
        f.write(f"  Total execution time: {stats['total_time']:.2f}s\n")
        f.write(f"  Average time per query: {stats['avg_time_per_query']:.4f}s\n")
        f.write(f"  Success rate: {stats['successful_tests']/stats['total_queries']:.1%}\n\n")

        accuracy = stats["accuracy"]
        f.write("ACCURACY METRICS:\n")
        f.write(f"  Exact match accuracy: {accuracy['exact_match_accuracy']:.2%}\n")
        f.write("\nPer-Worker Accuracy:\n")

        if accuracy["worker_accuracies"]:
            for worker, acc in sorted(accuracy["worker_accuracies"].items()):
                f.write(f"    {worker}: {acc:.2%}\n")
        else:
            f.write("  No worker accuracy data available\n")

        f.write("\nDETAILED RESULTS:\n")
        f.write("-" * 80 + "\n\n")

        for result in test_results["results"]:
            status = "✓" if result["success"] else "✗"
            f.write(f"{status} Query {result['query_index'] + 1}:\n")
            f.write(f"    Query: {result['query'][:60]}{'...' if len(result['query']) > 60 else ''}\n")
            f.write(f"    Expected: {result['expected_workers']}\n")
            f.write(f"    Predicted: {result.get('predicted_workers', 'ERROR')}\n")
            f.write(f"    Execution time: {result['execution_time']:.4f}s\n")
            if not result["success"]:
                f.write(f"    Error: {result.get('error', 'Unknown error')}\n")
            f.write("\n")

    logger.info(f"Saved text summary to {text_path}")

    return {
        "json_path": str(json_path),
        "text_path": str(text_path)
    }


def print_test_results(test_results: Dict[str, Any]):
    """Print formatted test results."""
    print("\n" + "="*80)
    print("WORKER LABELING NLP STRATEGY TEST RESULTS")
    print("="*80)

    stats = test_results["statistics"]

    print("\nOVERALL STATISTICS:")
    print(f"  Total queries tested: {stats['total_queries']}")
    print(f"  Successful tests: {stats['successful_tests']}")
    print(f"  Total execution time: {stats['total_time']:.2f}s")
    print(f"  Average time per query: {stats['avg_time_per_query']:.4f}s")
    print(f"  Success rate: {stats['successful_tests']/stats['total_queries']:.1%}")

    accuracy = stats["accuracy"]
    print("\nACCURACY METRICS:")
    print(f"  Exact match accuracy: {accuracy['exact_match_accuracy']:.2%}")
    print("\nPer-Worker Accuracy:")

    if accuracy["worker_accuracies"]:
        for worker, acc in sorted(accuracy["worker_accuracies"].items()):
            print(f"    {worker}: {acc:.2%}")
    else:
        print("  No worker accuracy data available")

    print("\nDETAILED RESULTS:")
    print("-" * 80)

    for result in test_results["results"]:
        status = "✓" if result["success"] else "✗"
        print(f"{status} Query {result['query_index'] + 1}:")
        print(f"    Query: {result['query'][:60]}{'...' if len(result['query']) > 60 else ''}")
        print(f"    Expected: {result['expected_workers']}")
        print(f"    Predicted: {result.get('predicted_workers', 'ERROR')}")
        print(f"    Execution time: {result['execution_time']:.4f}s")
        if not result["success"]:
            print(f"    Error: {result.get('error', 'Unknown error')}")
        print()


def main():
    """Main test execution function."""
    try:
        logger.info("Starting WorkerLabelingNLPStrategy tests...")

        # Setup components
        logger.info("Setting up DataLoader...")
        data_loader = setup_data_loader()

        logger.info("Setting up NLPStrategy...")
        strategy = NLPStrategy()
        logger.info(f"Strategy loaded with workers: {strategy.workers}")

        # Get test data
        test_queries = get_first_n_queries(data_loader, n=2000)
        logger.info(f"Loaded {len(test_queries)} test queries")

        # Run tests
        test_results = run_strategy_tests(test_queries, strategy)

        # Save results to reports folder
        saved_files = save_test_results(test_results)
        logger.info(f"Results saved to: {saved_files}")

        # Print results
        print_test_results(test_results)

        print(f"\nResults saved to:")
        print(f"  JSON: {saved_files['json_path']}")
        print(f"  Text: {saved_files['text_path']}")
        print(f"  Debug Logs: {log_filepath}")

        logger.info(f"Tests completed successfully! Debug logs saved to {log_filepath}")

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
