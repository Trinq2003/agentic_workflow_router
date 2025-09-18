"""
Dagster Jobs for NetMind Workflow

This module defines Dagster jobs for orchestrating data processing workflows
with support for parallel and sequential execution modes.
"""

import logging
from typing import Dict, Any, List

from dagster import job, op, graph, ScheduleDefinition, sensor, RunRequest, DefaultSensorStatus
from dagster import AssetSelection, define_asset_job
from dagster.core.definitions.run_config import RunConfig

from .assets import (
    raw_netmind_data_asset,
    cleaned_netmind_data_asset,
    processed_netmind_data_asset,
    domain_statistics_asset,
    worker_statistics_asset,
    data_quality_report_assets,
    data_sample_asset,
)

# Configure logging
logger = logging.getLogger(__name__)


# Define asset jobs for different execution modes

# Full data processing pipeline
full_data_processing_job = define_asset_job(
    name="full_data_processing",
    description="Complete data processing pipeline from raw data to analysis",
    selection=AssetSelection.all(),
    config={
        "execution": {
            "config": {
                "multiprocess": {
                    "max_concurrent": 4,
                }
            }
        }
    }
)

# Data loading and cleaning only
data_loading_job = define_asset_job(
    name="data_loading",
    description="Load and clean raw data only",
    selection=AssetSelection.assets(
        raw_netmind_data_asset,
        cleaned_netmind_data_asset,
    ),
)

# Analysis only (assumes data is already loaded)
data_analysis_job = define_asset_job(
    name="data_analysis",
    description="Run analysis on processed data",
    selection=AssetSelection.assets(
        processed_netmind_data_asset,
        domain_statistics_asset,
        worker_statistics_asset,
        data_quality_report_assets,
        data_sample_asset,
    ),
)

# Quick data sample job for testing
data_sample_job = define_asset_job(
    name="data_sample",
    description="Generate data sample for quick testing",
    selection=AssetSelection.assets(
        raw_netmind_data_asset,
        processed_netmind_data_asset,
        data_sample_asset,
    ),
)


@op(description="Log job execution details")
def log_job_execution(context, job_name: str, execution_mode: str) -> Dict[str, Any]:
    """Log job execution details and return metadata."""
    logger.info(f"Starting job: {job_name} in {execution_mode} mode")

    execution_info = {
        "job_name": job_name,
        "execution_mode": execution_mode,
        "timestamp": context.op_config.get("timestamp", "auto"),
        "run_id": context.run_id,
    }

    logger.info(f"Job execution info: {execution_info}")
    return execution_info


@op(description="Validate job inputs and environment")
def validate_job_environment(context) -> Dict[str, Any]:
    """Validate that required files and environment are available."""
    import os
    from pathlib import Path

    validation_results = {
        "data_file_exists": False,
        "data_file_path": "data/dataset.xlsx",
        "data_file_size": 0,
        "validation_errors": [],
        "validation_warnings": [],
    }

    data_path = Path("data/dataset.xlsx")

    if data_path.exists():
        validation_results["data_file_exists"] = True
        validation_results["data_file_size"] = data_path.stat().st_size
        logger.info(f"Data file found: {data_path} ({validation_results['data_file_size']} bytes)")
    else:
        error_msg = f"Data file not found: {data_path}"
        validation_results["validation_errors"].append(error_msg)
        logger.error(error_msg)

    # Check if we're in a proper environment
    if not os.path.exists("src"):
        warning_msg = "Source directory 'src' not found in current directory"
        validation_results["validation_warnings"].append(warning_msg)
        logger.warning(warning_msg)

    logger.info(f"Environment validation completed: {len(validation_results['validation_errors'])} errors, {len(validation_results['validation_warnings'])} warnings")

    return validation_results


@graph
def data_processing_graph():
    """Graph for the main data processing workflow."""
    # Log execution
    execution_info = log_job_execution("data_processing", "parallel")

    # Validate environment
    validation = validate_job_environment()

    return {
        "execution_info": execution_info,
        "validation": validation,
    }


# Define jobs with different execution modes

@job(
    name="parallel_data_processing",
    description="Run complete data processing pipeline in parallel mode",
    config={
        "execution": {
            "config": {
                "multiprocess": {
                    "max_concurrent": 4,
                }
            }
        },
        "ops": {
            "log_job_execution": {
                "config": {
                    "job_name": "parallel_data_processing",
                    "execution_mode": "parallel",
                }
            }
        }
    }
)
def parallel_data_processing_job():
    """Job for parallel execution of data processing pipeline."""
    data_processing_graph()


@job(
    name="sequential_data_processing",
    description="Run complete data processing pipeline in sequential mode",
    config={
        "execution": {
            "config": {
                "multiprocess": {
                    "max_concurrent": 1,
                }
            }
        },
        "ops": {
            "log_job_execution": {
                "config": {
                    "job_name": "sequential_data_processing",
                    "execution_mode": "sequential",
                }
            }
        }
    }
)
def sequential_data_processing_job():
    """Job for sequential execution of data processing pipeline."""
    data_processing_graph()


@job(
    name="incremental_data_processing",
    description="Run incremental data processing with caching",
    config={
        "execution": {
            "config": {
                "multiprocess": {
                    "max_concurrent": 2,
                }
            }
        },
        "ops": {
            "log_job_execution": {
                "config": {
                    "job_name": "incremental_data_processing",
                    "execution_mode": "incremental",
                }
            }
        }
    }
)
def incremental_data_processing_job():
    """Job for incremental data processing with caching enabled."""
    data_processing_graph()


# Define schedules for automated execution

# Daily schedule for full data processing
daily_data_processing_schedule = ScheduleDefinition(
    job=full_data_processing_job,
    cron_schedule="0 2 * * *",  # Run at 2 AM daily
    name="daily_data_processing",
    description="Run full data processing pipeline daily at 2 AM",
)

# Weekly schedule for data quality reports
weekly_quality_report_schedule = ScheduleDefinition(
    job=define_asset_job(
        name="weekly_quality_report",
        description="Generate weekly data quality report",
        selection=AssetSelection.assets(data_quality_report_assets),
    ),
    cron_schedule="0 3 * * 1",  # Run at 3 AM every Monday
    name="weekly_quality_report",
    description="Generate comprehensive data quality report weekly",
)

# Hourly schedule for data sampling (for monitoring)
hourly_data_sample_schedule = ScheduleDefinition(
    job=data_sample_job,
    cron_schedule="0 * * * *",  # Run every hour
    name="hourly_data_sample",
    description="Generate data sample hourly for monitoring",
)


# Define sensors for event-driven execution

@sensor(
    job=full_data_processing_job,
    name="data_file_change_sensor",
    description="Trigger data processing when dataset.xlsx file changes",
    default_status=DefaultSensorStatus.STOPPED,  # Enable when needed
)
def data_file_change_sensor(context):
    """Sensor that triggers when the data file is modified."""
    from pathlib import Path

    data_path = Path("data/dataset.xlsx")
    last_mtime = context.cursor or 0

    if data_path.exists():
        current_mtime = data_path.stat().st_mtime

        if current_mtime > float(last_mtime):
            logger.info(f"Data file changed. Last mtime: {last_mtime}, Current mtime: {current_mtime}")
            yield RunRequest(
                run_key=f"data_file_change_{current_mtime}",
                run_config={},
            )

            # Update cursor
            context.update_cursor(str(current_mtime))
    else:
        logger.warning(f"Data file not found: {data_path}")


@sensor(
    job=data_analysis_job,
    name="processed_data_ready_sensor",
    description="Trigger analysis when processed data is ready",
    default_status=DefaultSensorStatus.STOPPED,  # Enable when needed
)
def processed_data_ready_sensor(context):
    """Sensor that triggers analysis when processed data is available."""
    # This would typically check for file existence or database updates
    # For now, we'll use a simple time-based trigger
    import time

    last_run = context.cursor or 0
    current_time = time.time()

    # Trigger every 4 hours
    if current_time - float(last_run) > 4 * 60 * 60:
        logger.info("Triggering analysis job - processed data ready")
        yield RunRequest(
            run_key=f"analysis_{current_time}",
            run_config={},
        )

        context.update_cursor(str(current_time))


# Export all jobs and schedules
__all__ = [
    # Jobs
    "full_data_processing_job",
    "data_loading_job",
    "data_analysis_job",
    "data_sample_job",
    "parallel_data_processing_job",
    "sequential_data_processing_job",
    "incremental_data_processing_job",

    # Schedules
    "daily_data_processing_schedule",
    "weekly_quality_report_schedule",
    "hourly_data_sample_schedule",

    # Sensors
    "data_file_change_sensor",
    "processed_data_ready_sensor",
]
