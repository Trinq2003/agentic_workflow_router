"""
Dagster Repository Definition for NetMind Workflow

This module defines the Dagster repository containing all assets, jobs,
schedules, and sensors for the NetMind data processing workflow.
"""

from dagster import repository, define_asset_job, AssetSelection
from dagster.core.definitions.run_config import RunConfig

from .assets import *
from .jobs import *


@repository(
    name="netmind_workflow_repo",
    description="Dagster repository for NetMind data processing and analysis workflow",
)
def netmind_workflow_repository():
    """
    Main Dagster repository for NetMind workflow.

    Returns:
        List of all jobs, schedules, sensors, and assets in the repository
    """

    # Define additional utility jobs
    all_assets_job = define_asset_job(
        name="all_assets",
        description="Run all assets in the repository",
        selection=AssetSelection.all(),
    )

    # Core data processing job
    core_processing_job = define_asset_job(
        name="core_processing",
        description="Core data processing: load, clean, and process data",
        selection=AssetSelection.assets(
            raw_netmind_data_asset,
            cleaned_netmind_data_asset,
            processed_netmind_data_asset,
        ),
    )

    # Analysis job
    analysis_job = define_asset_job(
        name="analysis",
        description="Run all analysis assets",
        selection=AssetSelection.assets(
            domain_statistics_asset,
            worker_statistics_asset,
            data_quality_report_assets,
            data_sample_asset,
        ),
    )

    # Return all definitions
    return [
        # Asset jobs
        all_assets_job,
        core_processing_job,
        analysis_job,

        # Pre-defined jobs from jobs.py
        full_data_processing_job,
        data_loading_job,
        data_analysis_job,
        data_sample_job,
        parallel_data_processing_job,
        sequential_data_processing_job,
        incremental_data_processing_job,

        # Schedules
        daily_data_processing_schedule,
        weekly_quality_report_schedule,
        hourly_data_sample_schedule,

        # Sensors
        data_file_change_sensor,
        processed_data_ready_sensor,

        # All assets (automatically included)
        raw_netmind_data_asset,
        cleaned_netmind_data_asset,
        processed_netmind_data_asset,
        domain_statistics_asset,
        worker_statistics_asset,
        data_quality_report_assets,
        data_sample_asset,
    ]


# Configuration templates for different execution modes

PARALLEL_CONFIG = {
    "execution": {
        "config": {
            "multiprocess": {
                "max_concurrent": 4,
            }
        }
    }
}

SEQUENTIAL_CONFIG = {
    "execution": {
        "config": {
            "multiprocess": {
                "max_concurrent": 1,
            }
        }
    }
}

FAST_CONFIG = {
    "execution": {
        "config": {
            "multiprocess": {
                "max_concurrent": 2,
            }
        }
    },
    "ops": {
        # Limit sample size for faster execution
        "data_sample": {
            "config": {
                "sample_size": 50,
            }
        }
    }
}

# Resource configurations for different environments
DEV_RESOURCES = {
    "io_manager": {
        "config": {
            "base_dir": "data/processed/dev",
        }
    }
}

PROD_RESOURCES = {
    "io_manager": {
        "config": {
            "base_dir": "data/processed/prod",
        }
    }
}

TEST_RESOURCES = {
    "io_manager": {
        "config": {
            "base_dir": "data/processed/test",
            "sample_size": 100,
        }
    }
}


def get_run_config(environment: str = "dev", execution_mode: str = "parallel") -> RunConfig:
    """
    Get run configuration for specified environment and execution mode.

    Args:
        environment: Environment name ("dev", "prod", "test")
        execution_mode: Execution mode ("parallel", "sequential", "fast")

    Returns:
        RunConfig object with appropriate settings
    """
    # Select execution config
    if execution_mode == "parallel":
        exec_config = PARALLEL_CONFIG
    elif execution_mode == "sequential":
        exec_config = SEQUENTIAL_CONFIG
    elif execution_mode == "fast":
        exec_config = FAST_CONFIG
    else:
        exec_config = PARALLEL_CONFIG

    # Select resource config
    if environment == "prod":
        resource_config = PROD_RESOURCES
    elif environment == "test":
        resource_config = TEST_RESOURCES
    else:
        resource_config = DEV_RESOURCES

    # Combine configs
    run_config = {**exec_config}
    if "io_manager" in resource_config:
        run_config["resources"] = {"io_manager": resource_config["io_manager"]}

    return RunConfig(run_config)
