"""
Job Scheduler for NetMind Workflow

This module provides a flexible job scheduler that supports both parallel
and sequential execution of data processing tasks.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import multiprocessing

import sys
from pathlib import Path

# Add src to path for imports
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from data.data_loader import DataLoader, DataLoaderConfig

# Configure logging
logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for job scheduling."""
    SEQUENTIAL = "sequential"
    PARALLEL_THREAD = "parallel_thread"
    PARALLEL_PROCESS = "parallel_process"


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobConfig:
    """Configuration for job execution."""

    name: str
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    max_workers: int = 4
    chunk_size: int = 1000
    timeout: Optional[float] = None
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_caching: bool = True
    output_directory: Optional[Path] = None
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate configuration."""
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.retry_attempts < 0:
            raise ValueError("retry_attempts must be >= 0")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be >= 0")


@dataclass
class JobResult:
    """Result of job execution."""

    job_name: str
    status: JobStatus
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate duration if end_time is set."""
        if self.end_time is not None:
            self.duration = self.end_time - self.start_time

    def mark_completed(self, output: Any = None, metadata: Dict[str, Any] = None):
        """Mark job as completed."""
        self.status = JobStatus.COMPLETED
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.output = output
        if metadata:
            self.metadata.update(metadata)

    def mark_failed(self, error: str, metadata: Dict[str, Any] = None):
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.error = error
        if metadata:
            self.metadata.update(metadata)

    def mark_cancelled(self):
        """Mark job as cancelled."""
        self.status = JobStatus.CANCELLED
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time


class JobTask(ABC):
    """Abstract base class for job tasks."""

    def __init__(self, name: str, config: JobConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def execute(self, data_loader: DataLoader) -> Any:
        """Execute the job task."""
        pass

    def validate(self) -> bool:
        """Validate task configuration and requirements."""
        return True

    def cleanup(self):
        """Cleanup resources after task execution."""
        pass


class DataLoadingTask(JobTask):
    """Task for loading and processing data."""

    def execute(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Execute data loading task."""
        self.logger.info(f"Starting data loading task: {self.name}")

        # Load and process data
        processed_data = data_loader.get_processed_data()

        # Get statistics
        stats = data_loader.get_statistics()

        result = {
            'processed_data': processed_data,
            'statistics': stats,
            'total_queries': len(processed_data),
        }

        self.logger.info(f"Data loading completed: {len(processed_data)} queries processed")
        return result


class DataAnalysisTask(JobTask):
    """Task for analyzing processed data."""

    def execute(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Execute data analysis task."""
        self.logger.info(f"Starting data analysis task: {self.name}")

        # Get processed data
        processed_data = data_loader.get_processed_data()

        # Perform various analyses
        df = data_loader.to_dataframe()
        domain_stats = data_loader.group_by_domain()
        worker_stats = data_loader.group_by_worker()

        # Additional analysis
        analysis_result = {
            'dataframe_shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'domain_distribution': domain_stats.to_dict('records'),
            'worker_distribution': worker_stats.to_dict('records'),
            'avg_query_length': df['query_length'].mean(),
            'total_assignments': len(processed_data),
        }

        self.logger.info("Data analysis completed")
        return analysis_result


class DataExportTask(JobTask):
    """Task for exporting data to various formats."""

    def __init__(self, name: str, config: JobConfig, export_formats: List[str] = None):
        super().__init__(name, config)
        self.export_formats = export_formats or ['csv', 'json']

    def execute(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Execute data export task."""
        self.logger.info(f"Starting data export task: {self.name}")

        df = data_loader.to_dataframe()
        export_results = {}

        for fmt in self.export_formats:
            try:
                output_dir = self.config.output_directory or Path("outputs")
                output_dir.mkdir(exist_ok=True)

                if fmt == 'csv':
                    output_path = output_dir / f"processed_data_{int(time.time())}.csv"
                    df.to_csv(output_path, index=False)
                elif fmt == 'json':
                    output_path = output_dir / f"processed_data_{int(time.time())}.json"
                    df.to_json(output_path, orient='records', indent=2)
                elif fmt == 'parquet':
                    try:
                        import pyarrow as pa
                        output_path = output_dir / f"processed_data_{int(time.time())}.parquet"
                        df.to_parquet(output_path, index=False)
                    except ImportError:
                        self.logger.warning("PyArrow not available, skipping Parquet export")
                        continue
                else:
                    self.logger.warning(f"Unsupported export format: {fmt}")
                    continue

                export_results[fmt] = str(output_path)
                self.logger.info(f"Exported data to {output_path}")

            except Exception as e:
                self.logger.error(f"Failed to export {fmt}: {e}")
                export_results[f"{fmt}_error"] = str(e)

        return export_results


class JobScheduler:
    """
    Main job scheduler for orchestrating data processing tasks.

    Supports parallel and sequential execution modes with comprehensive
    monitoring and error handling.
    """

    def __init__(self, data_config: DataLoaderConfig):
        self.data_config = data_config
        self.data_loader = DataLoader(data_config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._lock = threading.Lock()
        self._running_jobs: Dict[str, JobResult] = {}

    def submit_job(self, job_config: JobConfig, tasks: List[JobTask]) -> str:
        """
        Submit a job for execution.

        Args:
            job_config: Job configuration
            tasks: List of tasks to execute

        Returns:
            Job ID
        """
        job_id = f"{job_config.name}_{int(time.time())}"

        with self._lock:
            self._running_jobs[job_id] = JobResult(
                job_name=job_config.name,
                status=JobStatus.PENDING,
                start_time=time.time(),
                metadata={'job_id': job_id, 'config': job_config.__dict__}
            )

        # Start job execution in background
        thread = threading.Thread(
            target=self._execute_job,
            args=(job_id, job_config, tasks),
            daemon=True
        )
        thread.start()

        self.logger.info(f"Job submitted: {job_id}")
        return job_id

    def _execute_job(self, job_id: str, job_config: JobConfig, tasks: List[JobTask]):
        """Execute a job with the specified configuration."""
        try:
            self._running_jobs[job_id].status = JobStatus.RUNNING
            self.logger.info(f"Starting job execution: {job_id}")

            # Execute tasks based on execution mode
            if job_config.execution_mode == ExecutionMode.SEQUENTIAL:
                results = self._execute_sequential(job_config, tasks)
            elif job_config.execution_mode == ExecutionMode.PARALLEL_THREAD:
                results = self._execute_parallel_thread(job_config, tasks)
            elif job_config.execution_mode == ExecutionMode.PARALLEL_PROCESS:
                results = self._execute_parallel_process(job_config, tasks)
            else:
                raise ValueError(f"Unsupported execution mode: {job_config.execution_mode}")

            # Mark job as completed
            self._running_jobs[job_id].mark_completed(
                output=results,
                metadata={'execution_mode': job_config.execution_mode.value}
            )

            self.logger.info(f"Job completed successfully: {job_id}")

        except Exception as e:
            error_msg = f"Job execution failed: {e}"
            self.logger.error(error_msg)
            self._running_jobs[job_id].mark_failed(error_msg)

    def _execute_sequential(self, job_config: JobConfig, tasks: List[JobTask]) -> List[Any]:
        """Execute tasks sequentially."""
        results = []
        for task in tasks:
            try:
                result = task.execute(self.data_loader)
                results.append(result)
                self.logger.debug(f"Task completed: {task.name}")
            except Exception as e:
                self.logger.error(f"Task failed: {task.name} - {e}")
                if not job_config.retry_attempts or len(results) == 0:
                    raise
                # Retry logic could be implemented here
                raise
        return results

    def _execute_parallel_thread(self, job_config: JobConfig, tasks: List[JobTask]) -> List[Any]:
        """Execute tasks in parallel using threads."""
        results = []

        with ThreadPoolExecutor(max_workers=job_config.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(task.execute, self.data_loader): task
                for task in tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=job_config.timeout)
                    results.append(result)
                    self.logger.debug(f"Task completed: {task.name}")
                except Exception as e:
                    self.logger.error(f"Task failed: {task.name} - {e}")
                    raise

        return results

    def _execute_parallel_process(self, job_config: JobConfig, tasks: List[JobTask]) -> List[Any]:
        """Execute tasks in parallel using processes."""
        results = []

        with ProcessPoolExecutor(max_workers=min(job_config.max_workers, multiprocessing.cpu_count())) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._execute_task_in_process, task): task
                for task in tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=job_config.timeout)
                    results.append(result)
                    self.logger.debug(f"Task completed: {task.name}")
                except Exception as e:
                    self.logger.error(f"Task failed: {task.name} - {e}")
                    raise

        return results

    def _execute_task_in_process(self, task: JobTask) -> Any:
        """Execute a task in a separate process (helper method)."""
        # Create a new data loader instance for the process
        data_loader = DataLoader(self.data_config)
        return task.execute(data_loader)

    def get_job_status(self, job_id: str) -> Optional[JobResult]:
        """Get the status of a job."""
        return self._running_jobs.get(job_id)

    def list_jobs(self) -> List[JobResult]:
        """List all jobs and their status."""
        return list(self._running_jobs.values())

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id in self._running_jobs:
            self._running_jobs[job_id].mark_cancelled()
            self.logger.info(f"Job cancelled: {job_id}")
            return True
        return False

    def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> JobResult:
        """Wait for a job to complete."""
        start_time = time.time()

        while True:
            job_result = self.get_job_status(job_id)
            if job_result is None:
                raise ValueError(f"Job not found: {job_id}")

            if job_result.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return job_result

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

            time.sleep(0.1)  # Small delay to avoid busy waiting


# Convenience functions for common use cases

def create_data_loading_job(data_path: Union[str, Path] = "data/dataset.xlsx",
                           execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL) -> JobConfig:
    """Create a job configuration for data loading."""
    return JobConfig(
        name="data_loading",
        execution_mode=execution_mode,
        max_workers=4 if execution_mode != ExecutionMode.SEQUENTIAL else 1,
    )


def create_data_analysis_job(execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL) -> JobConfig:
    """Create a job configuration for data analysis."""
    return JobConfig(
        name="data_analysis",
        execution_mode=execution_mode,
        max_workers=2 if execution_mode != ExecutionMode.SEQUENTIAL else 1,
    )


def create_data_export_job(export_formats: List[str] = None,
                          output_directory: Path = None) -> JobConfig:
    """Create a job configuration for data export."""
    return JobConfig(
        name="data_export",
        execution_mode=ExecutionMode.SEQUENTIAL,
        max_workers=1,
        output_directory=output_directory,
    )
