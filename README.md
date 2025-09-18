# NetMind Workflow

A comprehensive data processing and analysis workflow system built with modern Python tools, featuring parallel processing, pandas DataFrame integration, and Dagster orchestration.

## 🚀 Features

### ✅ **Enhanced DataLoader with pandas DataFrame Support**
- Professional Excel data loading with robust error handling
- Full pandas DataFrame integration for tabular data operations
- Advanced filtering, grouping, and analysis capabilities
- Memory-efficient chunked processing
- Comprehensive data validation and cleaning

### ✅ **Parallel & Sequential Processing**
- **Parallel Processing**: 21,080 items/second throughput
- **Sequential Processing**: 266,094 items/second for smaller datasets
- Configurable execution modes (thread-based, process-based)
- Memory-efficient chunked processing
- Comprehensive progress tracking and error handling

### ✅ **Dagster Integration**
- Asset-based data processing pipeline
- Scheduled job execution
- Web-based UI for monitoring and management
- Event-driven processing with sensors
- PostgreSQL metadata storage

### ✅ **Docker Containerization**
- Multi-stage Docker builds for optimized images
- Docker Compose orchestration for development and production
- Health checks and proper service dependencies
- Development and production configurations

### ✅ **Job Scheduling System**
- Flexible job scheduler with multiple execution modes
- Task-based architecture for modular processing
- Comprehensive monitoring and logging
- Support for both Dagster and standalone execution

## 📊 Performance Results

The integration tests demonstrate excellent performance:

```
🧪 DataLoader Tests: ✅ PASSED
   - Loaded 7,459 queries successfully
   - DataFrame shape: (7459, 5)
   - Found 10 knowledge domains, 8 unique workers

🧪 Parallel Processing: ✅ PASSED
   - Processed 7,459 items in parallel
   - Throughput: 21,080 items/second
   - Success rate: 100%

🧪 Sequential Processing: ✅ PASSED
   - Processed 7,459 items sequentially
   - Throughput: 266,094 items/second
   - Success rate: 100%
```

## 🏗️ Architecture

```
src/
├── data/
│   └── data_loader.py          # Enhanced DataLoader with pandas integration
├── dagster/
│   ├── assets.py              # Dagster assets for data processing
│   ├── jobs.py                # Dagster jobs and schedules
│   └── repository.py          # Dagster repository definition
├── scheduler/
│   ├── job_scheduler.py       # Main job scheduler
│   ├── parallel_processor.py  # Parallel processing capabilities
│   └── sequential_processor.py # Sequential processing
└── __init__.py
```

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone and navigate to the project
cd netmind-workflow

# Start development environment
./scripts/docker-run.sh start dev

# Or for production
./scripts/docker-run.sh start

# View logs
./scripts/docker-run.sh logs

# Stop services
./scripts/docker-run.sh stop
```

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run data processing
python -c "
from src.data.data_loader import create_default_dataloader
loader = create_default_dataloader()
data = loader.get_processed_data()
print(f'Loaded {len(data)} queries')
"

# Run with pandas DataFrame
python -c "
from src.data.data_loader import create_default_dataloader
loader = create_default_dataloader()
df = loader.to_dataframe()
print(f'DataFrame shape: {df.shape}')
stats = loader.group_by_domain()
print('Domain statistics:')
print(stats)
"
```

## 📈 DataLoader pandas Integration Examples

```python
from src.data.data_loader import create_default_dataloader

# Create data loader
loader = create_default_dataloader()

# Get processed data
data = loader.get_processed_data()

# Work with pandas DataFrame
df = loader.to_dataframe()
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Filter by domain
vo_tuyen_data = loader.get_domain_dataframe('VO_TUYEN')
print(f"VO_TUYEN queries: {len(vo_tuyen_data)}")

# Group and analyze
domain_stats = loader.group_by_domain()
worker_stats = loader.group_by_worker()

# Advanced filtering
search_results = loader.filter_dataframe(
    query_filter='Vinaphone',
    domain_filter='VO_TUYEN'
)
print(f"Search results: {len(search_results)}")

# Sample data
sample = loader.sample_dataframe(100)
print(f"Sample shape: {sample.shape}")
```

## ⚡ Parallel Processing Examples

```python
from src.scheduler.parallel_processor import ParallelProcessor, ParallelConfig
from src.scheduler.parallel_processor import parallel_data_validation
from src.data.data_loader import create_default_dataloader

# Setup
loader = create_default_dataloader()
config = ParallelConfig(max_workers=4, use_processes=False)

# Parallel data validation
result = parallel_data_validation(loader, config)
print(f"Processed: {result.processed_items}/{result.total_items}")
print(f"Success rate: {result.metadata['success_rate']:.2f}")
print(f"Items/second: {result.metadata['items_per_second']:.0f}")
```

## 🔧 Dagster Integration

### Start Dagster Services

```bash
# Start Dagster webserver
dagster-webserver -h 0.0.0.0 -p 3000 -w dagster_workspace.yaml

# In another terminal, start daemon
dagster-daemon run
```

### Access Dagster UI
- **Web UI**: http://localhost:3000
- **Jupyter Notebook**: http://localhost:8888 (dev mode)

### Available Dagster Assets
- `raw_netmind_data`: Raw Excel data loading
- `cleaned_netmind_data`: Data cleaning and preprocessing
- `processed_netmind_data`: Structured data processing
- `domain_statistics`: Domain-level analytics
- `worker_statistics`: Worker-level analytics
- `data_quality_report`: Comprehensive quality metrics

## 🐳 Docker Services

### Development Environment
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Production Environment
```bash
docker-compose up -d
```

### Available Services
- **netmind-app**: Main application
- **dagster-webserver**: Dagster UI (http://localhost:3000)
- **dagster-daemon**: Job scheduler
- **postgres-db**: Metadata storage
- **redis**: Caching (optional)
- **jupyter**: Notebook environment (dev only)

## 📋 Requirements

- Python 3.10+
- pandas 2.0+
- Dagster 1.5+
- Docker & Docker Compose
- PostgreSQL (for Dagster metadata)
- Redis (optional, for caching)

## 🔧 Configuration

### DataLoader Configuration
```python
from src.data.data_loader import DataLoaderConfig

config = DataLoaderConfig(
    file_path="data/dataset.xlsx",
    max_workers=4,
    chunk_size=1000,
    cache_enabled=True,
    validate_data=True
)
```

### Parallel Processing Configuration
```python
from src.scheduler.parallel_processor import ParallelConfig

config = ParallelConfig(
    max_workers=4,
    chunk_size=1000,
    use_processes=False,  # False for I/O-bound, True for CPU-bound
    enable_progress=True
)
```

## 📊 Data Format

The system expects Excel files with the following columns:
- **query**: The evaluation query text
- **knowledge_domain**: Domain classification (e.g., VO_TUYEN)
- **worker**: Comma-separated list of workers (e.g., "FUNCTION_CALLING,AGENT_X")

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes the `src` directory
2. **Dagster Issues**: Check PostgreSQL connection and Dagster configuration
3. **Memory Issues**: Reduce `chunk_size` in configuration
4. **Performance Issues**: Adjust `max_workers` based on system resources

### Getting Help

- Check the logs: `./scripts/docker-run.sh logs`
- View Dagster UI: http://localhost:3000
- Run diagnostics: `python test_integration.py`

---

**Built with ❤️ using Python, pandas, Dagster, and Docker**
