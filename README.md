# NetMind Workflow

A comprehensive data processing and analysis workflow system built with modern Python tools, featuring parallel processing, pandas DataFrame integration, and a simple FastAPI service.

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

### ✅ **FastAPI Service**
- Clean REST API to submit queries and get worker labels
- Endpoints: `/health`, `/labels`, `/label`

### ✅ **Docker Containerization**
- Multi-stage Docker builds for optimized images
- Single docker-compose.yml with dev profile for hot-reload
- Health checks and proper service dependencies
- Separate production compose using Docker Hub images

### ✅ **Job Scheduling System**
- Flexible job scheduler with multiple execution modes
- Task-based architecture for modular processing
- Comprehensive monitoring and logging

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
├── models/
│   └── nlp.py                 # Comprehensive NLP processing wrapper
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

# Development (instant hot-reload, no rebuilds)
make run-dev

# Production (Docker Hub image via unified compose prod profile)
API_IMAGE=your-dockerhub-user/netmind-workflow:tag make up-prod

# View logs
make logs

# Stop prod services
make down-prod
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

## 🧠 **NLP Processing Capabilities**

### **Multi-Language NLP Support**
- **English Processing**: spaCy + NLTK + VADER sentiment analysis
- **Vietnamese Processing**: underthesea library
- **Automatic Language Detection**: langdetect integration
- **Object-Oriented Design**: Abstract base classes with inheritance

### **NLP Features**
- ✅ **Lexical Analysis**: Tokenization, POS tagging, lemmatization
- ✅ **Syntactic Analysis**: Dependency parsing
- ✅ **Semantic Analysis**: Sentiment analysis, named entity recognition
- ✅ **Text Classification**: Topic modeling, keyword extraction
- ✅ **Grammar Analysis**: Basic grammar checking and validation
- ✅ **Comprehensive Analysis**: Readability metrics, lexical diversity

### **NLP Usage Examples**

```python
from src.models.nlp import create_default_nlp_processor, process_query_text

# Create NLP processor
processor = create_default_nlp_processor()

# Basic text processing
result = processor.process_text("NetMind provides excellent workflow automation")
print(f"Language: {result.language.value}")
print(f"Sentiment: {result.sentiment}")
print(f"Entities: {result.entities}")

# Comprehensive analysis
analysis = processor.analyze_text_comprehensive("Your query text here")
print(f"Sentiment Score: {analysis.sentiment_score:.3f}")
print(f"Keywords: {analysis.keywords}")
print(f"Topics: {analysis.topics}")

# Quick query processing
result = process_query_text("How to check internet speed?")
print(f"Language: {result.language.value}")
print(f"Sentiment: {result.sentiment.get('compound', 0.0):.3f}")

    # Check grammar issues
    analysis = processor.analyze_text_comprehensive("Hello world!")
    print(f"Grammar issues: {analysis.grammar_issues}")
```

### **NLP Demo Script**
```bash
# Run comprehensive NLP demonstration
python nlp_demo.py
```

### **API Usage**
```bash
# Health
curl http://localhost:8000/health

# List labels
curl http://localhost:8000/labels

# Label a query
curl -X POST http://localhost:8000/label -H 'Content-Type: application/json' -d '{"query":"/task finish report by tomorrow"}'
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

## 🔧 API Endpoints
See FastAPI app at `src/api/app.py` for details: `/health`, `/labels`, `/label`.

## 🐳 Docker Services

### Development Environment
```bash
make run-dev
```

### Production Environment
```bash
API_IMAGE=your-dockerhub-user/netmind-workflow:tag make up-prod
```

### Available Services
- **api** or **api-dev**: FastAPI service
- **redis**: Caching (optional)
- **jupyter**: Notebook environment (dev only)

## 📋 Requirements

### **Core Dependencies**
- Python 3.10+
- pandas 2.0+
- Docker & Docker Compose
- Redis (optional, for caching)

### **NLP Dependencies**
- **spaCy**: English language processing
- **underthesea**: Vietnamese language processing
- **NLTK**: General NLP toolkit
- **VADER**: Sentiment analysis
- **langdetect**: Language detection
- **textblob**: Additional text processing utilities

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
2. **API Issues**: Check container logs with `make logs`
3. **Memory Issues**: Reduce `chunk_size` in configuration
4. **Performance Issues**: Adjust `max_workers` based on system resources

### Getting Help

- Check the logs: `./scripts/docker-run.sh logs`
- View Dagster UI: http://localhost:3000
- Run diagnostics: `python test_integration.py`

---

**Built with ❤️ using Python, pandas, FastAPI, and Docker**
