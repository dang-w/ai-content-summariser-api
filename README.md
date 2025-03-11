---
title: AI Content Summariser API
emoji: 📝
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AI Content Summariser API (Backend)

This is the backend API for the AI Content Summariser, a tool that automatically generates concise summaries of articles, documents, and web content using natural language processing.

The frontend application is available in a separate repository: [ai-content-summariser](https://github.com/dang-w/ai-content-summariser).

## Features

- **Text Summarization**: Generate concise summaries using BART-large-CNN model
- **URL Content Extraction**: Automatically extract and process content from web pages
- **Adjustable Parameters**: Control summary length (30-500 chars) and style
- **Advanced Generation Options**: Temperature control (0.7-2.0) and sampling options
- **Caching System**: Store results to improve performance and reduce redundant processing
- **Status Monitoring**: Track model loading and summarization progress in real-time
- **Error Handling**: Robust error handling for various input scenarios
- **CORS Support**: Configured for cross-origin requests from the frontend

## API Endpoints

- `POST /api/summarise` - Summarize text content
- `POST /api/summarise-url` - Extract and summarize content from a URL
- `GET /api/status` - Get the current status of the model and any running jobs
- `GET /health` - Health check endpoint for monitoring

## Technology Stack

- **Framework**: FastAPI for efficient API development
- **NLP Models**: Hugging Face Transformers (BART-large-CNN)
- **Web Scraping**: BeautifulSoup4 for extracting content from URLs
- **HTTP Client**: HTTPX for asynchronous web requests
- **ML Framework**: PyTorch for running the NLP models
- **Testing**: Pytest for unit and integration testing
- **Deployment**: Docker containers on Hugging Face Spaces

## Project Structure

```
ai-content-summariser-api/
├── app/
│   ├── api/
│   │   └── routes.py      # API endpoints
│   ├── services/
│   │   ├── summariser.py  # Text summarization service
│   │   ├── url_extractor.py # URL content extraction
│   │   └── cache.py       # Caching functionality
│   └── check_transformers.py # Utility to verify model setup
├── tests/
│   ├── test_api.py        # API endpoint tests
│   └── test_summariser.py # Summarizer service tests
├── main.py                # Application entry point
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (not in repo)
```

## Getting Started

### Prerequisites

- Python (v3.8+)
- pip
- At least 4GB of RAM (8GB recommended for optimal performance)
- GPU support (optional, but recommended for faster processing)

### Installation

```bash
# Clone the repository
git clone https://github.com/dang-w/ai-content-summariser-api.git
cd ai-content-summariser-api

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the root directory with the following variables:

```
ENVIRONMENT=development
CORS_ORIGINS=http://localhost:3000,https://ai-content-summariser.vercel.app
TRANSFORMERS_CACHE=/path/to/cache  # Optional: custom cache location
```

### Running Locally

```bash
# Start the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`.

## Testing

The project includes a comprehensive test suite covering both unit and integration tests.

### Running Tests

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests and generate coverage report
pytest --cov=app tests/

# Run tests and generate detailed coverage report
pytest --cov=app --cov-report=term-missing tests/

# Run specific test file
pytest tests/test_api.py

# Run tests without warnings
pytest -W ignore::FutureWarning -W ignore::UserWarning
```

## Docker Deployment

```bash
# Build and run with Docker
docker build -t ai-content-summariser-api .
docker run -p 8000:8000 ai-content-summariser-api
```

## Deployment to Hugging Face Spaces

When deploying to Hugging Face Spaces:

1. Fork this repository to your Hugging Face account
2. Set the following environment variables in the Space settings:
   - `TRANSFORMERS_CACHE=/tmp/huggingface_cache`
   - `HF_HOME=/tmp/huggingface_cache`
   - `HUGGINGFACE_HUB_CACHE=/tmp/huggingface_cache`
   - `CORS_ORIGINS=https://ai-content-summariser.vercel.app,http://localhost:3000`
3. Ensure the Space is configured to use the Docker SDK
4. Your API will be available at `https://huggingface.co/spaces/your-username/ai-content-summariser-api`

## Performance Optimizations

The API includes several performance optimizations:

1. **Model Caching**: Models are loaded once and cached for subsequent requests
2. **Result Caching**: Frequently requested summaries are cached to avoid redundant processing
3. **Asynchronous Processing**: Long-running tasks are processed asynchronously
4. **Text Preprocessing**: Input text is cleaned and normalized before processing
5. **Batched Processing**: Large texts are processed in batches for better memory management

## API Request Examples

### Text Summarization

```bash
curl -X 'POST' \
  'http://localhost:8000/api/summarise' \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Your long text to summarize goes here...",
    "max_length": 150,
    "min_length": 50,
    "do_sample": true,
    "temperature": 1.2
  }'
```

### URL Summarization

```bash
curl -X 'POST' \
  'http://localhost:8000/api/summarise-url' \
  -H 'Content-Type: application/json' \
  -d '{
    "url": "https://example.com/article",
    "max_length": 150,
    "min_length": 50,
    "do_sample": true,
    "temperature": 1.2
  }'
```

## License

This project is licensed under the MIT License.
