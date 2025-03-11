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

- Text summarization using state-of-the-art NLP models (BART-large-CNN)
- URL content extraction and summarization
- Adjustable parameters for summary length and style
- Efficient API endpoints with proper error handling

## API Endpoints

- `POST /api/summarise` - Summarize text content
- `POST /api/summarise-url` - Extract and summarize content from a URL

## Technology Stack

- **Framework**: FastAPI for efficient API endpoints
- **NLP Models**: Transformer-based models (BART) for summarisation
- **Web Scraping**: BeautifulSoup4 for extracting content from URLs
- **HTTP Client**: HTTPX for asynchronous web requests
- **Deployment**: Hugging Face Spaces or Docker containers

## Getting Started

### Prerequisites

- Python (v3.8+)
- pip

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

### Running Locally

```bash
# Start the backend server
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

## Testing

The project includes a comprehensive test suite covering both unit and integration tests.

### Installing Test Dependencies

```bash
pip install pytest pytest-cov httpx
```

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

### Test Structure

- **Unit Tests**: Test individual components in isolation
  - `tests/test_summariser.py`: Tests for the summarization service

- **Integration Tests**: Test API endpoints and component interactions
  - `tests/test_api.py`: Tests for API endpoints

### Mocking Strategy

For faster and more reliable tests, we use mocking to avoid loading large ML models during testing:

```python
# Example of mocked test
def test_summariser_with_mock():
    with patch('app.services.summariser.AutoTokenizer') as mock_tokenizer_class, \
         patch('app.services.summariser.AutoModelForSeq2SeqLM') as mock_model_class:
        # Test implementation...
```

### Continuous Integration

Tests are automatically run on pull requests and pushes to the main branch using GitHub Actions.

## Running with Docker

```bash
# Build and run with Docker
docker build -t ai-content-summariser-api .
docker run -p 8000:8000 ai-content-summariser-api
```

## Deployment

See the deployment guide in the frontend repository for detailed instructions on deploying both the frontend and backend components.

### Deploying to Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Choose Docker as the SDK
3. Upload your backend code
4. Configure the environment variables:
   - `CORS_ORIGINS`: Your frontend URL

## Performance Optimizations

The API includes several performance optimizations:

1. **Model Caching**: Models are loaded once and cached for subsequent requests
2. **Result Caching**: Frequently requested summaries are cached to avoid redundant processing
3. **Asynchronous Processing**: Long-running tasks are processed asynchronously

## Development

### Testing the API

You can test the API endpoints using the built-in Swagger documentation at `/docs` when running locally.

### Checking Transformers Installation

To verify that the transformers library is installed correctly:

```bash
python -m app.check_transformers
```

## License

This project is licensed under the MIT License.
