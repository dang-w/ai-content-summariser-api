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

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
# Start the backend server
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### Running with Docker

```bash
# Build and run with Docker
docker build -t ai-content-summariser-api .
docker run -p 8000:8000 ai-content-summariser-api
```

## Deployment

See the deployment guide in the frontend repository for detailed instructions on deploying both the frontend and backend components.

### Deploying to Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Choose FastAPI as the SDK
3. Upload your backend code
4. Configure the environment variables:
   - `CORS_ORIGINS`: Your frontend URL

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
