from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Union
from app.services.summariser import SummariserService
from app.services.url_extractor import URLExtractorService
from app.services.cache import hash_text, get_cached_summary, cache_summary
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")
summariser_service = SummariserService()

class TextSummaryRequest(BaseModel):
    text: str = Field(..., min_length=10, description="The text to summarise")
    max_length: Optional[int] = Field(150, ge=30, le=500, description="Maximum length of the summary")
    min_length: Optional[int] = Field(50, ge=10, le=200, description="Minimum length of the summary")
    do_sample: Optional[bool] = Field(False, description="Whether to use sampling for generation")
    temperature: Optional[float] = Field(1.0, ge=0.7, le=2.0, description="Sampling temperature")

class URLSummaryRequest(BaseModel):
    url: HttpUrl = Field(..., description="The URL to extract content from and summarise")
    max_length: Optional[int] = Field(150, ge=30, le=500, description="Maximum length of the summary")
    min_length: Optional[int] = Field(50, ge=10, le=200, description="Minimum length of the summary")
    do_sample: Optional[bool] = Field(False, description="Whether to use sampling for generation")
    temperature: Optional[float] = Field(1.0, ge=0.7, le=2.0, description="Sampling temperature")

class SummaryResponse(BaseModel):
    original_text_length: int
    summary: str
    summary_length: int
    source_type: str = "text"  # "text" or "url"
    source_url: Optional[str] = None

@router.post("/summarise", response_model=SummaryResponse)
async def summarise_text(request: TextSummaryRequest):
    try:
        # Check cache first
        text_hash = hash_text(request.text)
        cached_summary = get_cached_summary(
            text_hash,
            request.max_length,
            request.min_length,
            request.do_sample,
            request.temperature
        )

        if cached_summary:
            return cached_summary

        # If not in cache, generate summary
        result = summariser_service.summarise(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=request.do_sample,
            temperature=request.temperature
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarise-url", response_model=SummaryResponse)
async def summarise_url(request: URLSummaryRequest):
    try:
        # Extract content from URL
        logger.info(f"Extracting content from URL: {request.url}")
        url_extractor = URLExtractorService()
        content = await url_extractor.extract_content(str(request.url))

        if not content or len(content) < 100:
            logger.warning(f"Insufficient content extracted from URL: {request.url}")
            raise HTTPException(status_code=422, detail="Could not extract sufficient content from the URL")

        logger.info(f"Extracted {len(content)} characters from {request.url}")

        # Summarise the extracted content
        result = summariser_service.summarise(
            text=content,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=request.do_sample,
            temperature=request.temperature
        )

        # Create a more structured response
        return {
            "original_text_length": len(content),
            "summary": result["summary"],
            "summary_length": len(result["summary"]),
            "source_type": "url",
            "source_url": str(request.url),
            "metadata": result.get("metadata", {})
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing URL {request.url}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_status():
    """Get the current status of the summariser service"""
    return summariser_service.get_status()
