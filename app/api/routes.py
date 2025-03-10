from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Union
from app.services.summariser import SummariserService
from app.services.url_extractor import URLExtractorService

router = APIRouter()

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
        summariser = SummariserService()
        summary = summariser.summarise(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=request.do_sample,
            temperature=request.temperature
        )

        return {
            "original_text_length": len(request.text),
            "summary": summary,
            "summary_length": len(summary),
            "source_type": "text"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summarise-url", response_model=SummaryResponse)
async def summarise_url(request: URLSummaryRequest):
    try:
        # Extract content from URL
        url_extractor = URLExtractorService()
        content = await url_extractor.extract_content(str(request.url))

        if not content or len(content) < 100:
            raise HTTPException(status_code=422, detail="Could not extract sufficient content from the URL")

        # Summarise the extracted content
        summariser = SummariserService()
        summary = summariser.summarise(
            text=content,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=request.do_sample,
            temperature=request.temperature
        )

        return {
            "original_text_length": len(content),
            "summary": summary,
            "summary_length": len(summary),
            "source_type": "url",
            "source_url": str(request.url)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
