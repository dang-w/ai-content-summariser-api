from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import os

# Import the router
from app.api.routes import router as api_router

app = FastAPI(
    title="AI Content Summariser API",
    description="An API for summarizing text content and web pages using NLP models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-content-summariser.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router
app.include_router(api_router)

@app.get("/", include_in_schema=True)
async def root():
    """
    Root endpoint that provides information about the API and available endpoints.
    """
    return {
        "name": "AI Content Summariser API",
        "version": "1.0.0",
        "description": "API for summarizing text content and web pages using NLP models",
        "endpoints": {
            "documentation": "/docs",
            "alternative_docs": "/redoc",
            "health_check": "/health",
            "api_endpoints": {
                "summarise_text": "/api/summarise",
                "summarise_url": "/api/summarise-url",
                "status": "/api/status"
            }
        },
        "github_repository": "https://github.com/dang-w/ai-content-summariser-api",
        "frontend_application": "https://ai-content-summariser.vercel.app"
    }

@app.get("/docs-redirect")
async def docs_redirect():
    """
    Redirects to the API documentation.
    """
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Global exception handler for better error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
