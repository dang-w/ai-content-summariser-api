from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(
    title="AI Content Summariser API",
    description="API for summarising text content using NLP models",
    version="0.1.0"
)

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Content Summariser API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Import and include API routes
from app.api.routes import router as api_router
app.include_router(api_router, prefix="/api")

# Import and include async API routes
from app.api.async_routes import router as async_router
app.include_router(async_router, prefix="/api/async")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
