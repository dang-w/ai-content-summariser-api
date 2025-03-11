from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Import the router
from app.api.routes import router as api_router

app = FastAPI(title="AI Content Summariser API")

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

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
