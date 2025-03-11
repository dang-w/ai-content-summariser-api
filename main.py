from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

# Import the router from the correct location
# Check which router file exists and use that one
if os.path.exists("app/api/routes.py"):
    from app.api.routes import router as api_router
elif os.path.exists("app/routers/api.py"):
    from app.routers.api import router as api_router
else:
    raise ImportError("Could not find router file")

app = FastAPI(title="AI Content Summariser API")

# Configure CORS - allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
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
