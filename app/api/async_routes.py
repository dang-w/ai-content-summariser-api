import asyncio
import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.api.routes import TextSummaryRequest
from app.services.summariser import SummariserService

router = APIRouter()

# In-memory storage for task results (use Redis or a database in production)
task_results = {}

async def process_summarization(task_id, request):
    try:
        summariser = SummariserService()
        summary = summariser.summarise(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            do_sample=request.do_sample,
            temperature=request.temperature
        )

        task_results[task_id] = {
            "status": "completed",
            "result": {
                "original_text_length": len(request.text),
                "summary": summary,
                "summary_length": len(summary),
                "source_type": "text"
            }
        }
    except Exception as e:
        task_results[task_id] = {
            "status": "failed",
            "error": str(e)
        }

@router.post("/summarise-async")
async def summarise_text_async(request: TextSummaryRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    task_results[task_id] = {"status": "processing"}

    background_tasks.add_task(process_summarization, task_id, request)

    return {"task_id": task_id, "status": "processing"}

@router.get("/summary-status/{task_id}")
async def get_summary_status(task_id: str):
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")

    return task_results[task_id]
