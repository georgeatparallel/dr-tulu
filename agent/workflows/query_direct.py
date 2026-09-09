import asyncio, sys
sys.path.insert(0, "/weka/nora-default/varshak/dr-tulu/agent")
from workflows.auto_search_sft import AutoReasonSearchWorkflow
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class Query(BaseModel):
    question: str
    dataset_name: str = "sqav2"

@app.post("/ask")
async def ask(query: Query):
    workflow = AutoReasonSearchWorkflow(
        configuration="auto_search_sft_s2_only_hamish.yaml"
    )
    result = await workflow(
        problem=query.question,
        dataset_name=query.dataset_name,
        verbose=False,
    )
    ft = result.get("full_traces")
    return {
        "answer": result["final_response"],
        "trace": ft.model_dump() if ft and hasattr(ft, "model_dump") else ft,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007)