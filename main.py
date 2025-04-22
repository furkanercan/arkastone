from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

job_store = {}

# CORS so Streamlit can talk to the backend if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared in-memory state
current_job = {}
job_taken = False
progress_log = []
final_result = None

@app.post("/run_config")
def run_config(data: dict):
    session_id = data.get("session_id")
    config = data.get("config")

    if not session_id or not config:
        return {"error": "Missing session_id or config"}

    job_store[session_id] = {
        "job": config,
        "taken": False,
        "progress": [],
        "result": None,
    }
    return {"message": "Configuration received"}

@app.get("/get_job")
def get_job(session_id: str):
    job_data = job_store.get(session_id)

    if not job_data or job_data["taken"]:
        return {}

    job_data["taken"] = True
    return job_data["job"]

@app.post("/update_progress")
def update_progress(data: dict):
    session_id = data.get("session_id")
    progress = data.get("progress")

    if session_id in job_store:
        job_store[session_id]["progress"].append(progress)
        return {"message": "Progress received"}
    return {"error": "Invalid session_id"}

@app.get("/get_progress")
def get_progress(session_id: str):
    return job_store.get(session_id, {}).get("progress", [])

@app.post("/submit_result")
def submit_result(data: dict):
    session_id = data.get("session_id")
    result = data.get("result")

    if session_id in job_store:
        job_store[session_id]["result"] = result
        return {"message": "Result received"}
    return {"error": "Invalid session_id"}

@app.get("/get_final_result")
def get_final_result(session_id: str):
    return job_store.get(session_id, {}).get("result", {})