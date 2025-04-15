from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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
async def run_config(request: Request):
    global current_job, job_taken, progress_log, final_result
    current_job = await request.json()
    job_taken = False
    progress_log.clear()
    final_result = None
    return {"status": "job loaded"}

@app.get("/get_job")
def get_job():
    global job_taken
    if not job_taken and current_job:
        job_taken = True
        return current_job
    return {}

@app.post("/update_progress")
async def update_progress(request: Request):
    global progress_log
    update = await request.json()
    progress_log.append(update)
    return {"status": "progress updated"}

@app.get("/get_progress")
def get_progress():
    return progress_log

@app.post("/submit_result")
async def submit_result(request: Request):
    global final_result
    final_result = await request.json()
    return {"status": "result received"}

@app.get("/get_final_result")
def get_final_result():
    return final_result if final_result else {"status": "pending"}