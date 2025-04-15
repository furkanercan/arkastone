from fastapi import FastAPI, Request

app = FastAPI()

# Shared state
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

@app.get("/get_progress")
def get_progress():
    return progress_log

@app.get("/get_final_result")
def get_final_result():
    return final_result if final_result else {"status": "pending"}
