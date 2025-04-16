import time
import requests
import json
from src.sim.sim_runner import run_simulation_from_config  # your real engine

SERVER_URL = "https://arkastone-backend.onrender.com"

def fetch_job():
    try:
        r = requests.get(f"{SERVER_URL}/get_job")
        if r.status_code == 200 and r.text.strip():
            return r.json()
    except Exception as e:
        print("Error fetching job:", e)
    return None

def make_json_serializable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic, np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def send_progress(update):
    try:
        safe_update = make_json_serializable(update)
        json.dumps(safe_update)  # <-- this triggers JSON check
        r = requests.post(f"{SERVER_URL}/update_progress", json=safe_update)
        # print("Progress sent:", r.status_code)
    except Exception as e:
        print("Error sending progress:", e)
        print("Offending update payload:", update)

def send_result(result):
    try:
        safe_result = make_json_serializable(result)
        r = requests.post(f"{SERVER_URL}/submit_result", json=safe_result)
        print("Result submitted:", r.status_code)
    except Exception as e:
        print("Error submitting result:", e)

def main():
    print("Client started. Polling for jobs...")
    while True:
        job = fetch_job()
        if job:
            print("Job received. Running simulation...")
            result = run_simulation_from_config(job, progress_callback=send_progress) #pack this later.
            send_result(result)
        else:
            print("No job found. Retrying...")
        time.sleep(5)

if __name__ == "__main__":
    main()
