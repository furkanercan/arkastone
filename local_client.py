import time
import requests
import json
import sys
import argparse
from functools import partial
from src.sim.sim_runner import run_simulation_from_config  # your real engine

SERVER_URL = "https://arkastone-backend.onrender.com"

def fetch_job(session_id):
    try:
        r = requests.get(f"{SERVER_URL}/get_job", params={"session_id": session_id})
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

def send_progress(update, session_id):
    try:
        safe_update = make_json_serializable(update)
        json.dumps(safe_update)  # test serializability

        payload = {
            "session_id": session_id,
            "progress": safe_update
        }

        r = requests.post(f"{SERVER_URL}/update_progress", json=payload)
        # print("Progress sent:", r.status_code)
    except Exception as e:
        print("Error sending progress:", e)
        print("Offending update payload:", update)


def send_result(result, session_id):
    try:
        safe_result = make_json_serializable(result)

        payload = {
            "session_id": session_id,
            "result": safe_result
        }

        r = requests.post(f"{SERVER_URL}/submit_result", json=payload)
        print("Result submitted:", r.status_code)
    except Exception as e:
        print("Error submitting result:", e)

def main():
    args = parse_args()
    session_id = args.session

    if not session_id:
        session_id = input("Enter your session ID: ").strip()

    print("Client started. Polling for jobs...")
    while True:
        job = fetch_job(session_id)
        if job:
            print("Job received. Running simulation...")
            progress_with_session = partial(send_progress, session_id=session_id)
            result = run_simulation_from_config(job, progress_callback=progress_with_session)
            send_result(result, session_id)
        else:
            print("No job found. Retrying...")
        time.sleep(5)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Arkastone Client")
    parser.add_argument("--session", type=str, help="Session ID from the UI")
    return parser.parse_args()

if __name__ == "__main__":
    main()
