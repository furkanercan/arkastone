import time
import requests
import json
import argparse
from functools import partial
from src.sim.sim_runner import run_simulation_from_config

SERVER_URL = "https://arkastone-backend.onrender.com"

def fetch_job(session_id):
    try:
        r = requests.get(f"{SERVER_URL}/get_job", params={"session_id": session_id}, timeout=10)
        r.raise_for_status()
        if r.text.strip():
            return r.json()
    except requests.RequestException as e:
        print(f"[!] Network error fetching job: {e}")
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

def safe_post(endpoint, payload, description):
    try:
        safe_payload = make_json_serializable(payload)
        json.dumps(safe_payload)  # validate
        r = requests.post(f"{SERVER_URL}/{endpoint}", json=safe_payload, timeout=10)
        print(f"[+] {description} sent: {r.status_code}")
    except requests.RequestException as e:
        print(f"[!] Network error sending {description}: {e}")
    except Exception as e:
        print(f"[!] Serialization error for {description}: {e}")

def send_progress(update, session_id):
    payload = {
        "session_id": session_id,
        "progress": update
    }
    safe_post("update_progress", payload, "progress")

def send_result(result, session_id):
    payload = {
        "session_id": session_id,
        "result": result
    }
    safe_post("submit_result", payload, "result")

def main():
    args = parse_args()
    session_id = args.session or input("Enter your session ID: ").strip()

    print("[*] Client started. Polling for jobs...")
    try:
        while True:
            job = fetch_job(session_id)
            if job:
                print("[*] Job received. Running simulation...")
                progress_with_session = partial(send_progress, session_id=session_id)
                result = run_simulation_from_config(job, progress_callback=progress_with_session)
                send_result(result, session_id)
                print("[*] Simulation complete. Waiting for next job...")
            else:
                print("[*] No job found. Retrying in 5 seconds...")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[!] Client stopped manually.")
    except Exception as e:
        print(f"[!] Unexpected error: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Arkastone Client")
    parser.add_argument("--session", type=str, help="Session ID from the UI")
    return parser.parse_args()

if __name__ == "__main__":
    main()
