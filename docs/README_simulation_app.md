# Arkastone Distributed Communication Simulator

This project implements a distributed simulation framework for communication systems. It includes:
- A **Streamlit-based UI** for configuring simulations and visualizing results
- A **FastAPI backend** for job orchestration and result tracking
- A **Python executable client** that runs simulations on the user's machine

---

## 🚀 Components Overview

### 1. **Frontend (Streamlit UI)**
- File: `streamlit_app_online.py`
- Role: Build configuration, submit job to backend, and visualize real-time results.

### 2. **Backend (FastAPI)**
- File: `main.py`
- Role: Stores simulation config, handles progress updates and results.
- Start with:
  ```bash
  source .venv/bin/activate
  python -m uvicorn main:app --reload --port 8001
  ```

### 3. **Client Executable**
- File: `local_client.py` → compiled to `.exe` using PyInstaller
- Role: Polls backend, runs simulation using `sim_runner.py`, sends progress/results.
- Build with:
  ```bash
  pyinstaller --onefile local_client.py
  ```
- Executable located at: `dist/local_client` or `dist/local_client.exe`

---

## 🧪 Running the System

### 1. **Start Backend**
```bash
run_backend  # or manually: source .venv/bin/activate && python -m uvicorn main:app --reload --port 8001
```

### 2. **Launch Streamlit App**
```bash
streamlit run streamlit_app_online.py
```

### 3. **Submit Config via UI**
- Modify parameters
- Click "Run Configuration"
- Wait for progress to appear in real time

### 4. **Start Client**
```bash
./dist/local_client
```

---

## 🔄 Simulation Flow

1. Streamlit sends config → `/run_config`
2. Backend stores job
3. Client polls `/get_job` and starts simulation
4. Client sends `/update_progress` (type="temp" and "perm")
5. Streamlit polls `/get_progress` and updates chart/table
6. Client sends `/submit_result`
7. Streamlit polls `/get_final_result` and ends session

---

## 🛡 Notes & Tips

- Use `make_json_serializable()` to sanitize NumPy values
- Use the "type" field in progress updates to distinguish between temporary and permanent updates
- Sort and deduplicate progress updates in the frontend for accurate plots

---

## 🧭 Future Improvements

- Session management and job history
- Export results (CSV, PDF)
- Deployment (Render, Railway)
- Config import/export support

---

Made with ❤️ by Furkan Ercan for the Arkastone project
