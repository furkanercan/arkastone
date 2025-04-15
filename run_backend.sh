#!/bin/bash
source .venv/bin/activate
python -m uvicorn main:app --reload --port 8001