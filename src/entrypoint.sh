#!/bin/bash
# 스크립트가 실패할 경우 즉시 종료하도록 설정
set -e

python3 -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

exec uvicorn main:app --host 0.0.0.0 --port 8000