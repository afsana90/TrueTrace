#!/bin/bash
# TrueTrace v8 Startup Script
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
cd "$(dirname "$0")/backend"
echo "Starting TrueTrace v8 (Claude Vision + EasyOCR + PaddleOCR)..."
python3 main.py
