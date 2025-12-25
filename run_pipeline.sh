#!/usr/bin/env bash
set -e

echo "========================================"
echo "AI-ML-SaaS-RAG-Distil | Full Pipeline Run"
echo "========================================"

# -----------------------------
# Sanity checks
# -----------------------------
if [ -z "$VIRTUAL_ENV" ]; then
  echo "ERROR: No active virtual environment detected."
  echo "Please activate your Python venv before running this script."
  exit 1
fi

if [ ! -f "requirements.txt" ]; then
  echo "ERROR: requirements.txt not found"
  exit 1
fi

if [ ! -d "raw_docs" ]; then
  echo "ERROR: raw_docs folder missing"
  exit 1
fi

# -----------------------------
# Install dependencies
# -----------------------------
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# -----------------------------
# Run pipeline
# -----------------------------
echo "Step 1: Populate vector database"
python populate_database.py --reset

echo "Step 2: Generate questions"
python scripts/1_generate_questions.py

echo "Step 3: Generate RAG answers"
python scripts/2_run_rag_answers.py

echo "Step 4: Filter dataset"
python scripts/3_filter_samples.py

echo "Step 5: Format dataset"
python scripts/4_format_dataset.py

echo "========================================"
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "Output:"
echo "filtered/final_dataset_instruct.jsonl"
echo "========================================"
