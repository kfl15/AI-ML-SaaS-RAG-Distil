# AI-ML-SaaS-RAG-Distil

A fully local, reproducible **RAG-based dataset distillation pipeline** designed to generate
high-quality instruction datasets for **LoRA / QLoRA fine-tuning**.

This project uses an existing RAG system as a **teacher** to produce grounded question–answer
pairs from documents, with strict filtering to suppress hallucinations.

---

## ✅ Key Features

- Fully **local execution** (WSL + Windows Ollama)
- No external APIs
- Deterministic, step-by-step pipeline
- Model-agnostic (Gemma, Mistral tested)
- Produces **instruction-ready JSONL datasets**
- Git-clean, reproducible workflow

---

## 📁 Project Structure

.
├── raw_docs/ # Input PDFs / TXTs (not committed)
├── data_2/ # Chroma vector DB (local)
├── scripts/
│ ├── 1_generate_questions.py
│ ├── 2_run_rag_answers.py
│ ├── 3_filter_samples.py
│ └── 4_format_dataset.py
├── filtered/
│ ├── final_dataset.jsonl
│ └── final_dataset_instruct.jsonl
├── populate_database.py
├── query_data.py
├── get_embedding_function.py
├── run_pipeline.sh # One-command runner
└── README.md


## ⚙️ Requirements

- Windows 10/11
- WSL (Ubuntu)
- Python 3.11
- Ollama (Windows)
- Models:
  - `nomic-embed-text`
  - `gemma:2b` (or `mistral`)

---

## 🚀 One-Command Run

After cloning the repo,ensure placing documents in `raw_docs/`as pdfs, txt. 


## 📦 Python Dependencies

All required Python packages are listed in `requirements.txt`.

## ▶️ How to Run

1. Create and activate your own Python virtual environment
2. Ensure Ollama is running
3. Place documents in `raw_docs/`
4. Run:

```bash
./run_pipeline.sh