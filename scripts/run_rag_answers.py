import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

QUESTIONS_FILE = BASE_DIR / "questions" / "questions.jsonl"
OUTPUT_FILE = BASE_DIR / "rag_answers" / "rag_distilled_raw.jsonl"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Import teacher RAG
# -----------------------------
from query_data import answer_question

# -----------------------------
# Main
# -----------------------------
def main():
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(f"Questions file not found: {QUESTIONS_FILE}")

    with QUESTIONS_FILE.open("r", encoding="utf-8") as qf, \
         OUTPUT_FILE.open("w", encoding="utf-8") as out_f:

        for line in qf:
            item = json.loads(line)

            question = item["question"]
            doc_id = item.get("doc_id")
            chunk_id = item.get("chunk_id")

            answer = answer_question(question)

            record = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "question": question,
                "answer": answer,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"RAG answers saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
