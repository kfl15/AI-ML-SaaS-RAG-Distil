import json
import re
from pathlib import Path

INPUT_FILE = Path("rag_answers/rag_distilled_raw.jsonl")
OUTPUT_FILE = Path("filtered/final_dataset.jsonl")

MIN_ANSWER_CHARS = 60

REJECT_PHRASES = [
    "not available in the provided context",
    "does not provide any information",
    "cannot answer this question",
]

NOISE_KEYWORDS = [
    "blank page",
    "glossary",
    "table of contents",
    "index",
    "pin",
]

def is_rejected_answer(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in REJECT_PHRASES)

def is_noisy_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in NOISE_KEYWORDS)

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0

    with INPUT_FILE.open("r", encoding="utf-8") as fin, \
         OUTPUT_FILE.open("w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            item = json.loads(line)

            q = item["question"]
            a = item["answer"]

            if is_rejected_answer(a):
                continue

            if is_noisy_question(q):
                continue

            if len(a.strip()) < MIN_ANSWER_CHARS:
                continue

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Filtered {kept} / {total} samples kept")

if __name__ == "__main__":
    main()
