import os
import json
from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from langchain_chroma import Chroma

import sys
from pathlib import Path

# Allow imports from project root
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
from get_embedding_function import get_embedding_function


# -----------------------------
# Configuration
# -----------------------------
# CHUNK_DIR = Path("chunked_docs")
CHROMA_PATH = "data_2"
OUTPUT_FILE = Path("questions/questions.jsonl")
MODEL_NAME = "mistral:latest"
TEMPERATURE = 0.2
MAX_QUESTIONS_PER_CHUNK = 3

PROMPT_TEMPLATE = """
You are generating training questions.

STRICT RULES:
- Use ONLY the SOURCE TEXT.
- Do NOT infer or interpret.
- Do NOT use external knowledge.
- Do NOT ask about meaning, symbolism, importance, or significance.
- Each question must be answerable using ONE sentence from the text.
- If a fact is not explicitly stated, do NOT ask about it.

TASK:
Generate up to {max_q} factual questions.

ALLOWED QUESTION TYPES:
- What happened
- Who did something
- What was described
- When something occurred
- Where something occurred
- How something was done (only if explicitly described)

FORBIDDEN QUESTION TYPES:
- Why is this important
- What does this represent
- What does this reveal
- Any opinion-based question

SELF-CHECK:
Before outputting a question, verify that its answer appears verbatim in the SOURCE TEXT.
If not, discard the question.

SOURCE TEXT:
<<<
{text}
>>>

OUTPUT FORMAT (JSON ONLY):
{{ "questions": [ "question 1", "question 2" ] }}
"""

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # llm = OllamaLLM(
    #     model=MODEL_NAME,
    #     temperature=TEMPERATURE,
    # )

    llm = OllamaLLM(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    base_url="http://172.28.128.1:11434"
    )


    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Load chunks directly from Chroma DB
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function()
    )

    data = db.get(include=["documents", "metadatas"])

    with OUTPUT_FILE.open("w", encoding="utf-8") as out_f:
        for text, meta in zip(data["documents"], data["metadatas"]):
            if not text or not text.strip():
                continue

            doc_id = meta.get("source", "unknown")
            chunk_id = meta.get("page", 0)

            formatted_prompt = prompt.format(
                text=text,
                max_q=MAX_QUESTIONS_PER_CHUNK,
            )

            response = llm.invoke(formatted_prompt)

            try:
                parsed = json.loads(response)
                questions = parsed.get("questions", [])
            except json.JSONDecodeError:
                continue

            for q in questions:
                record = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "question": q.strip(),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Questions generated at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
