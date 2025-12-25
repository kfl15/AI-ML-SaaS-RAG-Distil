import argparse
from langchain.prompts import ChatPromptTemplate


from get_embedding_function import get_embedding_function

# from langchain.vectorstores.chroma import Chroma
from langchain_chroma import Chroma

# from langchain_community.llms.ollama import Ollama
from langchain_ollama import OllamaLLM as Ollama



CHROMA_PATH = "data_2"

# PROMPT_TEMPLATE = """
# Answer the question based on the above context:
#
# {context}
#
# ---
#
# Answer the question based on the above context: don't answer in more than 12 to 15 points. always answer specifically.
# while answering, try to answer fully.  Write in points. dont
# {question}
# """


# PROMPT_TEMPLATE = """
# You are an assistant answering questions using ONLY the provided context.
#
# CONTEXT:
# {context}
#
# QUESTION:
# {question}
#
# INSTRUCTIONS:
# - Use only the information in the context to answer.
# - Do not invent or assume facts.
# - Quote or summarize key ideas directly from the text.
# - Write your answer in concise bullet points.
# - If examples are present, include them.
# - If the answer cannot be found, say: "The answer is not available in the provided context."
# - Do NOT give general motivational or abstract advice.
# """

# PROMPT_TEMPLATE = """
# You are a helpful assistant. Use ONLY the information provided in the context below to answer the question.
#
# If the answer cannot be found in the context, say "The answer is not available in the provided context."
#
# ---
# CONTEXT:
# {context}
# ---
#
# QUESTION:
# {question}
#
# INSTRUCTIONS:
# - Base your answer strictly on the context above. Do not use external knowledge.
# - Write your answer as clear, concise bullet points.
# - Provide a maximum of 12 to 15 bullet points.
# - If possible, organize your points logically and completely answer the question.
# - Do not include irrelevant details or speculation.
#
# Now, write the final answer below.
# """

PROMPT_TEMPLATE = """
You are a precise and knowledgeable assistant. Your purpose is to answer questions strictly based on the context provided below.

If the answer is not explicitly or implicitly present in the context, say exactly:
"The answer is not available in the provided context."

---
CONTEXT:
{context}
---

QUESTION:
{question}

INSTRUCTIONS:
1. Use ONLY the information from the provided context. Never rely on external knowledge or assumptions.
2. If multiple chunks of context are provided, integrate them logically — avoid repeating the same idea.
3. Write your answer in **clear, concise bullet points**.
4. Keep your answer well-structured and factual.
5. Include up to **12–15 bullet points maximum**.
6. Whenever possible, include examples, definitions, or comparisons that appear in the context.
7. If the context contains contradictory information, mention both sides briefly.
8. If appropriate, cite which context segment supports each point using a simple reference like **(Source 1)**, **(Source 2)**, etc.
9. Do NOT add generic motivational or opinionated statements.
10. Do NOT rephrase the question — directly answer it.

OUTPUT FORMAT:
- Begin directly with the bullet-pointed answer.
- Do not include any introduction. But must include a short closing summary.
"""




def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="mistral:latest")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()

import os
import json
from pathlib import Path
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate

# -----------------------------
# Configuration
# -----------------------------
CHUNK_DIR = Path("chunked_docs")
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

# -----------------------------
# Main logic
# -----------------------------
def main():
    if not CHUNK_DIR.exists():
        raise FileNotFoundError(f"Chunk directory not found: {CHUNK_DIR}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    with OUTPUT_FILE.open("w", encoding="utf-8") as out_f:
        for chunk_file in sorted(CHUNK_DIR.glob("*.txt")):
            doc_id = chunk_file.stem
            chunk_id = 0

            text = chunk_file.read_text(encoding="utf-8").strip()
            if not text:
                continue

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

def answer_question(question: str) -> str:
    return query_rag(question)

