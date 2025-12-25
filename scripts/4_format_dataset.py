import json
from pathlib import Path

INPUT_FILE = Path("filtered/final_dataset.jsonl")
OUTPUT_FILE = Path("filtered/final_dataset_instruct.jsonl")

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with INPUT_FILE.open("r", encoding="utf-8") as fin, \
         OUTPUT_FILE.open("w", encoding="utf-8") as fout:

        for line in fin:
            item = json.loads(line)

            record = {
                "instruction": item["question"],
                "input": "",
                "output": item["answer"],
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Formatted {count} samples into instruct format")

if __name__ == "__main__":
    main()
