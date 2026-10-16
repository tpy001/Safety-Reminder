"""MM-SafetyBench inputs, shared by JSONL inference and Parquet export."""
import json
from pathlib import Path

VARIANTS = {"sd": "SD", "typo": "TYPO", "sd_typo": "SD_TYPO"}


def records(source, variant="sd_typo"):
    source = Path(source)
    if variant not in VARIANTS:
        raise ValueError(f"Unknown MMSafety variant: {variant}")
    files = sorted((source / "processed_questions").glob("*.json"))
    if not files:
        raise FileNotFoundError(source / "processed_questions")
    field = "Rephrased Question(SD)" if variant == "sd" else "Rephrased Question"
    for file in files:
        category = file.stem
        questions = json.loads(file.read_text(encoding="utf-8"))
        for question_id, row in questions.items():
            image = source / "imgs" / category / VARIANTS[variant] / f"{question_id}.jpg"
            if not image.is_file():
                raise FileNotFoundError(image)
            text, original = row[field], row["Question"]
            if not isinstance(text, str) or not text.strip() or not isinstance(original, str):
                raise ValueError(f"Invalid question: {file}:{question_id}")
            yield {"id": f"{category}/{question_id}", "question_id": str(question_id),
                   "question": text, "ori_question": original, "image": str(image.resolve()),
                   "category": category, "variant": variant}


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, default='sd_typo')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x', encoding='utf-8') as writer:
        for row in records(args.source, args.variant):
            row['image'] = Path(os.path.relpath(row['image'],args.output.parent)).as_posix()
            writer.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
