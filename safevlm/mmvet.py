"""MM-Vet v1 inputs and strict official prediction export (no online grading)."""
import json
from pathlib import Path


def metadata(source):
    path = Path(source) / 'mm-vet.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not data:
        raise ValueError(f'{path}: expected the official MM-Vet v1 metadata object')
    for key, row in data.items():
        if not isinstance(key, str) or not key.startswith('v1_') or not key[3:].isdigit():
            raise ValueError(f'{path}: invalid MM-Vet v1 ID: {key}')
        if not isinstance(row, dict) or not isinstance(row.get('question'), str) or not row['question'].strip():
            raise ValueError(f'{path}: missing question for {key}')
    return data


def records(source):
    source = Path(source).resolve()
    for key, row in metadata(source).items():
        image = source / 'images' / row['imagename']
        if not image.is_file():
            raise FileNotFoundError(f'{key}: {image}')
        yield {'id': key, 'question': row['question'], 'image': str(image),
               'capability': row.get('capability', [])}


def export_predictions(source, predictions, destination):
    """Keep official IDs, validate the complete set, and write ID -> answer strings."""
    expected = metadata(source)
    answers, modes = {}, set()
    predictions, destination = Path(predictions), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with predictions.open(encoding='utf-8') as reader:
        for number, line in enumerate(reader, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('expected a JSON object')
                key = row.get('id')
                if not isinstance(key, str) or key not in expected:
                    raise ValueError(f'unknown official ID: {key}')
                if key in answers:
                    raise ValueError(f'duplicate ID: {key}')
                if 'question' in row and row['question'] != expected[key]['question']:
                    raise ValueError(f'question does not match metadata for {key}')
                if not isinstance(row.get('answer'), str):
                    raise ValueError(f'answer must be a string for {key}')
                if 'mode' in row:
                    if row['mode'] not in ('baseline', 'sapt'):
                        raise ValueError(f'invalid mode for {key}')
                    modes.add(row['mode'])
                answers[key] = row['answer']
            except (ValueError, TypeError) as error:
                raise ValueError(f'{predictions}: line {number}: {error}') from error
    missing = set(expected) - set(answers)
    if missing:
        raise ValueError(f'incomplete MM-Vet results: missing {len(missing)} IDs; first: {sorted(missing)[:5]}')
    if len(modes) > 1:
        raise ValueError('mixed baseline/SAPT results cannot form a single submission')
    result = {key: answers[key] for key in expected}
    with destination.open('x', encoding='utf-8') as writer:
        json.dump(result, writer, ensure_ascii=False, indent=2)
        writer.write('\n')
    return len(result)


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["prepare","export"])
    parser.add_argument("--source",type=Path,required=True,help="official v1 directory containing mm-vet.json and images/")
    parser.add_argument("--input",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.action=="export":
        if args.input is None:
            parser.error("export requires --input")
        print("Exported",export_predictions(args.source,args.input,args.output),"official predictions")
    else:
        rows = list(records(args.source))
        if len(rows)!=218:
            parser.error("MM-Vet v1 requires all 218 questions")
        with args.output.open("x",encoding="utf-8") as writer:
            for row in rows:
                row["image"] = Path(os.path.relpath(row["image"],args.output.parent)).as_posix()
                writer.write(json.dumps(row,ensure_ascii=False)+"\n")
        print("Prepared 218 official inputs without reference answers")


if __name__=="__main__":
    main()
