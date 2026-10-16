"""Command line single request and streaming JSONL inference."""

import argparse
import json
import sys
from pathlib import Path

from .config import MODEL_PATH, DATASET_REPO, DATASET_REVISION


def run_dataset(engine, dataset, destination):
    """Embedded Parquet images are consumed directly; result JSONL stores text metadata."""
    with Path(destination).open("x", encoding="utf-8") as writer:
        for index,source in enumerate(dataset):
            try:
                question = source.get("text",source.get("question"))
                result = engine.generate(question,source.get("image"))
                row = {k:v for k,v in source.items() if k!="image"}
                row["ori_question"] = source.get("ori_question",source.get("instruction",source.get("question",question)))
                row["question"] = question
                writer.write(json.dumps({**row,**result,"mode":engine.mode},ensure_ascii=False)+"\n")
                writer.flush()
                print(f"{engine.mode}: {index+1}/{len(dataset)}",file=sys.stderr,flush=True)
            except Exception as error:
                raise ValueError(f"dataset row {index}: {error}") from error


def run_jsonl(engine, source, destination):
    source, destination = Path(source), Path(destination)
    with source.open(encoding="utf-8") as reader, destination.open("x", encoding="utf-8") as writer:
        for line_number, line in enumerate(reader, 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("each line must be a JSON object")
                question = record.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("question must be a non-empty string")
                image = record.get("image")
                if image is not None:
                    if not isinstance(image, str) or not image:
                        raise ValueError("image must be a non-empty path string or null")
                    image = str(source.parent / image)
                result = engine.generate(question, image)
                writer.write(json.dumps({**record, **result, "mode": engine.mode}, ensure_ascii=False) + "\n")
                writer.flush()
            except Exception as error:
                raise ValueError(f"{source}: line {line_number}: {error}") from error


def main():
    parser = argparse.ArgumentParser(description="LLaVA baseline / SAPT inference")
    parser.add_argument("--mode", choices=["baseline", "sapt"], default="baseline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question")
    group.add_argument("--input", type=Path)
    group.add_argument("--dataset-config", help="e.g. figstep, mmsafety_sd_typo, spavl_harm")
    parser.add_argument("--image")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--device", default="cuda:0", help="e.g. cuda:5")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt-path")
    parser.add_argument("--classifier-path")
    parser.add_argument("--dataset-repo",default=DATASET_REPO)
    parser.add_argument("--dataset-revision",default=DATASET_REVISION)
    parser.add_argument("--data-dir",type=Path,help="local Hub snapshot containing data/<config>/*.parquet")
    parser.add_argument("--split",default="test")
    parser.add_argument("--limit",type=int,help="first N dataset rows; default: all")
    parser.add_argument("--ids-file",type=Path,help="ordered JSON list of dataset sample_id/id values")
    parser.add_argument("--mmvet-source",type=Path,help="after JSONL inference, also export official MM-Vet v1 JSON")
    args = parser.parse_args()
    if args.input and (args.output is None or args.image is not None):
        parser.error("--input requires --output; put image paths in the JSONL records")
    if args.question is not None and args.output is not None:
        parser.error("--output is only supported with --input")
    if args.dataset_config and (args.output is None or args.image is not None):
        parser.error("--dataset-config requires --output and uses embedded images")
    if (args.limit is not None or args.ids_file is not None) and not args.dataset_config:
        parser.error("--limit/--ids-file require --dataset-config")
    if args.limit is not None and args.limit<1:
        parser.error("--limit must be positive")
    if args.mmvet_source is not None and args.input is None:
        parser.error("--mmvet-source requires --input")
    if args.output and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.mmvet_source and args.output.with_suffix(".mmvet.json").exists():
        parser.error("official MM-Vet output already exists")
    try:
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.dataset_config:
            from .training_data import load_pool
            dataset = load_pool(args.dataset_config,args.split,args.data_dir,args.dataset_repo,args.dataset_revision)
            if args.ids_file:
                ids = json.loads(args.ids_file.read_text(encoding="utf-8"))
                if not isinstance(ids,list) or len(ids)!=len(set(ids)):
                    raise ValueError("ids-file must be a JSON list of unique IDs")
                field = "sample_id" if "sample_id" in dataset.column_names else "id"
                lookup = {key:i for i,key in enumerate(dataset[field])}
                dataset = dataset.select([lookup[key] for key in ids])
            if args.limit:
                dataset = dataset.select(range(min(args.limit,len(dataset))))
        from .inference import LlavaInference
        engine = LlavaInference(mode=args.mode, model_path=args.model_path, device=args.device,
                                max_new_tokens=args.max_new_tokens, prompt_path=args.prompt_path,
                                classifier_path=args.classifier_path)
        if args.input:
            run_jsonl(engine, args.input, args.output)
            if args.mmvet_source:
                from .mmvet import export_predictions
                export_predictions(args.mmvet_source,args.output,args.output.with_suffix(".mmvet.json"))
        elif args.dataset_config:
            run_dataset(engine,dataset,args.output)
        else:
            print(json.dumps({**engine.generate(args.question, args.image), "mode": engine.mode}, ensure_ascii=False))
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
