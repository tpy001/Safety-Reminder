"""Train four SAPT embeddings with a frozen LLaVA and frozen PCA classifier."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from .config import MODEL_PATH, RESOURCES, DATASET_REPO, DATASET_REVISION
from .inference import LlavaInference
from .sapt import SafetyReminder
from .training_data import balanced_data

INIT_TEXT = "Refuse if partial output is unsafe; else, continue helpfully."


def split_answer(answer, rng):
    """The old character/space prefix sampling, including its 1/3 empty prefix."""
    if rng.random() < 1/3:
        return "",answer.strip()
    for _ in range(10):
        length = len(answer) if len(answer)<20 else rng.randint(20,len(answer))
        space = answer[:length].rfind(" ")
        if space != -1:
            return answer[:space].strip(),answer[space:].strip()
    return "",answer.strip()


def training_text(engine,row,rng):
    benign = row["is_benign"]
    source = row["chosen"] if benign else row["rejected"]
    prefix,suffix = split_answer(source,rng)
    target = suffix if benign else row["chosen"].strip()
    if not target:
        raise ValueError(f"Empty training target: {row['sample_id']}")
    prompt = engine._format(row["question"],row.get("image") is not None)
    prompt = (prompt+" "+prefix.strip()).rstrip()+" "+" ".join(["<soft_prompt>"]*4)
    return prompt.strip()+" "+target.strip()


def training_labels(ids,token_id,attention_mask):
    positions = torch.where(ids[0]==token_id)[0]
    if len(positions)!=4:
        raise ValueError("Training input must contain exactly four soft prompt tokens")
    labels = ids.clone()
    labels[:,:positions[-1]+1] = -100
    labels[attention_mask==0] = -100
    if not (labels!=-100).any():
        raise ValueError("Training input has no target tokens")
    return labels,positions[-1]


def train(args):
    if args.epochs<1 or args.grad_accum<1 or args.lr<=0 or (args.max_steps is not None and args.max_steps<1):
        raise ValueError("epochs, grad-accum, learning rate and max-steps must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    data = balanced_data(args.samples_per_pool,args.seed,args.safety_pool,args.data_dir,
                         args.dataset_repo,args.dataset_revision)
    args.output_dir.mkdir(parents=True)
    selection = [{"sample_id":key,"is_benign":bool(benign)}
                 for key,benign in zip(data["sample_id"],data["is_benign"])]
    # Only select metadata here; image contents are loaded lazily by the dataset.
    (args.output_dir/"selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config["counts"] = dict(Counter("benign" if r["is_benign"] else "harmful" for r in selection))
    config["loss"] = "token-mean LM + 0.05 * example-mean BCE(PCA classifier probability, is_benign)"
    config["classifier_sha256"] = sha256(args.classifier_path.read_bytes()).hexdigest()
    (args.output_dir/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    engine = LlavaInference(mode="baseline",model_path=args.model_path,device=args.device)
    init_ids = engine.tokenizer(INIT_TEXT,add_special_tokens=False,return_tensors="pt")["input_ids"][0]
    weight = engine.model.get_input_embeddings()(init_ids.to(engine.device)).detach().float()[:4]
    if len(weight)<4:
        weight = weight.repeat((4+len(weight)-1)//len(weight),1)[:4]
    # A temporary initialized checkpoint uses the same validation and embedding hook as inference.
    init_path = args.output_dir/"initial_prompt.pth"
    torch.save({"embedding.weight":weight.cpu()},init_path)
    detector = SafetyReminder(engine.model,engine.tokenizer,args.init_prompt or init_path,args.classifier_path)
    detector.weight = torch.nn.Parameter(detector.weight)
    optimizer = torch.optim.AdamW([detector.weight],lr=args.lr,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01)
    rng,step = random.Random(args.seed),0
    loader = torch.utils.data.DataLoader(data,batch_size=args.grad_accum,shuffle=True,collate_fn=lambda rows:rows,
                                         generator=torch.Generator().manual_seed(args.seed))
    with (args.output_dir/"loss.jsonl").open("x",encoding="utf-8") as log:
        for epoch in range(args.epochs):
            for rows in loader:
                optimizer.zero_grad(set_to_none=True)
                prepared = []
                for row in rows:
                    text = training_text(engine,row,rng)
                    inputs = engine._inputs(text,row.get("image"),reserve_generation=False)
                    labels,position = training_labels(inputs["input_ids"],detector.token_id,inputs["attention_mask"])
                    prepared.append((row,inputs,labels,position,int((labels!=-100).sum())))
                total_tokens = sum(item[-1] for item in prepared)
                totals = {"lm_loss":0.,"cls_loss":0.}
                # One GPU example at a time, while retaining the old batch's token-mean LM loss.
                for row,inputs,labels,position,n_tokens in prepared:
                    outputs = engine.model(**inputs,labels=labels,use_cache=False,output_hidden_states=True,return_dict=True)
                    # Paper Eq. (10): supervise the final soft token through a frozen safety head.
                    # Do not detach h: gradients must flow through the frozen backbone to the prompt.
                    hidden = outputs.hidden_states[-1][:,position].float()
                    classifier = detector.classifier
                    projected = (hidden-classifier["pca_mean"]) @ classifier["pca_components"][:4].T
                    probability = torch.sigmoid(projected @ classifier["clf_weights"].T+classifier["clf_bias"])
                    cls_loss = F.binary_cross_entropy(probability,torch.full_like(probability,float(row["is_benign"])))
                    # Eqs. (11)-(13): target-token LM loss + 0.05 * hidden-state BCE.
                    # These weights reproduce token-mean LM and example-mean BCE over this group.
                    loss = outputs.loss*n_tokens/total_tokens+0.05*cls_loss/len(rows)
                    if not torch.isfinite(loss):
                        raise ValueError(f"Nonfinite loss: {row['sample_id']}")
                    loss.backward()
                    totals['lm_loss'] += outputs.loss.item()*n_tokens/total_tokens
                    totals['cls_loss'] += cls_loss.item()/len(rows)
                    del outputs,hidden,loss,cls_loss
                grad = detector.weight.grad
                if grad is None or not torch.isfinite(grad).all() or grad.abs().sum()==0:
                    raise ValueError("Soft prompt gradient is missing, zero or nonfinite")
                optimizer.step(); step += 1
                record = {"epoch":epoch+1,"step":step,**totals,
                          "total_loss":totals['lm_loss']+.05*totals['cls_loss']}
                log.write(json.dumps(record)+"\n"); log.flush(); print(json.dumps(record),flush=True)
                if args.max_steps is not None and step>=args.max_steps:
                    break
            torch.save({"embedding.weight":detector.weight.detach().cpu()},args.output_dir/f"prompt_epoch_{epoch+1}.pth")
            if args.max_steps is not None and step>=args.max_steps:
                break
    if any(p.grad is not None for p in engine.model.parameters()):
        raise RuntimeError("Frozen base model received gradients")
    torch.save({"embedding.weight":detector.weight.detach().cpu()},args.output_dir/"prompt.pth")
    detector.hook.remove()
    print(f"Saved {args.output_dir/'prompt.pth'}; only 4 x {detector.weight.shape[1]} parameters trained",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path",default=MODEL_PATH)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--data-dir",type=Path)
    parser.add_argument("--dataset-repo",default=DATASET_REPO)
    parser.add_argument("--dataset-revision",default=DATASET_REVISION)
    parser.add_argument("--safety-pool",choices=["legacy","full"],default="legacy")
    parser.add_argument("--samples-per-pool",type=int,default=256)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--epochs",type=int,default=20)
    parser.add_argument("--lr",type=float,default=1e-4)
    parser.add_argument("--grad-accum",type=int,default=4)
    parser.add_argument("--max-steps",type=int,help="optional optimizer-step limit for a smoke test")
    parser.add_argument("--init-prompt",type=Path,help="optional existing prompt for continued training")
    parser.add_argument("--classifier-path",type=Path,default=RESOURCES/"llava_pca_cls.npz")
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    try:
        train(args)
    except Exception as error:
        parser.exit(1,f"error: {error}\n")


if __name__=="__main__":
    main()
