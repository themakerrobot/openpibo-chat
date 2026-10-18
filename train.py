"""작은 모델을 파이보 대화로 미세조정합니다 (전체 미세조정, bf16, GPU 하나).

    python train.py --base Qwen/Qwen3-0.6B --data data/train.jsonl --out runs/qwen3-0.6b

파이보의 대답(그리고 도구 호출)에만 손실을 줍니다. 앞의 대화와 도구 목록은 읽기만
합니다. 끊겼다가 다시 돌리면 마지막 체크포인트부터 이어서 합니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prompts import tools_for


def render(tok, messages, tools, generation: bool) -> str:
    return tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=generation, enable_thinking=False)


def targets(tok, example: dict, max_len: int):
    """한 예시에서 파이보가 말하는 차례마다 (input_ids, labels) 하나."""
    msgs, tools = example["messages"], tools_for(example)
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prompt = render(tok, msgs[:i], tools, True)
        full = render(tok, msgs[:i + 1], tools, False)
        if not full.startswith(prompt):
            yield None                       # 템플릿이 예상과 다름: 세어서 알려 줍니다
            continue
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[-max_len:]
        labels = ([-100] * len(p_ids) + a_ids)[-max_len:]
        if any(x != -100 for x in labels):
            yield ids, labels


def read_extra(path) -> list[dict]:
    """data/extra.txt: "아이:" 줄과 "파이보:" 줄이 번갈아 나오는 대화, 빈 줄로 구분.
    #으로 시작하는 줄은 무시합니다."""
    from tools import SYSTEM

    out, msgs = [], []

    def close():
        if any(m["role"] == "assistant" for m in msgs):
            out.append({"kind": "chat", "tools": len(out) % 2 == 0,
                        "messages": [{"role": "system", "content": SYSTEM}] + msgs[:]})
        msgs.clear()

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if not line:
            close()
        elif line.startswith("아이:"):
            msgs.append({"role": "user", "content": line[3:].strip()})
        elif line.startswith("파이보:"):
            msgs.append({"role": "assistant", "content": line[4:].strip()})
    close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--max-len", type=int, default=1536)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    out = Path(args.out)
    if (out / "final" / "config.json").exists():
        print(f"{out / 'final'}: 이미 끝남", flush=True)
        return
    tok = AutoTokenizer.from_pretrained(args.base)
    rows, broken = [], 0
    lines = Path(args.data).read_text().splitlines()
    extra = Path(args.data).with_name("extra.txt")
    if extra.exists():   # 직접 쓴 예시: 만든 데이터보다 세 번 더 보게
        own = [json.dumps(e, ensure_ascii=False) for e in read_extra(extra)]
        print(f"직접 쓴 예시 {len(own)}개 추가 (세 번씩)", flush=True)
        lines += own * 3
    for line in lines:
        for t in targets(tok, json.loads(line), args.max_len):
            if t is None:
                broken += 1
            else:
                rows.append({"input_ids": t[0], "labels": t[1]})
    print(f"학습 차례 {len(rows)}개, 템플릿이 안 맞아 뺀 것 {broken}개", flush=True)
    if broken > 0.02 * (len(rows) + broken):
        raise SystemExit("대화 템플릿이 예상과 달라 학습할 수 없어요 (2% 넘게 안 맞음)")

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad)
        labels = torch.full((len(batch), n), -100)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    targs = TrainingArguments(
        output_dir=str(out), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
        lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0, bf16=True,
        gradient_checkpointing=True, logging_steps=20, save_steps=500, save_total_limit=2,
        report_to=[], group_by_length=False, dataloader_num_workers=2, seed=0,
        remove_unused_columns=False)
    trainer = Trainer(model=model, args=targs, train_dataset=rows, data_collator=collate)
    resume = any(out.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(str(out / "final"))
    tok.save_pretrained(str(out / "final"))
    print(f"저장: {out / 'final'}", flush=True)


if __name__ == "__main__":
    main()
