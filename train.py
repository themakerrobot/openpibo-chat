"""Qwen3.5-0.8B를 대화 데이터로 미세조정합니다 (전체 미세조정, bf16, GPU 하나).

    python train.py --base Qwen/Qwen3.5-0.8B --data data/train-themaker.jsonl \
        --out runs/themaker-0.8b --lr 2e-5 --batch 2 --accum 16

모델의 대답에만 손실을 줍니다 (앞의 대화와 시스템 글은 읽기만). data/extra.txt(사람이 직접 쓴
대화)는 세 번씩 더 봅니다. 끊겼다가 다시 돌리면 끝까지 저장된 마지막 체크포인트부터 이어서
합니다. GPU가 안 보이면 학습하지 않습니다 (CPU로 몇십 배 느리게 돌지 않게).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(tok, messages, generation: bool, tools=None) -> str:
    return tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=generation, enable_thinking=False)


def tools_of(example: dict):
    """"tools"가 있는 대화는 기기 앱처럼 도구 목록(tools.TOOLS, 모든 기기 공통)을 함께 보여 줌."""
    from tools import TOOLS

    return TOOLS if example.get("tools") else None


def targets(tok, example: dict, max_len: int):
    """한 대화에서 모델이 말하는(또는 도구를 부르는) 차례마다 (input_ids, labels) 하나."""
    msgs, tools = example["messages"], tools_of(example)
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        prompt = render(tok, msgs[:i], True, tools)
        full = render(tok, msgs[:i + 1], False, tools)
        if not full.startswith(prompt):
            yield None                       # 템플릿이 예상과 다름: 세어서 알려 줍니다
            continue
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[-max_len:]
        labels = ([-100] * len(p_ids) + a_ids)[-max_len:]
        if any(x != -100 for x in labels):
            yield ids, labels


#: 직접 쓴 대화의 줄 머리 (한국어 extra.txt, 영어 extra-en.txt)
USER_TAGS, MODEL_TAGS = ("아이:", "Kid:"), ("더메이커:", "TheMaker:")


def read_extra(path, lang: str = "ko") -> list[dict]:
    """data/extra.txt: "아이:" 줄과 "더메이커:" 줄이 번갈아 나오는 대화, 빈 줄로 구분 (영어는
    data/extra-en.txt에 "Kid:", "TheMaker:"). #으로 시작하는 줄은 무시합니다."""
    from tools import SYSTEMS

    SYSTEM = SYSTEMS[lang]
    out, msgs = [], []

    def close():
        if any(m["role"] == "assistant" for m in msgs):
            # 절반은 도구 목록과 함께 (도구가 있어도 이런 말엔 대화로)
            out.append({"kind": "chat", "tools": len(out) % 2 == 0,
                        "messages": [{"role": "system", "content": SYSTEM}] + msgs[:]})
        msgs.clear()

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        if not line:
            close()
        elif line.startswith(USER_TAGS):
            msgs.append({"role": "user", "content": line.split(":", 1)[1].strip()})
        elif line.startswith(MODEL_TAGS):
            msgs.append({"role": "assistant", "content": line.split(":", 1)[1].strip()})
    close()
    return out


def build_rows(tok, data: str, max_len: int, extra: str | None = None,
               lang: str = "ko") -> list[dict]:
    """학습 데이터와 직접 쓴 대화(세 번씩)를 학습 차례 목록으로."""
    rows, broken = [], 0
    lines = Path(data).read_text().splitlines()
    extra = Path(extra) if extra else Path(data).with_name("extra.txt")
    if extra.exists():   # 직접 쓴 대화: 만든 데이터보다 세 번 더 보게
        own = [json.dumps(e, ensure_ascii=False) for e in read_extra(extra, lang)]
        print(f"직접 쓴 대화 {len(own)}개 추가 (세 번씩)", flush=True)
        lines += own * 3
    for line in lines:
        for t in targets(tok, json.loads(line), max_len):
            if t is None:
                broken += 1
            else:
                rows.append({"input_ids": t[0], "labels": t[1]})
    print(f"학습 차례 {len(rows)}개, 템플릿이 안 맞아 뺀 것 {broken}개", flush=True)
    if broken > 0.02 * (len(rows) + broken):
        raise SystemExit("대화 템플릿이 예상과 달라 학습할 수 없어요 (2% 넘게 안 맞음)")
    return rows


def last_checkpoint(out: Path) -> str | None:
    """끝까지 저장된 마지막 체크포인트: 저장하던 중에 꺼졌으면(재부팅 등) 그 앞의 것."""
    whole = [c for c in out.glob("checkpoint-*")
             if c.name.split("-")[-1].isdigit() and (c / "trainer_state.json").exists()
             and (c / "optimizer.pt").exists() and any(c.glob("*.safetensors"))]
    return str(max(whole, key=lambda c: int(c.name.split("-")[-1]))) if whole else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=16)
    p.add_argument("--max-len", type=int, default=1536)
    p.add_argument("--allow-cpu", action="store_true", help="GPU 없이 (시험용, 아주 느림)")
    p.add_argument("--extra", help="직접 쓴 대화 (기본: --data 옆의 extra.txt)")
    p.add_argument("--lang", choices=["ko", "en"], default="ko", help="직접 쓴 대화의 시스템 글")
    args = p.parse_args()

    import gc
    import math
    import shutil

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    out = Path(args.out)
    if (out / "final" / "config.json").exists():
        print(f"{out / 'final'}: 이미 끝남", flush=True)
        return
    if not torch.cuda.is_available() and not args.allow_cpu:
        # 드라이버 업데이트 뒤 재부팅 전후로 GPU가 안 보일 때 CPU로 몇십 배 느리게 돌지 않게
        raise SystemExit("GPU가 안 보여요: nvidia-smi를 확인하세요 (드라이버 업데이트 뒤면 "
                         "재부팅이나 드라이버 재설치가 필요할 수 있어요)")
    tok = AutoTokenizer.from_pretrained(args.base)
    rows = build_rows(tok, args.data, args.max_len, args.extra, args.lang)
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

    # transformers 4.56에서 torch_dtype이 dtype으로 바뀌었습니다
    new = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56)

    def load():
        return AutoModelForCausalLM.from_pretrained(
            args.base, **({"dtype": torch.bfloat16} if new else {"torch_dtype": torch.bfloat16}))

    model = load()
    batch, accum = args.batch, args.accum
    while True:
        steps = math.ceil(len(rows) / (batch * accum)) * args.epochs
        warmup = max(1, int(0.03 * steps))   # warmup_ratio는 5.x에서 없어져 걸음 수로
        targs = TrainingArguments(
            output_dir=str(out), num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=batch, gradient_accumulation_steps=accum,
            lr_scheduler_type="cosine", warmup_steps=warmup, weight_decay=0.0,
            bf16=torch.cuda.is_available(), gradient_checkpointing=True, logging_steps=20,
            save_steps=500, save_total_limit=2, report_to=[], dataloader_num_workers=2,
            seed=0, remove_unused_columns=False)
        trainer = Trainer(model=model, args=targs, train_dataset=rows, data_collator=collate)
        resume = last_checkpoint(out)
        if resume:
            print(f"{resume}에서 이어서", flush=True)
        print(f"배치 {batch} × 누적 {accum} (한 걸음에 {batch * accum}개)", flush=True)
        try:
            trainer.train(resume_from_checkpoint=resume)
            break
        except torch.OutOfMemoryError:
            if batch == 1:
                raise
            # 메모리가 모자라면 배치를 반으로, 누적을 두 배로: 한 걸음에 보는 양은 같게
            batch, accum = batch // 2, accum * 2
            print(f"GPU 메모리 부족: 배치 {batch}로 줄여 처음부터 다시", flush=True)
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            for ck in out.glob("checkpoint-*"):
                shutil.rmtree(ck)
            model = load()
    trainer.save_model(str(out / "final"))
    tok.save_pretrained(str(out / "final"))
    print(f"저장: {out / 'final'}", flush=True)


if __name__ == "__main__":
    main()
