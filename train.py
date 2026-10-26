"""작은 모델을 파이보 대화로 미세조정합니다 (전체 미세조정, bf16, GPU 하나).

    python train.py --base Qwen/Qwen3-0.6B --data data/train.jsonl --out runs/qwen3-0.6b
    python train.py ... --kd data/kd      # 증류: 선생 모델의 토큰 확률도 따라 배움

파이보의 대답(그리고 도구 호출)에만 손실을 줍니다. 앞의 대화와 도구 목록은 읽기만
합니다. 끊겼다가 다시 돌리면 마지막 체크포인트부터 이어서 합니다.

--kd를 주면 distill.py가 미리 구해 둔 선생 모델의 확률(토큰마다 상위 K개)을 읽어,
토큰마다 kd-weight × KL(선생‖학생) + (1 − kd-weight) × 정답 CE로 배웁니다.
"""

from __future__ import annotations

import argparse
import hashlib
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


def build_rows(tok, data: str, max_len: int) -> list[dict]:
    """학습 데이터와 직접 쓴 대화(세 번씩)를 학습 차례 목록으로. distill.py도 같은 것을 씁니다."""
    rows, broken = [], 0
    lines = Path(data).read_text().splitlines()
    extra = Path(data).with_name("extra.txt")
    if extra.exists():   # 직접 쓴 예시: 만든 데이터보다 세 번 더 보게
        own = [json.dumps(e, ensure_ascii=False) for e in read_extra(extra)]
        print(f"직접 쓴 예시 {len(own)}개 추가 (세 번씩)", flush=True)
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


def kd_key(ids) -> str:
    """학습 차례 하나의 이름: 토큰 그대로의 해시 (데이터 순서가 바뀌어도 맞게)."""
    import numpy as np

    return hashlib.sha1(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()[:20]


def load_kd(path) -> dict:
    """distill.py가 쓴 data/kd/part-*.npz → {차례 이름: (상위 K개 토큰 [n,K], 로그확률 [n,K])}.
    n은 그 차례에서 맞힐 토큰 수(labels가 -100이 아닌 자리)와 같습니다."""
    import numpy as np
    import torch

    store = {}
    for part in sorted(Path(path).glob("part-*.npz")):
        d = np.load(part)
        ids, lp = torch.from_numpy(d["ids"]), torch.from_numpy(d["lp"])
        start = 0
        for key, n in zip(d["keys"].tolist(), d["counts"].tolist(), strict=True):
            store[key] = (ids[start:start + n], lp[start:start + n])
            start += n
    return store


def kd_loss(logits, labels, kd_ids, kd_lp, weight: float):
    """(맞힐 토큰마다 weight × KL(선생‖학생) + (1 − weight) × CE 의 합, 맞힐 토큰 수).

    logits [B,L,V], labels [B,L], kd_ids·kd_lp [B,L,K]는 labels와 같은 자리에 선생이
    그 토큰을 두고 매긴 상위 K개 (선생이 없는 자리는 kd_ids가 -1). 선생의 K개는
    합이 1이 되게 다시 나누고, 학생은 전체 어휘의 확률을 씁니다."""
    import torch

    labels, kd_ids, kd_lp = labels[:, 1:], kd_ids[:, 1:], kd_lp[:, 1:]
    mask = labels != -100
    logp = torch.log_softmax(logits[:, :-1][mask].float(), dim=-1)
    ce = -logp.gather(1, labels[mask][:, None]).squeeze(1)
    ids, lp = kd_ids[mask], kd_lp[mask].float()
    have = ids[:, 0] >= 0
    p = torch.softmax(lp, dim=-1)
    kl = (p * (torch.log(p.clamp_min(1e-12)) - logp.gather(1, ids.clamp_min(0)))).sum(-1)
    tok = torch.where(have, weight * kl + (1 - weight) * ce, ce)
    return tok.sum(), mask.sum()


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
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--max-len", type=int, default=1536)
    p.add_argument("--kd", help="distill.py가 만든 선생 확률 폴더 (증류)")
    p.add_argument("--kd-weight", type=float, default=0.5)
    p.add_argument("--allow-cpu", action="store_true", help="GPU 없이 (시험용, 아주 느림)")
    args = p.parse_args()

    import torch
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
    rows = build_rows(tok, args.data, args.max_len)

    kd_k = 0
    if args.kd:
        store = load_kd(args.kd)
        if not store:
            raise SystemExit(f"{args.kd}에 선생 확률이 없어요 (distill.py를 먼저)")
        kd_k = next(iter(store.values()))[0].shape[1]
        hit = 0
        for r in rows:
            got = store.get(kd_key(r["input_ids"]))
            n = sum(x != -100 for x in r["labels"])
            r["kd"] = got if got is not None and len(got[0]) == n else None
            hit += r["kd"] is not None
        print(f"증류: 선생 확률이 있는 차례 {hit}/{len(rows)}, 상위 {kd_k}개, "
              f"가중치 {args.kd_weight}", flush=True)
        if hit < 0.9 * len(rows):
            raise SystemExit("선생 확률이 있는 차례가 90%도 안 돼요 (데이터가 바뀌었으면 "
                             "distill.py를 다시)")

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
        out = {"input_ids": ids, "labels": labels, "attention_mask": mask}
        if kd_k:
            kd_ids = torch.full((len(batch), n, kd_k), -1, dtype=torch.long)
            kd_lp = torch.zeros((len(batch), n, kd_k), dtype=torch.float16)
            for i, b in enumerate(batch):
                if b.get("kd") is not None:
                    pos = [j for j, x in enumerate(b["labels"]) if x != -100]
                    kd_ids[i, pos] = b["kd"][0].long()
                    kd_lp[i, pos] = b["kd"][1]
            out |= {"kd_ids": kd_ids, "kd_lp": kd_lp}
        return out

    class KDTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            kd_ids, kd_lp, labels = inputs.pop("kd_ids"), inputs.pop("kd_lp"), inputs.pop("labels")
            outputs = model(**inputs)
            total, n = kd_loss(outputs.logits, labels, kd_ids, kd_lp, args.kd_weight)
            # num_items_in_batch: 누적한 묶음 전체에서 맞힐 토큰 수 (Trainer가 셉니다)
            loss = total / (num_items_in_batch if num_items_in_batch is not None
                            else n.clamp_min(1))
            return (loss, outputs) if return_outputs else loss

    import math

    import transformers

    # transformers 4.56에서 torch_dtype이 dtype으로 바뀌었습니다
    new = tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, **({"dtype": torch.bfloat16} if new else {"torch_dtype": torch.bfloat16}))
    import gc
    import shutil

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
        trainer = (KDTrainer if kd_k else Trainer)(model=model, args=targs, train_dataset=rows,
                                                   data_collator=collate)
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
            model = AutoModelForCausalLM.from_pretrained(
                args.base, **({"dtype": torch.bfloat16} if new else {"torch_dtype": torch.bfloat16}))
    trainer.save_model(str(out / "final"))
    tok.save_pretrained(str(out / "final"))
    print(f"저장: {out / 'final'}", flush=True)


if __name__ == "__main__":
    main()
