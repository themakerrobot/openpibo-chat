"""학습하지 않은 바탕 모델 GGUF를 받아 견줌용으로 둡니다 (코딩 도우미: 학습이 도움이 되는지, 더 큰 모델은 어떤지).

    python fetch_base.py --repo unsloth/Qwen3.5-4B-GGUF --file Qwen3.5-4B-Q4_K_M.gguf --out baseline/qwen3.5-4b-Q4_K_M.gguf

받은 뒤 대화 템플릿을 늘 생각 없이로 고칩니다 (export.retemplate): 원본 Qwen3.5는 대답 전에 길게 생각해서,
채점의 최대 토큰 안에 대답이 안 나옵니다.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--file", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"{out}: 이미 있음", flush=True)
        return
    from huggingface_hub import hf_hub_download

    from export import retemplate

    out.parent.mkdir(parents=True, exist_ok=True)
    got = Path(hf_hub_download(args.repo, args.file, local_dir=out.parent / ".hf"))
    tmp = out.with_suffix(".part")
    shutil.copyfile(got, tmp)
    retemplate(tmp)
    tmp.replace(out)
    shutil.rmtree(out.parent / ".hf", ignore_errors=True)
    print(f"{out}: {out.stat().st_size / 2**20:.0f}MB (생각 없이 템플릿)", flush=True)


if __name__ == "__main__":
    main()
