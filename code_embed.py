"""코딩 도우미 찾기 실험: 낱말 점수(BM25, code_rag)에 뜻 점수(임베딩)를 섞으면 정답을 더 잘 찾는지 봅니다.

    python code_embed.py --eval data/heldout-code.jsonl          # 방법별 1등·상위 3·5개 비율 (숫자만)
    python code_embed.py --ask 파이보 "불 좀 켜고 싶어"            # 섞은 찾기로 찾아 보기

임베딩 모델은 multilingual-e5-small을 브라우저용으로 양자화한 ONNX(약 118MB)입니다. 웹 실행기(transformers.js)도
같은 파일을 쓰게 하려고, PC에서도 torch 대신 같은 ONNX를 onnxruntime으로 돌립니다 (PC 점수 = 브라우저 결과).
처음 한 번 models/에 받습니다. 필요한 것: pip install onnxruntime tokenizers (numpy, huggingface_hub는 이미 있음).

설명마다 두 가지 뜻 점수를 봅니다: 설명 글(이름·카테고리·설명)과 질문의 닮음, 그 설명을 찾던 학습 질문들
(code_rag.asked_from, 채점용과 겹치지 않음) 중 가장 닮은 것. 섞는 법은 순위 섞기(RRF)와 점수 섞기(가중 평균).
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

import code_rag

REPO = "Xenova/multilingual-e5-small"
ONNX = "onnx/model_quantized.onnx"
MODELS = Path("models/e5-small")
#: 순위 섞기 상수 (흔히 쓰는 60)
RRF_K = 60
#: 쓸 때 섞는 비율: 낱말 점수(가장 높은 것으로 나눔) × WEIGHT + 뜻 점수(기기 안에서 0~1로) × (1 - WEIGHT).
#: 채점 질문에서 0.3·0.5·0.7 중 가장 좋았음 (상위 3개 0.813, 같은 기능 0.872)
WEIGHT = 0.5


def usable() -> str | None:
    """섞은 찾기를 못 쓰는 까닭 (쓸 수 있으면 None)."""
    import importlib.util

    for name in ("onnxruntime", "tokenizers", "huggingface_hub"):
        if importlib.util.find_spec(name) is None:
            return f"{name} 없음 (pip install onnxruntime tokenizers)"
    return None


def as32(a) -> np.ndarray:
    """float32로 한 번 깎은 뒤 float64로: 브라우저(float32 벡터를 JS 숫자로 계산)와 같은 값."""
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def mix(idx: list[int], bm: dict[int, float], emb: dict[int, float], w: float = WEIGHT) -> list[int]:
    """점수 섞기 순위 (웹 실행기 codeSearch와 같은 식)."""
    top_b = max(bm.values(), default=0.0) or 1.0
    lo, hi = min(emb.values(), default=0.0), max(emb.values(), default=0.0)
    score = {i: w * bm[i] / top_b + (1 - w) * (emb[i] - lo) / ((hi - lo) or 1) for i in idx}
    return sorted(idx, key=lambda i: (-score[i], i))


class Embedder:
    """e5: 질문은 "query: ", 설명은 "passage: "를 앞에 붙이고, 마지막 층을 평균 내어 길이 1로."""

    def __init__(self, repo: str = REPO, onnx: str = ONNX, root: Path = MODELS):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        tok = Path(hf_hub_download(repo, "tokenizer.json", local_dir=root))
        model = Path(hf_hub_download(repo, onnx, local_dir=root))
        self.tok = Tokenizer.from_file(str(tok))
        self.tok.enable_truncation(512)
        self.tok.enable_padding()
        self.sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def __call__(self, texts: list[str], batch: int = 32) -> np.ndarray:
        out = []
        for k in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[k:k + batch])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feed)[0]
            vec = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
            out.append(vec / np.linalg.norm(vec, axis=1, keepdims=True))
        return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)


def passage(u: dict) -> str:
    """설명 글: 카드의 이름·카테고리·설명 (코드는 뺌: 뜻 비교에 잡음)."""
    c = code_rag.card(u)
    return f"passage: {c['title']} {c['tag']} {c['text']}"[:1000]


class Hybrid(code_rag.Index):
    """낱말 점수와 뜻 점수를 한 기기의 설명 안에서 섞어 찾습니다 (code_rag.Index 대신 그대로 씀).
    search는 점수 섞기(WEIGHT), ranks는 실험용으로 방법별 순위."""

    kind = "낱말+뜻 (e5-small)"

    def __init__(self, units: list[dict], asked: dict[str, list[str]], embed):
        super().__init__(units, asked)
        self.embed = embed
        self.P = as32(embed([passage(u) for u in units]))
        self.asked_rows: list[tuple[int, str]] = [(i, q) for i, u in enumerate(units)
                                                   for q in asked.get(u["id"], [])[:code_rag.MAX_ASKED]]
        self.Q = as32(embed([f"query: {q}" for _, q in self.asked_rows])) if self.asked_rows else np.zeros((0, 0))

    def qvec(self, query: str) -> np.ndarray:
        return as32(self.embed([f"query: {query}"])[0])

    def parts(self, query: str, device: str, q: np.ndarray):
        """설명 번호들, 낱말 점수, 뜻 점수(설명 글), 뜻 점수(설명 글과 학습 질문 중 큰 것)."""
        idx = self.by_device.get(device, [])
        emb_p = {i: float(self.P[i] @ q) for i in idx}
        emb_a = dict.fromkeys(idx, -1.0)
        if len(self.asked_rows):
            for (i, _), s in zip(self.asked_rows, self.Q @ q):
                if i in emb_a and s > emb_a[i]:
                    emb_a[i] = float(s)
        emb = {i: max(emb_p[i], emb_a[i]) for i in idx}
        bm = dict.fromkeys(idx, 0.0)
        for s, i in self.scored(query, device):
            bm[i] = s
        return idx, bm, emb_p, emb

    def search_with(self, query: str, device: str, q: np.ndarray, k: int = code_rag.TOP_K) -> list[dict]:
        idx, bm, _, emb = self.parts(query, device, q)
        return [self.units[i] for i in mix(idx, bm, emb)[:k]]

    def search(self, query: str, device: str, k: int = code_rag.TOP_K) -> list[dict]:
        return self.search_with(query, device, self.qvec(query), k)

    def ranks(self, query: str, device: str) -> dict[str, list[int]]:
        """방법별 설명 번호 순위."""
        idx, bm, emb_p, emb = self.parts(query, device, self.qvec(query))

        def order(score: dict[int, float]) -> list[int]:
            return sorted(idx, key=lambda i: (-score[i], i))

        out = {"낱말 (지금)": order(bm), "뜻: 설명 글": order(emb_p), "뜻: 설명 글 + 학습 질문": order(emb)}
        rb, re_ = order(bm), order(emb)
        pos_b, pos_e = {i: r for r, i in enumerate(rb)}, {i: r for r, i in enumerate(re_)}
        # 낱말 점수가 0인 설명은 낱말 순위에 없는 것으로 (순서만으로 점수를 받지 않게)
        out["섞기: 순위(RRF)"] = order({i: (1 / (RRF_K + pos_b[i]) if bm[i] > 0 else 0) + 1 / (RRF_K + pos_e[i])
                                    for i in idx})
        for w in (0.3, 0.5, 0.7):
            out[f"섞기: 점수 낱말 {w}"] = mix(idx, bm, emb, w)
        return out

    def export_part(self) -> dict:
        """웹 실행기용: 설명·학습 질문 벡터(float32, base64)와 섞는 비율. 질문은 브라우저가 같은 모델로 임베딩."""
        def b64(a):
            return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()

        return {"model": REPO, "dtype": "q8", "weight": WEIGHT, "dim": int(self.P.shape[1]),
                "units": b64(self.P), "asked": b64(self.Q) if len(self.asked_rows) else "",
                "asked_unit": [i for i, _ in self.asked_rows]}


def evaluate(path: Path, hy: Hybrid, ks=(1, 3, 5)) -> dict[str, dict[str, float]]:
    """방법별 정답이 상위 k개에 든 비율, 그리고 상위 3개에 정답이나 같은 기능(code_rag.same_feature)이 든 비율."""
    by_id = {u["id"]: u for u in hy.units}
    hits: dict[str, list[int]] = {}
    n = 0
    for line in path.read_text().splitlines():
        ex = json.loads(line)
        if not ex.get("unit") or ":none:" in ex["unit"]:
            continue
        n += 1
        for name, order in hy.ranks(code_rag.first_question(ex), ex["device"]).items():
            found = [hy.units[i] for i in order[:max(ks)]]
            ids = [u["id"] for u in found]
            row = hits.setdefault(name, [0] * (len(ks) + 1))
            for j, k in enumerate(ks):
                row[j] += ex["unit"] in ids[:k]
            row[-1] += code_rag.same_feature(by_id.get(ex["unit"]), found[:3])
    names = [f"상위 {k}" for k in ks] + ["같은 기능 상위 3"]
    return {name: {key: round(v / max(n, 1), 3) for key, v in zip(names, row)} for name, row in hits.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--units", default=str(code_rag.UNITS))
    p.add_argument("--asked", default=str(code_rag.ASKED))
    p.add_argument("--eval", metavar="채점 대화")
    p.add_argument("--ask", nargs=2, metavar=("기기", "질문"))
    p.add_argument("--method", default="섞기: 순위(RRF)", help="--ask에 쓸 방법")
    args = p.parse_args()
    units = code_rag.load_units(Path(args.units))
    hy = Hybrid(units, code_rag.asked_from(Path(args.asked)), Embedder())
    if args.ask:
        for i in hy.ranks(args.ask[1], args.ask[0])[args.method][:code_rag.TOP_K]:
            print(units[i]["id"])
    if args.eval:
        res = evaluate(Path(args.eval), hy)
        width = max(len(k) for k in res)
        for name, row in res.items():
            print(f"{name:<{width}}  " + " / ".join(f"{k} {v:.3f}" for k, v in row.items()), flush=True)


if __name__ == "__main__":
    main()
