"""코딩 도우미 재료: 기기 저장소(openpibo-os.pibo / .pibrain)에서 블록과 openpibo 파이썬 함수를 뽑습니다.

    python code_docs.py --repo 파이보=~/openpibo-os.pibo --repo 파이브레인=~/openpibo-os.pibrain

- 블록: 학생 화면의 블록 상자(ide/static/customblock_toolbox.js)에 보이는 블록만. 이름·한국어 글·툴팁·고르는
  값·그 블록이 만드는 파이썬 코드 (code_blocks.js가 가짜 Blockly 위에서 돌려 뽑음, node 필요)
- 함수: 그 블록들이 쓰는 클래스(예: 파이브레인은 DeviceByPiBrain as Device)의 공개 메서드와 설명(docstring)
- 검사용 목록: 기기마다 쓸 수 있는 import, 객체 이름 → 클래스, 클래스 → 메서드 (code_data.py가 대답 속 코드를 대조)

결과: data/code-units.jsonl (한 줄에 블록 하나 또는 함수 하나), data/code-api.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNITS = Path("data/code-units.jsonl")
API = Path("data/code-api.json")
FROM = re.compile(r"^from (openpibo\.[\w.]+) import (\w+)(?: as (\w+))?$")
NEW = re.compile(r"^(\w+) = (\w+)\(\);?$")
#: 블록 안내서(docs/source/blocks/guide.md)의 블록 그림 줄과 그 아래 설명 한 단락
GUIDE = re.compile(r'<img src="img/(\w+)\.png" alt="([^"]*)"[^>]*>\s*\n\s*\n([^\n<#][^\n]*(?:\n[^\n<#][^\n]*)*)')
#: 설명만으로는 무엇을 하는지 알 수 없는 블록에 사람이 붙인 설명
NOTES = {
    "flag_event": ("코드를 실행하는 시작 블록이에요. 이 블록 아래에 이어 붙인 블록들이 위에서부터 차례로 실행돼요. "
                   "값을 넣는 칸은 없고, 이 블록 자체는 파이썬 코드를 만들지 않아요."),
}


def blocks_of(repo: Path) -> list[dict]:
    node = shutil.which("node")
    if not node:
        raise SystemExit("node가 필요해요: sudo apt install -y nodejs")
    out = subprocess.run([node, str(HERE / "code_blocks.js"), str(repo / "ide" / "static")],
                         capture_output=True, text=True, check=True)
    if out.stderr.strip():
        print(out.stderr.strip(), flush=True)
    return json.loads(out.stdout)


def signature(fn: ast.FunctionDef) -> str:
    a = fn.args
    parts = []
    pos = a.posonlyargs + a.args
    defaults = [None] * (len(pos) - len(a.defaults)) + list(a.defaults)
    for arg, default in zip(pos, defaults, strict=True):
        if arg.arg == "self":
            continue
        parts.append(arg.arg + (f"={ast.unparse(default)}" if default is not None else ""))
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for arg, default in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        parts.append(arg.arg + (f"={ast.unparse(default)}" if default is not None else ""))
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return f"{fn.name}({', '.join(parts)})"


def methods_of(repo: Path, module: str, cls: str) -> list[dict]:
    """openpibo 모듈 파일에서 클래스의 공개 메서드들: [{method, signature, doc}]."""
    path = repo / Path(*module.split("."))
    path = path.with_suffix(".py") if path.with_suffix(".py").exists() else path / "__init__.py"
    if not path.exists():
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            out = []
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and not fn.name.startswith("_"):
                    doc = re.sub(r"\n{3,}", "\n\n", (ast.get_docstring(fn) or "").strip())
                    out.append({"method": fn.name, "signature": signature(fn), "doc": doc[:1500]})
            return out
    return []


def guide_of(repo: Path) -> dict[str, tuple[str, str]]:
    """블록 이름 → (안내서의 블록 모양, 설명)."""
    path = repo / "docs" / "source" / "blocks" / "guide.md"
    if not path.exists():
        return {}
    return {m.group(1): (m.group(2).strip(), " ".join(m.group(3).split()))
            for m in GUIDE.finditer(path.read_text(encoding="utf-8"))}


def short_name(text: str) -> str:
    """블록 글에서 칸과 아이콘을 뺀 부름 이름 (「RGB 눈 불빛 바꾸기」)."""
    return " ".join(re.sub(r"\([^)]*\)|□", " ", text).split())


def extract(device: str, repo: Path) -> tuple[list[dict], dict]:
    guide = guide_of(repo)
    blocks = []
    for b in blocks_of(repo):
        if b["type"] in guide:
            b["shape"], b["guide"] = guide[b["type"]]
        if b["type"] in NOTES:
            b["note"] = NOTES[b["type"]]
        if not b.get("code") and not b.get("note"):      # 무엇이 되는지 모르는 블록은 문답을 만들지 않음
            print(f"{device}: {b['type']} 블록은 만드는 코드가 없어 뺌", flush=True)
            continue
        b["name"] = b.get("shape") or short_name(b["text"])     # 안내서의 블록 모양 (칸의 기본값 포함)
        blocks.append(b)
    imports, instances = {}, {}
    for b in blocks:
        for line in b["imports"]:
            if m := FROM.match(line.strip()):
                imports[m.group(3) or m.group(2)] = [m.group(1), m.group(2)]
            elif m := NEW.match(line.strip()):
                instances[m.group(1)] = m.group(2)
    units, methods = [], {}
    for b in blocks:
        units.append({"id": f"{device}:block:{b['type']}", "device": device, "kind": "block", **b})
    for alias, (module, cls) in sorted(imports.items()):
        ms = methods_of(repo, module, cls)
        methods[alias] = [m["method"] for m in ms]
        inst = next((i for i, c in instances.items() if c == alias), None)
        for m in ms:
            units.append({"id": f"{device}:func:{cls}.{m['method']}", "device": device, "kind": "func",
                          "module": module, "class": cls, "alias": alias, "instance": inst, **m})
    by_cat: dict[str, list[str]] = {}
    for b in blocks:
        by_cat.setdefault(b.get("category") or "", []).append(b["name"])
    api = {"imports": imports, "instances": instances, "methods": methods, "blocks": by_cat}
    return units, api


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", action="append", required=True, metavar="기기=경로",
                   help="예: 파이보=~/openpibo-os.pibo (여러 번)")
    args = p.parse_args()
    all_units, apis = [], {}
    for item in args.repo:
        device, path = item.split("=", 1)
        repo = Path(path).expanduser()
        if not (repo / "ide" / "static" / "customblock.js").exists():
            raise SystemExit(f"{repo}에 ide/static/customblock.js가 없어요")
        units, api = extract(device, repo)
        n_block = sum(u["kind"] == "block" for u in units)
        print(f"{device}: 블록 {n_block}개, 함수 {len(units) - n_block}개 "
              f"(클래스 {', '.join(f'{a}={c[1]}' for a, c in api['imports'].items())})", flush=True)
        all_units += units
        apis[device] = api
    UNITS.parent.mkdir(exist_ok=True)
    UNITS.write_text("".join(json.dumps(u, ensure_ascii=False) + "\n" for u in all_units))
    API.write_text(json.dumps(apis, ensure_ascii=False, indent=1))
    print(f"{UNITS}: {len(all_units)}개, {API}", flush=True)


if __name__ == "__main__":
    main()
