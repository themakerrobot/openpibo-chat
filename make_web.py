"""파이썬이 없는 윈도우 PC에서 대화 모델을 시험하는 꾸러미를 만듭니다 (학습 PC에서 실행).

    python make_web.py                       # web/index.html만 다시 만듦
    python make_web.py --package out/themaker-win   # 윈도우 꾸러미 폴더 + zip

꾸러미: llama.cpp 윈도우판(llama-server.exe, CPU용) + 모델(GGUF) + 웹 페이지 하나(web/index.html) +
실행 파일(run-ko.bat, run-en.bat). 윈도우에서 bat을 두 번 누르면 llama-server가 그 페이지를 띄우고
브라우저가 열립니다. 파이썬도 설치도 필요 없습니다.

페이지는 기기 앱(pibo_chat.py)과 같은 규칙으로 대답합니다: 시스템 글·도구 목록(tools.py에서 그대로 넣음),
"멈춰"는 바로 멈춤, 자리를 옮기는 동작은 학생 말에 방향 낱말이 있을 때만, 같은 말이면 한 번 더 묻기.
도구는 실행하지 않고 화면에 "[도구] 불빛 빨강"처럼 보여 줍니다.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pibo_chat
from tools import SYSTEMS, TOOLS

WEB = Path("web/index.html")
MODELS = {"ko": "out/themaker-0.8b-Q4_K_M.gguf", "en": "out/themaker-en-0.8b-Q4_K_M.gguf"}

#: 기기 앱의 규칙 (브라우저에서 돔). 상수는 make_web이 tools.py·pibo_chat.py에서 넣음
LOGIC = r"""
const C = __CONSTANTS__;
const STOP = new RegExp(C.stop, "i");
const MOVE_WORDS = Object.fromEntries(Object.entries(C.moveWords).map(([k, v]) => [k, new RegExp(v, "i")]));
const HISTORY = 3;

function same(t) { return (t || "").replace(/[\s.,!?~…'"]/g, ""); }
function hangul(t) { return /[가-힣]/.test(t); }
function windowOf(history) {
  const starts = history.map((m, i) => m.role === "user" ? i : -1).filter(i => i >= 0);
  return starts.length > HISTORY ? history.slice(starts[starts.length - HISTORY]) : history.slice();
}
function argsOf(fn) {
  if (typeof fn.arguments !== "string") return fn.arguments || {};
  try { return JSON.parse(fn.arguments || "{}"); } catch (e) { return null; }
}
function callsOf(msg) {
  return (msg.tool_calls || []).map(c => [(c.function || {}).name || "", argsOf(c.function || {}) || {}]);
}
function checkCall(name, args) {
  const tool = C.tools.find(t => t.function.name === name);
  if (!tool || !args || typeof args !== "object") return "없는 도구나 잘못된 인자";
  const p = tool.function.parameters;
  for (const k of p.required || []) if (!(k in args)) return k + " 빠짐";
  for (const [k, v] of Object.entries(args)) {
    const spec = (p.properties || {})[k];
    if (!spec) return "모르는 인자 " + k;
    if (spec.enum && !spec.enum.includes(v)) return k + "=" + v + " 목록에 없음";
  }
  return null;
}
function blocked(text, call) {
  if (!call || call[0] !== "move") return false;
  const words = MOVE_WORDS[String(call[1].action)];
  return !!words && !words.test(text);
}
function act(state, name, args, said, did) {
  const S = C.say[state.lang];
  if (checkCall(name, args)) return said || S.unclear;
  if (name === "move" && !state.canMove) return state.lang === "ko" ? C.cantMoveKo : S.cant_move;
  if (name === "light") { did.push("불빛 " + args.color); return said || (args.color === "끄기" ? S.light_off : S.light_on); }
  if (name === "move") { did.push("움직이기 " + args.action); return said || S.moving; }
  did.push("보기 (카메라 흉내: 컵, 사과)");
  return (said ? said + " " : "") + (state.lang === "en" ? "I can see an apple and a cup." : "앞에 사과, 컵이 보여요.");
}
async function ask(state, messages, extra) {
  const body = Object.assign({messages, max_tokens: 150}, C.sampling, extra || {});
  if (state.tools) body.tools = C.tools;
  const r = await fetch(state.url + "/v1/chat/completions", {method: "POST",
    headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  if (!r.ok) throw new Error("서버 " + r.status);
  const data = await r.json();
  state.last = data.timings || null;
  return data.choices[0].message;
}
// 학생 말 하나 → {said, did, notes}. history에는 학생 말과 실제로 한 말만 남김 (기기 앱과 같게)
async function turn(state, history, text) {
  const did = [], notes = [];
  history.push({role: "user", content: text});
  if (STOP.test(text)) {
    if (state.canMove) did.push("멈춤");
    const said = hangul(text) ? "네, 멈출게요." : "Okay, I'll stop.";
    history.push({role: "assistant", content: said});
    return {said, did, notes};
  }
  const msgs = [{role: "system", content: C.systems[state.lang]}].concat(windowOf(history));
  let msg = await ask(state, msgs);
  let calls = state.tools ? callsOf(msg) : [];
  if (calls.length && blocked(text, calls[0])) {
    notes.push("막음: move(" + calls[0][1].action + ") 학생 말에 방향 낱말이 없음");
    msg = await ask(state, msgs, {temperature: 1.0});
    calls = callsOf(msg);
    if (calls.length && blocked(text, calls[0])) { msg = {content: C.say[state.lang].not_moving}; calls = []; }
  }
  const content = (msg.content || "").trim();
  let said;
  if (calls.length) {
    said = act(state, calls[0][0], calls[0][1], content, did);
  } else {
    said = content;
    const before = new Set(history.filter(m => m.role === "assistant").map(m => same(m.content)));
    if (before.has(same(said))) {
      notes.push("앞과 같은 대답이라 다시 물음");
      const again = await ask(state, msgs, {temperature: 1.0});
      if (!callsOf(again).length) said = (again.content || "").trim() || said;
    }
  }
  history.push({role: "assistant", content: said});
  return {said, did, notes};
}
"""

PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>더메이커 시험</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f5f6f8;color:#222}
header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 16px;background:#fff;border-bottom:1px solid #ddd}
main{max-width:760px;margin:0 auto;padding:12px 16px 90px}
.msg{margin:8px 0;padding:8px 12px;border-radius:10px;max-width:85%;white-space:pre-wrap}
.me{background:#dcecff;margin-left:auto}.ai{background:#fff;border:1px solid #e3e3e3}
.act{font-size:13px;color:#0a6b3d;margin:2px 0 0 4px}.note{font-size:12px;color:#888;margin:0 0 0 4px}
#status{font-size:13px;color:#888}
footer{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #ddd;padding:10px 16px}
footer form{display:flex;gap:8px;max-width:760px;margin:0 auto}
input[type=text]{flex:1;padding:10px;font-size:16px;border:1px solid #ccc;border-radius:8px}
button,select{padding:8px 12px;font-size:15px}
</style></head><body>
<header><b>더메이커 시험</b>
<select id="lang"><option value="ko">한국어 모델</option><option value="en">영어 모델</option></select>
<select id="device"><option value="pibo">파이보</option><option value="pibrain">파이브레인</option></select>
<label><input type="checkbox" id="tools" checked> 도구</label>
<label><input type="checkbox" id="notes"> 시간·로그</label>
<button id="reset" type="button">새 대화</button><span id="status">모델 준비 중…</span></header>
<main id="log"></main>
<footer><form id="f"><input type="text" id="t" autocomplete="off" placeholder="말을 입력하세요"><button>보내기</button></form></footer>
<script id="logic">__LOGIC__</script>
<script>
const $ = id => document.getElementById(id), log = $("log");
let history = [];
function state() { return {url: "", lang: $("lang").value, canMove: $("device").value !== "pibrain",
  tools: $("tools").checked}; }
function add(cls, text) { const d = document.createElement("div"); d.className = cls; d.textContent = text;
  log.appendChild(d); window.scrollTo(0, document.body.scrollHeight); return d; }
async function ready() {   // 서버가 모델을 다 읽을 때까지 기다림 (읽는 동안 /health는 503)
  for (;;) {
    try { const r = await fetch("/health"); if (r.ok) break; } catch (e) {}
    await new Promise(r => setTimeout(r, 1000));
  }
  try { const p = await (await fetch("/props")).json();
    if (/themaker-en/.test(p.model_path || "")) $("lang").value = "en"; } catch (e) {}
  $("status").textContent = "준비됨";
}
$("f").onsubmit = async e => { e.preventDefault(); const text = $("t").value.trim(); if (!text) return;
  $("t").value = ""; add("msg me", text); const wait = add("msg ai", "…"); const s = state();
  try { const t0 = performance.now(); const r = await turn(s, history, text);
    wait.textContent = r.said; r.did.forEach(a => add("act", "[도구] " + a));
    if ($("notes").checked) { add("note", ((performance.now() - t0) / 1000).toFixed(1) + "초");
      r.notes.forEach(n => add("note", n)); }
  } catch (err) { wait.textContent = "오류: " + err.message; } };
function fresh() { history = []; log.innerHTML = ""; }
$("reset").onclick = fresh; $("lang").onchange = fresh;
ready();
</script></body></html>
"""

#: 윈도우 명령 창은 글자 표가 PC마다 달라 bat 안의 글은 영어로만
BAT = """@echo off
cd /d "%~dp0"
if not exist "llama\\llama-server.exe" (
  echo llama\\llama-server.exe not found. See README.txt
  pause
  exit /b 1
)
echo TheMaker {label} model. Close this window to stop.
start "" http://127.0.0.1:8080
"llama\\llama-server.exe" -m "models\\{model}" --jinja -c 4096 -np 1 --host 127.0.0.1 --port 8080 --path web
pause
"""

README = """더메이커 시험 꾸러미 (윈도우, 파이썬 필요 없음)

1. run-ko.bat (한국어 모델) 또는 run-en.bat (영어 모델)을 두 번 누릅니다.
2. 검은 창이 뜨고 브라우저가 열립니다. "준비됨"이 보이면 아래 칸에 말을 입력합니다.
3. 끝내려면 검은 창을 닫습니다.

- 도구(불빛, 보기, 움직이기)는 실제로 실행하지 않고 대답 밑에 "[도구] 불빛 빨강"처럼 보여 줍니다.
- "파이브레인"을 고르면 움직이기는 못 한다고 대답합니다.
- 한 번에 모델 하나만 띄웁니다. 다른 모델을 쓰려면 창을 닫고 다른 bat을 누릅니다.
- llama 폴더가 비어 있으면 https://github.com/ggml-org/llama.cpp/releases 에서
  "bin-win-cpu-x64.zip"으로 끝나는 파일을 받아 llama 폴더에 풉니다 (llama-server.exe가 llama 폴더 바로 아래에).
- 윈도우 보안 경고(SmartScreen)가 뜨면 "추가 정보 → 실행"을 누릅니다.
- 이 PC 안에서만 돕니다 (127.0.0.1). 대화는 저장하지 않습니다.

모델: Qwen3.5-0.8B(Apache-2.0)를 미세조정. llama.cpp: MIT License.
"""
#: 데이터 출처 고지 (학습 PC에만 두는 파일: git에 올리지 않음). 있으면 README 끝에 붙임
NOTICE = Path("data/NOTICE.txt")


def constants() -> dict:
    return {"systems": SYSTEMS, "tools": TOOLS, "sampling": pibo_chat.SAMPLING,
            "stop": pibo_chat.STOP.pattern, "moveWords": {k: v.pattern for k, v in pibo_chat.MOVE_WORDS.items()},
            "say": pibo_chat.SAY, "cantMoveKo": pibo_chat.Pibo.cant_move}


def build_page() -> str:
    logic = LOGIC.replace("__CONSTANTS__", json.dumps(constants(), ensure_ascii=False))
    return PAGE.replace("__LOGIC__", logic)


def latest_llama_zip(dest: Path) -> Path | None:
    """llama.cpp 최신 릴리스의 윈도우 CPU판 zip을 받습니다 (못 받으면 None: README대로 손으로)."""
    try:
        with urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
                                    timeout=30) as r:
            assets = json.loads(r.read())["assets"]
        asset = next(a for a in assets if a["name"].endswith("bin-win-cpu-x64.zip"))
        out = dest / asset["name"]
        print(f"llama.cpp 받는 중: {asset['name']}", flush=True)
        urllib.request.urlretrieve(asset["browser_download_url"], out)
        return out
    except (OSError, StopIteration, KeyError, ValueError) as exc:
        print(f"llama.cpp 윈도우판을 못 받았어요 ({exc}): README.txt대로 손으로 넣으세요", flush=True)
        return None


def package(folder: Path) -> Path:
    """윈도우 꾸러미 폴더와 zip. 모델이 없는 말은 빼고, 파일은 학습 PC의 out/에서 복사."""
    (folder / "web").mkdir(parents=True, exist_ok=True)
    (folder / "models").mkdir(exist_ok=True)
    (folder / "llama").mkdir(exist_ok=True)
    (folder / "web" / "index.html").write_text(build_page(), encoding="utf-8")
    notice = NOTICE.read_text(encoding="utf-8").strip() if NOTICE.exists() else ""
    if not notice:
        print(f"주의: {NOTICE}가 없어 README에 데이터 출처 고지가 빠집니다 (배포 전에 넣으세요)", flush=True)
    (folder / "README.txt").write_text((README + ("\n" + notice + "\n" if notice else "")).replace("\n", "\r\n"),
                                       encoding="utf-8")
    labels = {"ko": "Korean", "en": "English"}
    for lang, src in MODELS.items():
        if not Path(src).exists():
            print(f"{src}이 없어 {lang} 모델은 빼요", flush=True)
            continue
        shutil.copy2(src, folder / "models" / Path(src).name)
        bat = BAT.replace("{label}", labels[lang]).replace("{model}", Path(src).name)
        (folder / f"run-{lang}.bat").write_bytes(bat.replace("\n", "\r\n").encode("utf-8"))
    if not (folder / "llama" / "llama-server.exe").exists():
        z = latest_llama_zip(folder)
        if z:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(folder / "llama")
            z.unlink()
            for exe in (folder / "llama").rglob("llama-server.exe"):   # 하위 폴더에 풀렸으면 끌어올림
                if exe.parent != folder / "llama":
                    for f in exe.parent.iterdir():
                        shutil.move(str(f), folder / "llama" / f.name)
                break
    out = folder.with_suffix(".zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:       # GGUF는 압축해도 거의 안 줄어듦
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                zf.write(f, Path(folder.name) / f.relative_to(folder))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", metavar="폴더", help="윈도우 꾸러미 폴더 (예: out/themaker-win)와 zip을 만듦")
    args = p.parse_args()
    WEB.parent.mkdir(exist_ok=True)
    WEB.write_text(build_page(), encoding="utf-8")
    print(f"{WEB}: 만듦", flush=True)
    if args.package:
        out = package(Path(args.package))
        print(f"{out}: {out.stat().st_size / 2**20:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
