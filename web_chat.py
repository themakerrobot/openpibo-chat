"""다른 PC(윈도우 등)의 브라우저로 대화 모델을 시험하는 웹 화면. 학습 PC에서 띄웁니다.

    python web_chat.py                 # 브라우저에서 http://<학습 PC IP>:8500
    python web_chat.py --port 8600 --models themaker-ko

모델마다 llama-server를 하나씩 띄우고(GPU), 기기 앱(pibo_chat.py)과 같은 규칙으로 대답합니다: 시스템 글과
도구 목록, "멈춰"는 바로 멈춤, 자리를 옮기는 동작은 학생 말에 방향 낱말이 있을 때만. 도구는 실제로
실행하지 않고 화면에 "불빛 빨강"처럼 보여 줍니다. 기기를 파이브레인으로 고르면 움직이기는 못 한다고 말합니다.

같은 네트워크 안에서만 씁니다 (로그인 없음). 대화는 PC 메모리에만 두고 파일로 남기지 않습니다.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pibo_chat
from tools import SYSTEMS

#: 이름 → (GGUF, 말, llama-server 포트, 도구 목록을 보낼지)
MODELS = {
    "themaker-ko": ("out/themaker-0.8b-Q4_K_M.gguf", "ko", 8401, True),
    "themaker-en": ("out/themaker-en-0.8b-Q4_K_M.gguf", "en", 8402, True),
    "gemma3": ("baseline/before-Q4_K_M.gguf", "ko", 8403, False),
}
LABELS = {"themaker-ko": "더메이커 한국어", "themaker-en": "더메이커 영어", "gemma3": "젬마3 (비교용, 도구 없음)"}


class WebRobot(pibo_chat.Pibo):
    """실행하지 않고 한 일을 적어 두는 기기."""

    def __init__(self, can_move: bool):
        super().__init__(fake=True)
        self.can_move = can_move
        self.did: list[str] = []

    def light(self, color: str) -> None:
        self.did.append(f"불빛 {color}")

    def move(self, action: str) -> None:
        self.did.append(f"움직이기 {action}")

    def stop(self) -> None:
        self.did.append("멈춤")

    def look(self) -> str:
        self.did.append("보기 (카메라 흉내: 컵, 사과)")
        return super().look()


class Chat:
    """세션마다 대화 기억. pibo_chat의 LANG은 전역이라 한 번에 하나씩 대답합니다."""

    def __init__(self, urls: dict[str, str]):
        self.urls = urls
        self.sessions: dict[str, dict] = {}
        self.lock = threading.Lock()

    def reply(self, sid: str, model: str, device: str, text: str, tools: bool) -> dict:
        if model not in self.urls:
            return {"error": f"{model}은 띄우지 않았어요"}
        s = self.sessions.setdefault(sid, {"model": model, "history": []})
        if s["model"] != model:                  # 모델을 바꾸면 기억을 비움
            s.update(model=model, history=[])
        _, lang, _, model_tools = MODELS[model]
        robot = WebRobot(can_move=device != "pibrain")
        log = io.StringIO()
        with self.lock, contextlib.redirect_stdout(log):
            pibo_chat.LANG = lang
            t = time.time()
            said = pibo_chat.turn(self.urls[model], s["history"], text, robot, SYSTEMS[lang],
                                  tools and model_tools)
            seconds = time.time() - t
        notes = [ln.strip() for ln in log.getvalue().splitlines() if ln.strip()]
        return {"reply": said, "actions": robot.did, "notes": notes, "seconds": round(seconds, 1)}

    def reset(self, sid: str) -> None:
        self.sessions.pop(sid, None)


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>더메이커 시험</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f5f6f8;color:#222}
header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 16px;background:#fff;border-bottom:1px solid #ddd}
main{max-width:760px;margin:0 auto;padding:12px 16px 90px}
.msg{margin:8px 0;padding:8px 12px;border-radius:10px;max-width:85%;white-space:pre-wrap}
.me{background:#dcecff;margin-left:auto}.ai{background:#fff;border:1px solid #e3e3e3}
.act{font-size:13px;color:#0a6b3d;margin:2px 0 0 4px}.note{font-size:12px;color:#888;margin:0 0 0 4px}
footer{position:fixed;bottom:0;left:0;right:0;background:#fff;border-top:1px solid #ddd;padding:10px 16px}
footer form{display:flex;gap:8px;max-width:760px;margin:0 auto}
input[type=text]{flex:1;padding:10px;font-size:16px;border:1px solid #ccc;border-radius:8px}
button,select{padding:8px 12px;font-size:15px}
</style></head><body>
<header><b>더메이커 시험</b>
<select id="model">__OPTIONS__</select>
<select id="device"><option value="pibo">파이보</option><option value="pibrain">파이브레인</option></select>
<label><input type="checkbox" id="tools" checked> 도구</label>
<label><input type="checkbox" id="notes"> 시간·로그</label>
<button id="reset" type="button">새 대화</button></header>
<main id="log"></main>
<footer><form id="f"><input type="text" id="t" autocomplete="off" placeholder="말을 입력하세요"><button>보내기</button></form></footer>
<script>
let sid = crypto.randomUUID ? crypto.randomUUID() : String(Math.random());
const $ = id => document.getElementById(id), log = $("log");
function add(cls, text) { const d = document.createElement("div"); d.className = cls; d.textContent = text;
  log.appendChild(d); window.scrollTo(0, document.body.scrollHeight); return d; }
async function post(path, body) { const r = await fetch(path, {method: "POST",
  headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)}); return r.json(); }
$("f").onsubmit = async e => { e.preventDefault(); const text = $("t").value.trim(); if (!text) return;
  $("t").value = ""; add("msg me", text); const wait = add("msg ai", "…");
  try { const r = await post("/api/chat", {sid, text, model: $("model").value, device: $("device").value,
      tools: $("tools").checked});
    wait.textContent = r.error || r.reply;
    (r.actions || []).forEach(a => add("act", "[도구] " + a));
    if ($("notes").checked) { add("note", r.seconds + "초"); (r.notes || []).forEach(n => add("note", n)); }
  } catch (err) { wait.textContent = "연결 오류: " + err; } };
function fresh() { post("/api/reset", {sid}); log.innerHTML = ""; }
$("reset").onclick = fresh; $("model").onchange = fresh;
</script></body></html>"""


def handler_for(chat: Chat, options: str):
    page = PAGE.replace("__OPTIONS__", options).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):              # 대화 내용을 로그로 남기지 않음
            pass

        def send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, page, "text/html; charset=utf-8")
            else:
                self.send(404, b"", "text/plain")

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if self.path == "/api/chat":
                    out = chat.reply(str(body.get("sid") or uuid.uuid4()), str(body.get("model")),
                                     str(body.get("device")), str(body.get("text") or "").strip(),
                                     bool(body.get("tools", True)))
                elif self.path == "/api/reset":
                    chat.reset(str(body.get("sid")))
                    out = {}
                else:
                    return self.send(404, b"", "text/plain")
            except (OSError, ValueError) as exc:     # 서버 오류에도 화면은 계속
                out = {"error": f"오류: {exc}"}
            self.send(200, json.dumps(out, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8500, help="웹 화면 포트")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS),
                   help="띄울 모델 (파일이 없으면 건너뜀)")
    args = p.parse_args()
    from export import server

    with contextlib.ExitStack() as stack:
        urls = {}
        for name in args.models:
            gguf, _, port, _ = MODELS[name]
            if not Path(gguf).exists():
                print(f"{name}: {gguf}이 없어 건너뜁니다", flush=True)
                continue
            print(f"{name}: llama-server 띄우는 중 ({port})", flush=True)
            urls[name] = stack.enter_context(server(Path(gguf), port))
        if not urls:
            raise SystemExit("띄울 모델이 없어요 (out/ 에 GGUF가 있는지 보세요)")
        options = "".join(f'<option value="{n}">{LABELS[n]}</option>' for n in urls)
        web = ThreadingHTTPServer(("0.0.0.0", args.port), handler_for(Chat(urls), options))
        print(f"준비됨: 브라우저에서 http://<이 PC의 IP>:{args.port} (끄려면 Ctrl+C)", flush=True)
        try:
            web.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            web.server_close()


if __name__ == "__main__":
    main()
