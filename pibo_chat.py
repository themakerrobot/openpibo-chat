"""파이보에서: llama-server의 모델과 대화하고, 모델이 부른 도구를 openpibo로 실행합니다.

    llama-server -m pibo-qwen3.5-0.8b-school-small-Q4_K_M.gguf --jinja -c 2048 -np 1 --cache-ram 0 \
        -t 4 -ub 256 --port 8080 &
    python3 pibo_chat.py                 # 키보드로 말하기
    python3 pibo_chat.py --no-tools      # 대화만 (도구 목록을 보내지 않아 더 빠름)
    python3 pibo_chat.py --fake          # openpibo가 없는 PC에서: 도구는 흉내만

대화(chat)와 도구 실행만 합니다. 음성 인식·합성은 speak()/listen()을 지금
쓰는 것으로 바꿔 끼우면 됩니다. 이 파일은 tools.py 하나만 있으면 됩니다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request

from tools import SYSTEM, TOOLS, check_call

#: 기억하는 최근 주고받기 수 (지금 말 포함, --history로 바꿈). 서버는 앞 요청과 같은 앞부분
#: (시스템 글과 도구 목록)을 다시 쓰고 새로 붙은 말만 읽습니다 (재 보니 지난 대화 세 번을 붙여도
#: 새로 읽는 건 수십 토큰). 대화마다 찍히는 '읽기 N토큰'이 1천 가까이 나오면 다시 쓰지 못한 것
HISTORY = 3
FAKE = False  # --fake: 로봇 없이 도구를 부른 척만 합니다


def lang_of(text: str) -> str:
    """대답의 말: 한글이 있으면 "ko", 없으면 "en". TTS 목소리(언어)를 이걸로 고르세요:
    학생이 영어로 말하면 모델이 영어로 대답합니다."""
    return "ko" if re.search(r"[가-힣]", text) else "en"


def speak(text: str) -> None:
    print(f"파이보: {text}", flush=True)


def listen() -> str:
    # 원격 터미널에서 한글을 지우고 다시 쓰면 글자 조각(깨진 UTF-8)이 들어올 때가 있어,
    # 멈추지 않고 깨진 부분만 버립니다
    if hasattr(sys.stdin, "reconfigure") and getattr(sys.stdin, "errors", "") != "ignore":
        try:
            sys.stdin.reconfigure(errors="ignore")
        except (OSError, ValueError):        # 이미 읽기 시작한 입력(파이프 등)은 못 바꿈
            pass
    return input("나: ").strip()


# ------------------------------------------------------------ the tools on the robot

_parts: dict = {}


def _part(name):
    if name not in _parts:
        if name == "motion":
            from openpibo.motion import Motion
            _parts[name] = Motion()
        elif name == "device":
            from openpibo.device import Device
            _parts[name] = Device()
    return _parts[name]


#: YOLO 같은 검출기의 COCO 이름(VOC 이름도) → 한국어. 학습 데이터의 도구 결과가
#: "컵, 사과"처럼 한국어라, 영어 이름을 그대로 주면 모델이 영어 낱말을 소리 내 읽습니다.
KOREAN = {
    "person": "사람", "bicycle": "자전거", "car": "자동차", "motorcycle": "오토바이",
    "motorbike": "오토바이", "airplane": "비행기", "aeroplane": "비행기", "bus": "버스",
    "train": "기차", "truck": "트럭", "boat": "배", "traffic light": "신호등",
    "fire hydrant": "소화전", "stop sign": "멈춤 표지판", "parking meter": "주차 요금기",
    "bench": "벤치", "bird": "새", "cat": "고양이", "dog": "강아지", "horse": "말",
    "sheep": "양", "cow": "소", "elephant": "코끼리", "bear": "곰", "zebra": "얼룩말",
    "giraffe": "기린", "backpack": "가방", "umbrella": "우산", "handbag": "손가방",
    "tie": "넥타이", "suitcase": "여행 가방", "frisbee": "원반", "skis": "스키",
    "snowboard": "스노보드", "sports ball": "공", "kite": "연", "baseball bat": "야구 방망이",
    "baseball glove": "야구 글러브", "skateboard": "스케이트보드", "surfboard": "서핑보드",
    "tennis racket": "테니스 라켓", "bottle": "병", "wine glass": "유리잔", "cup": "컵",
    "fork": "포크", "knife": "칼", "spoon": "숟가락", "bowl": "그릇", "banana": "바나나",
    "apple": "사과", "sandwich": "샌드위치", "orange": "오렌지", "broccoli": "브로콜리",
    "carrot": "당근", "hot dog": "핫도그", "pizza": "피자", "donut": "도넛", "cake": "케이크",
    "chair": "의자", "couch": "소파", "sofa": "소파", "potted plant": "화분",
    "pottedplant": "화분", "bed": "침대", "dining table": "식탁", "diningtable": "식탁",
    "toilet": "변기", "tv": "텔레비전", "tvmonitor": "텔레비전", "laptop": "노트북",
    "mouse": "마우스", "remote": "리모컨", "keyboard": "키보드", "cell phone": "휴대폰",
    "microwave": "전자레인지", "oven": "오븐", "toaster": "토스터", "sink": "싱크대",
    "refrigerator": "냉장고", "book": "책", "clock": "시계", "vase": "꽃병",
    "scissors": "가위", "teddy bear": "곰 인형", "hair drier": "드라이기",
    "toothbrush": "칫솔",
}


def korean_names(names) -> str:
    """검출된 이름들 → "사람, 컵" (겹치면 하나, 모르는 이름은 그대로)."""
    said = {KOREAN.get(str(n).strip().lower().replace("_", " "), str(n).strip())
            for n in names if n and str(n).strip()}
    return ", ".join(sorted(said)) or "아무것도 안 보임"


def find_objects() -> str:
    try:
        from openpibo.vision_camera import Camera
        from openpibo.vision_detect import Detect
    except ImportError:
        from openpibo.vision import Camera, Detect
    img = Camera().read()
    items = Detect().detect_object(img)
    return korean_names(i.get("name") if isinstance(i, dict) else i for i in items or [])


def fake_tool(name: str, args: dict) -> str:
    """학습 데이터의 결과와 같은 모양으로 지어낸 결과."""
    return {"get_battery": "배터리 80%", "find_objects": "컵, 사과",
            "get_weather": f"{args.get('region', '서울')} 맑음 18도"}.get(name, "완료")


def run_tool(name: str, args: dict) -> str:
    problem = check_call(name, args)
    if problem:
        return f"실패: {problem}"
    if FAKE:
        return fake_tool(name, args)
    try:
        if name == "set_motion":
            _part("motion").set_motion(args["name"])
        elif name == "eye_on":
            _part("device").eye_on(args["r"], args["g"], args["b"])
        elif name == "eye_off":
            _part("device").eye_off()
        elif name == "get_battery":
            return f"배터리 {_part('device').get_battery()}"
        elif name == "find_objects":
            return find_objects()
        elif name == "get_weather":
            from openpibo.collect import Weather
            return str(Weather().search(args["region"]))[:200]
        return "완료"
    except Exception as exc:  # noqa: BLE001 — 로봇이 멈추지 않게
        return f"실패: {exc}"


# ------------------------------------------------------------ talking

def ask(url: str, messages: list, tools: bool, max_tokens: int = 150, timeout: int = 300) -> dict:
    body = {"messages": messages, "temperature": 0.6, "max_tokens": max_tokens}
    if tools:
        body["tools"] = TOOLS
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    t = data.get("timings") or {}
    if t:   # 서버가 다시 읽은 토큰 수: 앞 요청을 못 쓰면 여기가 1천 넘게 나옵니다
        print(f"  (읽기 {t.get('prompt_n', 0)}토큰 {t.get('prompt_ms', 0) / 1000:.1f}초, "
              f"말하기 {t.get('predicted_n', 0)}토큰 {t.get('predicted_ms', 0) / 1000:.1f}초)",
              flush=True)
    return data["choices"][0]["message"]


def window(history: list) -> list:
    """최근 HISTORY번의 주고받기. 아이 말에서 끊어 도구 호출과 그 결과가 짝으로 남게 하고,
    젬마 템플릿처럼 아이 말로 시작해야 하는 경우도 맞춥니다."""
    starts = [i for i, m in enumerate(history) if m["role"] == "user"]
    return history[starts[-HISTORY]:] if len(starts) > HISTORY else history[:]


#: 도구를 끈 동안 모델이 도구를 부르면 실행하지 않고 하는 말
PAUSED = "지금은 대화만 하기로 했어요. 다시 움직이려면 이제 움직여도 된다고 말해 주세요."
#: 멈추라는 말은 모델에 맡기지 않고 바로 멈춥니다 (안전: 작은 모델이 놓치면 안 되는 명령)
STOP = re.compile(r"멈춰|멈추라|멈춰라|멈춰봐|정지|스톱|그만\s*움직|stop|\bfreeze\b", re.IGNORECASE)


def first_move(url: str, text: str) -> list | None:
    """도구를 부를지는 지난 대화 없이 지금 말 하나로 정합니다: 학습한 도구 예시가 모두 한 번
    묻고 끝나는 대화라, 지난 대화가 붙으면 "멈춰"에 "멈추고 있어요"처럼 말로만 하고 도구를
    안 부릅니다. 스트리밍으로 받아 첫 글이 말이면 바로 끊으므로(토큰 한두 개) 시간이 거의 안
    듭니다. 도구 호출이면 끝까지 받아 돌려주고, 아니면 None."""
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            "tools": TOOLS, "temperature": 0, "max_tokens": 120, "stream": True}
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    calls: dict = {}
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            choices = json.loads(data).get("choices") or []
            delta = (choices[0].get("delta") or {}) if choices else {}
            for tc in delta.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"id": "", "type": "function",
                                                           "function": {"name": "", "arguments": ""}})
                c["id"] = tc.get("id") or c["id"]
                f = tc.get("function") or {}
                c["function"]["name"] = f.get("name") or c["function"]["name"]   # 이름은 한 번에 옴
                c["function"]["arguments"] += f.get("arguments") or ""
            if not calls and (delta.get("content") or "").strip():
                return None                  # 말로 시작: 연결을 닫으면 서버가 생성을 멈춥니다
    return [calls[i] for i in sorted(calls)] or None


def run_calls(calls: list, msgs: list) -> None:
    # 새 llama-server는 기록 속 도구 호출에 type이 없으면 500 오류를 냅니다
    calls = [dict(c, type="function", id=c.get("id") or f"call_{k}") for k, c in enumerate(calls)]
    msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
    for c in calls:
        f = c["function"]
        args = f.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        t = time.time()
        result = run_tool(f["name"], args)
        print(f"  [도구] {f['name']}({args}) -> {result} ({time.time() - t:.1f}초)", flush=True)
        msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": result})


def turn(url: str, history: list, text: str, tools: bool, allowed: bool = True) -> str:
    """allowed가 거짓(아이가 "대화만 하자")이면 모델이 부른 도구를 실행하지 않고 PAUSED로
    대답합니다 (도구 목록을 빼면 모델이 "앞으로 갈게요!"처럼 하지도 않은 일을 했다고 말함)."""
    history.append({"role": "user", "content": text})
    if tools and STOP.search(text):
        run_calls([{"id": "stop", "function": {"name": "set_motion",
                                               "arguments": '{"name": "stop"}'}}], [])
        history.append({"role": "assistant", "content": "네, 멈출게요."})
        return "네, 멈출게요."
    msgs = [{"role": "system", "content": SYSTEM}] + window(history)
    new_from = len(msgs)
    calls = None
    if tools:
        t = time.time()
        try:
            calls = first_move(url, text)
        except (OSError, ValueError, KeyError):   # 스트리밍 도구 호출을 못 하는 옛 서버
            calls = None
        print(f"  (도구 판단 {time.time() - t:.1f}초: {'도구' if calls else '말'})", flush=True)
    reply = {"content": "", "tool_calls": calls} if calls else ask(url, msgs, tools)
    if not allowed and reply.get("tool_calls"):
        history.append({"role": "assistant", "content": PAUSED})
        return PAUSED
    for _ in range(3):                       # 도구 → 결과 → 다시 묻기, 세 번까지
        if not reply.get("tool_calls"):
            break
        run_calls(reply["tool_calls"], msgs)
        reply = ask(url, msgs, tools)
    text = (reply.get("content") or "").strip()
    # 도구를 부른 것과 그 결과도 기억합니다: 마지막 말만 남기면 모델이 앞 차례를 "도구 없이
    # '앞으로 갔어요'라고 말했다"로 보고 따라 합니다
    history += msgs[new_from:] + [{"role": "assistant", "content": text}]
    return text


def warm_up(url: str, tools: bool) -> None:
    """시스템 글과 도구 목록(약 1천 토큰)을 서버가 미리 읽어 두게 합니다: 파이보 CPU에서는
    이걸 읽는 데 수십 초가 걸려, 첫 말이 그만큼 늦어집니다. 다음부터는 서버가 앞부분을
    기억해(같은 자리, 같은 앞부분) 새로 한 말만 읽습니다."""
    try:
        urllib.request.urlopen(f"{url}/health", timeout=10).read()
    except OSError as exc:
        raise SystemExit(f"llama-server에 연결할 수 없어요 ({url}): 서버가 떠 있는지, 포트가 맞는지 "
                         f"확인하세요 ({exc})") from None
    print("  (준비 중: 도구 목록을 미리 읽어요. 파이보에서는 1분쯤 걸려요)", flush=True)
    t = time.time()
    ask(url, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "안녕"}],
        tools, max_tokens=1, timeout=900)
    print(f"  (준비 {time.time() - t:.1f}초)", flush=True)


#: "대화만 하자" 같은 말로 도구를 끄고 켭니다. 모델은 이런 지시를 배운 적이 없어 스스로는
#: 안 지키므로 앱이 지킵니다: 끈 동안 부른 도구는 실행하지 않습니다 (turn의 allowed)
SWITCH_REPLY = {True: "좋아요, 이제 다시 움직일 수 있어요!",
                False: "좋아요, 지금부터는 대화만 할게요."}


def tool_switch(text: str, on: bool) -> bool:
    """아이 말에 따라 바뀐 도구 사용 여부 (띄어쓰기는 보지 않음)."""
    t = re.sub(r"\s", "", text)
    if re.search(r"(쓰지|사용하지|움직이지)(마|말)|대화만|말로만|얘기만|이야기만|가만히있", t):
        return False
    if re.search(r"움직여도|(이제|다시)움직|도구.*(쓰|써|사용)", t):
        return True
    return on


def main():
    global FAKE
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--no-tools", action="store_true")
    p.add_argument("--fake", action="store_true", help="openpibo 없이 도구를 부른 척만")
    p.add_argument("--history", type=int, default=HISTORY,
                   help="기억하는 주고받기 수 (지금 말 포함, 늘리면 파이보에서 많이 느려질 수 있음)")
    args = p.parse_args()
    FAKE = args.fake
    globals()["HISTORY"] = max(1, args.history)
    warm_up(args.url, not args.no_tools)
    history: list = []
    use_tools = not args.no_tools
    while True:
        try:
            text = listen()
        except EOFError:
            break
        if not text:
            continue
        if text in ("그만", "종료", "quit"):
            break
        if not args.no_tools:
            was, use_tools = use_tools, tool_switch(text, use_tools)
            if was != use_tools:
                # 끄고 켜는 말은 모델에 보내지 않고(보내면 "움직여도 돼"에 춤을 춤) 정해진 말로
                # 대답하고, 끈 동안의 대화를 따라 하지 않게 기억을 비웁니다
                history.clear()
                print(f"  (도구 {'켬' if use_tools else '끔'})", flush=True)
                speak(SWITCH_REPLY[use_tools])
                continue
        t = time.time()
        try:
            speak(turn(args.url, history, text, not args.no_tools, use_tools))
        except OSError as exc:               # 서버 오류(HTTPError 포함)·연결 끊김: 앱은 계속
            body = exc.read().decode("utf-8", "ignore")[:300] if hasattr(exc, "read") else ""
            print(f"  (서버 오류: {exc} {body})", flush=True)
            history.clear()
            speak("잠깐 문제가 생겼어요. 한 번만 다시 말해 줄래요?")
        print(f"  (모두 {time.time() - t:.1f}초)", flush=True)


if __name__ == "__main__":
    main()
