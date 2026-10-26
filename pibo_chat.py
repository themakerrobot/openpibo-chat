"""파이보에서: llama-server의 대화 모델과 이야기하고, 몇 안 되는 동작은 앱이 말을 보고 바로 실행합니다.

    llama-server -m themaker-0.8b-Q4_K_M.gguf --jinja -c 2048 -np 1 --cache-ram 0 \\
        -t 4 -ub 256 --port 8080 &
    python3 pibo_chat.py                 # 키보드로 말하기
    python3 pibo_chat.py --fake          # openpibo가 없는 PC에서: 동작은 흉내만
    python3 pibo_chat.py --no-commands   # 동작 없이 대화만
    python3 pibo_chat.py --name 파이브레인 --about "코딩 로봇, 서큘러스에서 만듦" \\
        --tools eye_on,eye_off,get_weather   # 다른 기기: 이름·소개·있는 동작만 (모델은 같음)

대화 모델(themaker)은 모든 기기 공통이고 도구를 모릅니다 (도구 목록 1천 토큰을 읽느라
파이보에서 너무 느려서 뺐습니다). 동작(멈춰, 춤, 인사, 앞으로·뒤로, 눈 색, 배터리, 앞에 뭐 있어, 날씨)은
COMMANDS가 말을 보고 정해 바로 실행하고, 정해진 말로 대답합니다. 날씨만 받은 결과를 모델이 말로
풀어 줍니다. 음성 인식·합성은 listen()/speak()를 지금 쓰는 것으로 바꿔 끼우면 됩니다 (TTS
목소리는 lang_of로). 이 파일은 tools.py 하나만 있으면 됩니다.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request

from tools import COLORS, SYSTEM, TOOLS, check_call, has_batchim, system_for, tool_list

HISTORY = 3   # 기억하는 최근 주고받기 수 (지금 말 포함, --history로 바꿈)
FAKE = False  # --fake: 로봇 없이 동작을 한 척만 합니다
NAME = "파이보"   # --name: 기기 이름


def lang_of(text: str) -> str:
    """대답의 말: 한글이 있으면 "ko", 없으면 "en". TTS 목소리(언어)를 이걸로 고르세요."""
    return "ko" if re.search(r"[가-힣]", text) else "en"


def speak(text: str) -> None:
    print(f"{NAME}: {text}", flush=True)


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


#: YOLO 같은 검출기의 COCO 이름(VOC 이름도) → 한국어. 영어 이름을 그대로 주면 영어 낱말을
#: 소리 내 읽습니다.
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
    """로봇 없이 시험할 때의 결과."""
    return {"get_battery": "배터리 80%", "find_objects": "컵, 사과",
            "get_weather": f"{args.get('region', '서울')} 맑음 18도"}.get(name, "완료")


def run_tool(name: str, args: dict) -> str:
    if name not in [t["function"]["name"] for t in TOOLS]:
        return "실패: 이 기기에 없는 도구"
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


# ------------------------------------------------------------ commands (앱이 정하는 동작)

#: 말 끝에 붙는 부탁 ("줘", "봐" …)과 문장부호: 동작 명령은 말이 그걸로 끝날 때만 (이야기 속
#: "어제 앞으로 가 봤어"나 "춤 학원 갔어"는 동작이 아님)
TAIL = r"(\s*(줘|줄래|봐|볼래|라|자|요|주세요|줘요|봐요))?\s*[!.~?ㅋㅎ^]*$"
#: 멈추라는 말은 어디에 있어도 바로 멈춥니다 (안전)
STOP = re.compile(r"멈춰|멈추라|멈춰라|멈춰봐|정지|스톱|그만\s*움직|stop|\bfreeze\b", re.IGNORECASE)
#: 눈 색 낱말의 앞부분 → tools.COLORS의 색
COLOR_WORDS = {"빨": "빨강", "주황": "주황", "노": "노랑", "초록": "초록", "파": "파랑", "하늘": "하늘",
               "남": "남색", "보라": "보라", "분홍": "분홍", "핑크": "분홍", "하얗": "하양",
               "하얀": "하양", "흰": "하양", "연두": "연두", "청록": "청록", "갈": "갈색"}
EYE_COLOR = (r"눈\s*(을|색|색깔|불빛|빛|불)?\s*(을|좀)?\s*(?P<color>빨갛|빨간|주황|노랗|노란|초록|파랗|"
             r"파란|하늘|남|보라|분홍|핑크|하얗|하얀|흰|연두|청록|갈)\S*\s*(으로|로)?\s*(해|바꿔|켜|만들어)")
#: (말 모양, 도구, 인자, 한국어 대답, 영어 대답). 인자가 None이면 아래 command_for가 정함
COMMANDS = [
    (r"춤\s*(을\s*)?(좀\s*)?(춰|추어|추자|출래|보여)" + TAIL, "set_motion", "dance",
     "신나게 춤춰 볼게요!", None),
    (r"\b(let'?s dance|dance for me|can you dance|please dance|dance now)\b", "set_motion", "dance",
     None, "Let's dance!"),
    (r"인사\s*(를\s*)?(해|하자|할래)" + TAIL, "set_motion", {"name": "greeting"},
     "안녕하세요! 반가워요.", None),
    (r"손\s*(을\s*)?흔들어" + TAIL, "set_motion", {"name": "wave1"}, "손을 흔들게요!", None),
    (r"\b(wave|say hi)\b", "set_motion", {"name": "wave1"}, None, "Hi there!"),
    (r"박수\s*(를\s*)?(쳐|치자|칠래)" + TAIL, "set_motion", {"name": "clapping1"}, "짝짝짝!", None),
    (r"하이\s*파이브" + TAIL, "set_motion", {"name": "hifive"}, "하이파이브!", None),
    (r"악수\s*(를\s*)?(해|하자|할래)" + TAIL, "set_motion", {"name": "handshaking"}, "악수해요!", None),
    (r"고개\s*(를\s*)?끄덕여" + TAIL, "set_motion", {"name": "yes_h"}, "끄덕끄덕!", None),
    (r"고개\s*(를\s*)?(저어|흔들어)" + TAIL, "set_motion", {"name": "no_h"}, "도리도리!", None),
    (r"앞으로\s*(가|와|와\s*봐|가\s*봐|이동해|전진)" + TAIL, "set_motion", {"name": "forward1"},
     "앞으로 갈게요.", None),
    (r"뒤로\s*(가|물러\s*나|물러나|이동해)" + TAIL, "set_motion", {"name": "backward1"},
     "뒤로 갈게요.", None),
    (r"(왼쪽|왼편)\s*(으로)?\s*(돌아|가|돌|돌자)" + TAIL, "set_motion", {"name": "left"},
     "왼쪽으로 돌게요.", None),
    (r"(오른쪽|오른편)\s*(으로)?\s*(돌아|가|돌|돌자)" + TAIL, "set_motion", {"name": "right"},
     "오른쪽으로 돌게요.", None),
    (r"눈\s*(불빛|빛|불)?\s*(을|좀)?\s*(꺼|끄자|끌래)" + TAIL, "eye_off", {}, "눈 불빛을 껐어요.", None),
    (EYE_COLOR + TAIL, "eye_on", "color", "눈 색을 바꿨어요!", None),
    # 휴대폰 배터리 이야기는 빼고
    (r"^(?!.*(휴대폰|핸드폰|폰|노트북|태블릿)).*배터리\s*(가|는|이|좀)?\s*(얼마|몇|남았|남은|확인)",
     "get_battery", {}, "battery", None),
    (r"(앞에|눈앞에|지금|카메라로)\s*(뭐|무엇|뭐가|누가|누구)\s*(가\s*)?(있|보)", "find_objects", {},
     "objects", None),
    (r"날씨\s*(가|는|좀)?\s*(어때|어떄|어떨|어떻|알려)", "get_weather", "region",
     "weather", None),
]
REGIONS = ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "제주", "전국"]
DANCES = ["dance1", "dance2", "dance3", "dance4", "dance5"]


def command_for(text: str) -> tuple[str, dict, str] | None:
    """앱이 바로 할 동작: (도구, 인자, 대답 종류 또는 대답) 또는 None (그냥 대화).
    기기에 없는 도구·동작은 고르지 않습니다."""
    have = {t["function"]["name"]: t["function"] for t in TOOLS}
    for pattern, tool, args, ko, en in COMMANDS:
        m = re.search(pattern, text, re.IGNORECASE) if tool in have else None
        if not m:
            continue
        if args == "dance":
            moves = [m for m in DANCES if m in _motions(have)]
            if not moves:
                continue
            args = {"name": random.choice(moves)}
        elif args == "color":
            word = m.group("color")
            r, g, b = COLORS[next(n for k, n in COLOR_WORDS.items() if word.startswith(k))]
            args = {"r": r, "g": g, "b": b}
        elif args == "region":
            args = {"region": next((r for r in REGIONS if r in text), "서울")}
        if tool == "set_motion" and args["name"] not in _motions(have):
            continue
        return tool, args, (ko if ko is not None and lang_of(text) == "ko" or en is None else en)
    return None


def _motions(have: dict) -> list:
    spec = have.get("set_motion", {}).get("parameters", {}).get("properties", {}).get("name", {})
    return spec.get("enum", [])


def do_command(url: str, text: str, tool: str, args: dict, reply: str) -> str:
    """동작을 실행하고 할 말을 돌려줍니다. 날씨는 결과를 모델이 말로 풀어 줍니다."""
    t = time.time()
    result = run_tool(tool, args)
    print(f"  [동작] {tool}({args}) -> {result} ({time.time() - t:.1f}초)", flush=True)
    failed = result.startswith("실패")
    if reply == "battery":
        return "배터리를 확인하지 못했어요." if failed else \
            f"배터리가 {result.removeprefix('배터리').strip()} 남았어요."
    if reply == "objects":
        if failed:
            return "지금은 카메라로 보지 못했어요."
        if result == "아무것도 안 보임":
            return "앞에 아무것도 안 보여요."
        return f"앞에 {result}{'이' if has_batchim(result) else '가'} 보여요."
    if reply == "weather":
        if failed:
            return "지금은 날씨를 알아보지 못했어요."
        msg = ask(url, [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"{text}\n(알아본 날씨: {result})"}], False)
        return (msg.get("content") or "").strip()
    return "미안해요, 지금은 그걸 하지 못했어요." if failed else reply


# ------------------------------------------------------------ talking

#: 대답 설정: 같은 말을 되풀이하지 않게 (repeat_penalty: 같은 토큰, dry: 앞에 나온 말 덩어리)
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1,
            "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 3}


def ask(url: str, messages: list, tools: bool, max_tokens: int = 150, timeout: int = 300,
        **extra) -> dict:
    body = {"messages": messages, **SAMPLING, "max_tokens": max_tokens, **extra}
    if tools:
        body["tools"] = TOOLS
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    t = data.get("timings") or {}
    if t:   # 서버가 다시 읽은 토큰 수: 앞 요청을 못 쓰면 크게 나옵니다
        print(f"  (읽기 {t.get('prompt_n', 0)}토큰 {t.get('prompt_ms', 0) / 1000:.1f}초, "
              f"말하기 {t.get('predicted_n', 0)}토큰 {t.get('predicted_ms', 0) / 1000:.1f}초)",
              flush=True)
    return data["choices"][0]["message"]


def window(history: list) -> list:
    """최근 HISTORY번의 주고받기. 아이 말에서 끊어 도구 호출과 그 결과가 짝으로 남게 하고,
    젬마 템플릿처럼 아이 말로 시작해야 하는 경우도 맞춥니다."""
    starts = [i for i, m in enumerate(history) if m["role"] == "user"]
    return history[starts[-HISTORY]:] if len(starts) > HISTORY else history[:]


#: 동작을 끈 동안 동작을 부탁하면 하는 말
PAUSED = "지금은 대화만 하기로 했어요. 다시 움직이려면 이제 움직여도 된다고 말해 주세요."


def turn(url: str, history: list, text: str, tools: bool, allowed: bool = True) -> str:
    """아이 말 하나 → 파이보가 할 말. tools가 거짓이면 동작 없이 대화만. allowed가 거짓("대화만
    하자")이면 동작을 부탁해도 하지 않고 PAUSED로 대답합니다."""
    history.append({"role": "user", "content": text})
    if tools and STOP.search(text):
        if any(t["function"]["name"] == "set_motion" for t in TOOLS):   # 몸이 없는 기기는 말로만
            run_tool("set_motion", {"name": "stop"})
        said = "네, 멈출게요." if lang_of(text) == "ko" else "Okay, I'll stop."
        history.append({"role": "assistant", "content": said})
        return said
    msgs = [{"role": "system", "content": SYSTEM}] + window(history)
    cmd = command_for(text) if tools else None
    if cmd:
        said = do_command(url, text, *cmd) if allowed else PAUSED
    else:
        said = (ask(url, msgs, False).get("content") or "").strip()
        before = {same(m["content"]) for m in history if m["role"] == "assistant"}
        if same(said) in before:             # 앞에서 한 말 그대로: 한 번만 더 다르게
            print("  (앞과 같은 대답이라 다시 물음)", flush=True)
            said = (ask(url, msgs, False, temperature=1.0).get("content") or "").strip() or said
    history.append({"role": "assistant", "content": said})
    return said


def same(text: str) -> str:
    """대답을 견줄 모양: 띄어쓰기와 문장부호를 뺌."""
    return re.sub(r"[\s.,!?~…'\"]", "", text or "")


def warm_up(url: str, tools: bool = False) -> None:
    """시스템 글을 서버가 미리 읽어 두게 합니다. 다음부터는 서버가 앞부분을 기억해 새로 한
    말만 읽습니다."""
    try:
        urllib.request.urlopen(f"{url}/health", timeout=10).read()
    except OSError as exc:
        raise SystemExit(f"llama-server에 연결할 수 없어요 ({url}): 서버가 떠 있는지, 포트가 맞는지 "
                         f"확인하세요 ({exc})") from None
    print("  (준비 중)", flush=True)
    t = time.time()
    ask(url, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "안녕"}],
        False, max_tokens=1, timeout=900)
    print(f"  (준비 {time.time() - t:.1f}초)", flush=True)


#: "대화만 하자" 같은 말로 동작을 끄고 켭니다 (끈 동안 부탁한 동작은 하지 않음: turn의 allowed)
SWITCH_REPLY = {True: "좋아요, 이제 다시 움직일 수 있어요!",
                False: "좋아요, 지금부터는 대화만 할게요."}


def tool_switch(text: str, on: bool) -> bool:
    """아이 말에 따라 바뀐 동작 사용 여부 (띄어쓰기는 보지 않음)."""
    t = re.sub(r"\s", "", text)
    if re.search(r"(쓰지|사용하지|움직이지)(마|말)|대화만|말로만|얘기만|이야기만|가만히있", t):
        return False
    if re.search(r"움직여도|(이제|다시)움직|도구.*(쓰|써|사용)", t):
        return True
    return on


def main():
    global FAKE, NAME
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--no-commands", "--no-tools", dest="no_commands", action="store_true",
                   help="동작 없이 대화만")
    p.add_argument("--fake", action="store_true", help="openpibo 없이 동작을 한 척만")
    p.add_argument("--history", type=int, default=HISTORY,
                   help="기억하는 주고받기 수 (지금 말 포함)")
    p.add_argument("--name", default="파이보", help="기기 이름 (시스템 글에 들어감, 예: 파이브레인)")
    p.add_argument("--about", help="기기 소개 (예: '코딩 로봇, 서큘러스에서 만듦, 하얀색, 키 20센티미터')")
    p.add_argument("--tools", help="이 기기에 있는 도구만, 쉼표로 (예: eye_on,eye_off,get_weather)")
    p.add_argument("--motions", help="set_motion에서 쓸 수 있는 동작만, 쉼표로 (예: greeting,dance1,stop)")
    args = p.parse_args()
    FAKE, NAME = args.fake, args.name
    globals()["HISTORY"] = max(1, args.history)
    # 기기마다 다른 이름·소개·동작은 시스템 글과 도구 목록으로 (모델은 하나)
    globals()["SYSTEM"] = system_for(args.name, args.about)
    globals()["TOOLS"] = tool_list(args.tools.split(",") if args.tools else None,
                                   args.motions.split(",") if args.motions else None, args.name)
    use_tools = not args.no_commands
    warm_up(args.url, use_tools)
    history: list = []
    while True:
        try:
            text = listen()
        except EOFError:
            break
        if not text:
            continue
        if text in ("그만", "종료", "quit"):
            break
        if not args.no_commands:
            was, use_tools = use_tools, tool_switch(text, use_tools)
            if was != use_tools:
                # 끄고 켜는 말은 모델에 보내지 않고 정해진 말로 대답하고 기억을 비웁니다
                history.clear()
                print(f"  (동작 {'켬' if use_tools else '끔'})", flush=True)
                speak(SWITCH_REPLY[use_tools])
                continue
        t = time.time()
        try:
            speak(turn(args.url, history, text, not args.no_commands, use_tools))
        except OSError as exc:               # 서버 오류(HTTPError 포함)·연결 끊김: 앱은 계속
            body = exc.read().decode("utf-8", "ignore")[:300] if hasattr(exc, "read") else ""
            print(f"  (서버 오류: {exc} {body})", flush=True)
            history.clear()
            speak("잠깐 문제가 생겼어요. 한 번만 다시 말해 줄래요?")
        print(f"  (모두 {time.time() - t:.1f}초)", flush=True)


if __name__ == "__main__":
    main()
