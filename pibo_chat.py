"""파이보에서: llama-server의 모델과 대화하고, 모델이 부른 도구를 openpibo로 실행합니다.

    llama-server -m pibo-qwen3-0.6b-Q4_K_M.gguf --jinja -c 2048 -np 1 --port 8080 &
    python3 pibo_chat.py                 # 키보드로 말하기
    python3 pibo_chat.py --no-tools      # 대화만 (도구 목록을 보내지 않아 더 빠름)
    python3 pibo_chat.py --fake          # openpibo가 없는 PC에서: 도구는 흉내만

대화(chat)와 도구 실행만 합니다. 음성 인식·합성은 speak()/listen()을 지금
쓰는 것으로 바꿔 끼우면 됩니다. 이 파일은 tools.py 하나만 있으면 됩니다.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

from tools import SYSTEM, TOOLS, check_call

HISTORY = 6   # 기억하는 최근 메시지 수: 길수록 파이보에서 느려집니다
FAKE = False  # --fake: 로봇 없이 도구를 부른 척만 합니다


def speak(text: str) -> None:
    print(f"파이보: {text}", flush=True)


def listen() -> str:
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

def ask(url: str, messages: list, tools: bool) -> dict:
    body = {"messages": messages, "temperature": 0.6, "max_tokens": 150}
    if tools:
        body["tools"] = TOOLS
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]


def turn(url: str, history: list, text: str, tools: bool) -> str:
    history.append({"role": "user", "content": text})
    recent = history[-HISTORY:]
    while recent and recent[0]["role"] != "user":   # 젬마 템플릿은 아이 말로 시작해야 합니다
        recent = recent[1:]
    msgs = [{"role": "system", "content": SYSTEM}] + recent
    reply = ask(url, msgs, tools)
    for _ in range(3):                       # 도구 → 결과 → 다시 묻기, 세 번까지
        calls = reply.get("tool_calls") or []
        if not calls:
            break
        msgs.append({"role": "assistant", "content": reply.get("content") or "",
                     "tool_calls": calls})
        for c in calls:
            f = c["function"]
            args = f.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result = run_tool(f["name"], args)
            print(f"  [도구] {f['name']}({args}) -> {result}", flush=True)
            msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": result})
        reply = ask(url, msgs, tools)
    text = (reply.get("content") or "").strip()
    history.append({"role": "assistant", "content": text})
    return text


def main():
    global FAKE
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--no-tools", action="store_true")
    p.add_argument("--fake", action="store_true", help="openpibo 없이 도구를 부른 척만")
    args = p.parse_args()
    FAKE = args.fake
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
        t = time.time()
        speak(turn(args.url, history, text, not args.no_tools))
        print(f"  ({time.time() - t:.1f}초)", flush=True)


if __name__ == "__main__":
    main()
