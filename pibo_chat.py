"""파이보에서: llama-server의 대화 모델(themaker)과 이야기합니다. 불빛, 보기, 움직이기는 모델이
학생 말을 보고 도구로 부르고, 이 앱이 파이보의 openpibo 함수로 실행합니다.

    llama-server -m themaker-0.8b-Q4_K_M.gguf --jinja -c 2048 -np 1 --cache-ram 0 \\
        -t 4 -ub 256 --port 50020 &
    sudo ~/.pyenv/bin/python3 pibo_chat.py --url http://127.0.0.1:50020   # 키보드로 말하기
    python3 pibo_chat.py --fake        # openpibo가 없는 PC에서: 동작은 흉내만
    python3 pibo_chat.py --no-tools    # 도구 없이 대화만

파이브레인은 pibrain_chat.py (같은 모델, 같은 도구 목록, 불빛 함수만 다르고 움직이기는 없음).

- 앱은 매번 tools.TOOLS(모든 기기 공통: 불빛, 보기, 움직이기)를 모델에 보냅니다. 도구를 부를지는
  모델이 정합니다. 시스템 글과 도구 목록은 매번 같은 앞부분이라, 켤 때 한 번 읽혀 두면(warm_up)
  대화할 때는 새로 한 말만 읽습니다.
- 모델이 도구를 부르면 그 앞의 짧은 말("빨간색으로 바꿀게요!")을 하고 실행합니다. 보기는 본
  결과("앞에 컵이 보여요.")를 덧붙입니다. 기기에 없는 동작이면 실행하지 않고 없다고 말합니다.
- "멈춰"는 안전을 위해 모델에 묻지 않고 바로 멈춥니다.
- 음성 인식·합성은 listen()/speak()를 지금 쓰는 것으로 바꿔 끼우면 됩니다 (TTS 목소리는 lang_of로).
- openpibo의 Device(MCU)·카메라는 root 권한이 필요해 sudo로 실행합니다.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request

from tools import ACTIONS, COLORS, TOOLS, calls_of, check_call, has_batchim, system_for

HISTORY = 3   # 기억하는 최근 주고받기 수 (지금 말 포함, --history로 바꿈)


def lang_of(text: str) -> str:
    """대답의 말: 한글이 있으면 "ko", 없으면 "en". TTS 목소리(언어)를 이걸로 고르세요."""
    return "ko" if re.search(r"[가-힣]", text) else "en"


def speak(name: str, text: str) -> None:
    print(f"{name}: {text}", flush=True)


def listen() -> str:
    # 원격 터미널에서 한글을 지우고 다시 쓰면 글자 조각(깨진 UTF-8)이 들어올 때가 있어,
    # 멈추지 않고 깨진 부분만 버립니다
    if hasattr(sys.stdin, "reconfigure") and getattr(sys.stdin, "errors", "") != "ignore":
        try:
            sys.stdin.reconfigure(errors="ignore")
        except (OSError, ValueError):        # 이미 읽기 시작한 입력(파이프 등)은 못 바꿈
            pass
    return input("나: ").strip()


# ------------------------------------------------------------ 기기 (파이보)

#: YOLO 같은 검출기의 COCO 이름 → 한국어 (영어 이름을 그대로 주면 영어 낱말을 소리 내 읽음)
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


class Pibo:
    """파이보의 openpibo 함수 (openpibo-os.pibo 기준). fake면 실행하지 않고 흉내만."""

    name = "파이보"
    can_move = True
    #: 몸이 없는 기기가 움직이라는 말을 들었을 때 (파이브레인)
    cant_move = "저는 몸이 없어서 움직일 수는 없어요. 대신 불빛을 바꿔 볼까요?"

    def __init__(self, fake: bool = False):
        self.fake = fake
        self._parts: dict = {}

    def part(self, name: str):
        if name not in self._parts:
            if name == "motion":
                from openpibo.motion import Motion
                self._parts[name] = Motion()
            elif name == "device":
                from openpibo.device import Device
                self._parts[name] = Device()
        return self._parts[name]

    def light(self, color: str) -> None:
        """눈 불빛: Device.eye_on(r, g, b) / eye_off()."""
        rgb = COLORS[color]
        if self.fake:
            return
        dev = self.part("device")
        dev.eye_off() if rgb is None else dev.eye_on(*rgb)

    def move(self, action: str) -> None:
        """Motion.set_motion(동작 이름): 춤처럼 여럿이면 하나를 골라서."""
        if not self.fake:
            self.part("motion").set_motion(random.choice(ACTIONS[action]))

    def stop(self) -> None:
        if not self.fake:
            self.part("motion").set_motion("stop")

    def look(self) -> str:
        """카메라로 찍어 사물 인식 → "컵, 사람" (없으면 "아무것도 안 보임")."""
        if self.fake:
            return "컵, 사과"
        from openpibo.vision_camera import Camera
        from openpibo.vision_detect import Detect
        items = Detect().detect_object(Camera().read())
        return korean_names(i.get("name") if isinstance(i, dict) else i for i in items or [])


def act(robot: Pibo, name: str, args: dict, said: str) -> str:
    """모델이 부른 도구 하나를 실행하고 할 말을 돌려줍니다. said: 모델이 호출 앞에 쓴 말."""
    if check_call(name, args):
        return said or "미안해요, 잘 못 알아들었어요. 한 번만 다시 말해 줄래요?"
    t = time.time()
    try:
        if name == "move" and not robot.can_move:
            return robot.cant_move          # 모델의 "춤출게요!"는 버림
        if name == "light":
            robot.light(args["color"])
            done = said or ("불빛을 껐어요." if args["color"] == "끄기" else "불빛을 바꿨어요.")
        elif name == "move":
            robot.move(args["action"])
            done = said or "움직여 볼게요!"
        else:                               # look
            seen = robot.look()
            done = (f"{said} " if said else "") + (
                "앞에 아무것도 안 보여요." if seen == "아무것도 안 보임" else
                f"앞에 {seen}{'이' if has_batchim(seen) else '가'} 보여요.")
    except Exception as exc:  # noqa: BLE001 — 로봇이 멈추지 않게
        print(f"  [도구] {name}({args}) 실패: {exc}", flush=True)
        return "미안해요, 지금은 그걸 하지 못했어요."
    print(f"  [도구] {name}({args}) ({time.time() - t:.1f}초)", flush=True)
    return done


# ------------------------------------------------------------ 모델

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
    if t:   # 서버가 다시 읽은 토큰 수: 앞부분을 기억하지 못하면 크게 나옵니다
        print(f"  (읽기 {t.get('prompt_n', 0)}토큰 {t.get('prompt_ms', 0) / 1000:.1f}초, "
              f"말하기 {t.get('predicted_n', 0)}토큰 {t.get('predicted_ms', 0) / 1000:.1f}초)",
              flush=True)
    return data["choices"][0]["message"]


def window(history: list) -> list:
    """최근 HISTORY번의 주고받기 (학생 말에서 시작)."""
    starts = [i for i, m in enumerate(history) if m["role"] == "user"]
    return history[starts[-HISTORY]:] if len(starts) > HISTORY else history[:]


def same(text: str) -> str:
    """대답을 견줄 모양: 띄어쓰기와 문장부호를 뺌."""
    return re.sub(r"[\s.,!?~…'\"]", "", text or "")


#: 멈추라는 말은 어디에 있어도 바로 멈춥니다 (안전, 모델에 묻지 않음)
STOP = re.compile(r"멈춰|멈추라|정지|스톱|그만\s*움직|stop|\bfreeze\b", re.IGNORECASE)


def turn(url: str, history: list, text: str, robot: Pibo, system: str, tools: bool = True) -> str:
    """학생 말 하나 → 할 말. tools가 거짓("대화만 하자")이면 도구 목록 없이 대화만.
    기억(history)에는 학생 말과 실제로 한 말만 남깁니다 (학습 때의 앞 대화와 같은 모양)."""
    history.append({"role": "user", "content": text})
    if STOP.search(text):
        if robot.can_move:
            robot.stop()
        said = "네, 멈출게요." if lang_of(text) == "ko" else "Okay, I'll stop."
        history.append({"role": "assistant", "content": said})
        return said
    msgs = [{"role": "system", "content": system}] + window(history)
    msg = ask(url, msgs, tools)
    content = (msg.get("content") or "").strip()
    calls = calls_of(msg) if tools else []
    if calls:
        said = act(robot, *calls[0], content)
    else:
        said = content
        before = {same(m["content"]) for m in history if m["role"] == "assistant"}
        if same(said) in before:             # 앞에서 한 말 그대로: 한 번만 더 다르게
            print("  (앞과 같은 대답이라 다시 물음)", flush=True)
            again = ask(url, msgs, tools, temperature=1.0)
            if not calls_of(again):
                said = (again.get("content") or "").strip() or said
    history.append({"role": "assistant", "content": said})
    return said


def warm_up(url: str, system: str, tools: bool) -> None:
    """시스템 글과 도구 목록을 서버가 미리 읽어 두게 합니다 (파이보에서 수십 초). 다음부터는
    서버가 앞부분을 기억해 새로 한 말만 읽습니다."""
    try:
        urllib.request.urlopen(f"{url}/health", timeout=10).read()
    except OSError as exc:
        raise SystemExit(f"llama-server에 연결할 수 없어요 ({url}): 서버가 떠 있는지, 포트가 맞는지 "
                         f"확인하세요 ({exc})") from None
    print("  (준비 중)", flush=True)
    t = time.time()
    ask(url, [{"role": "system", "content": system}, {"role": "user", "content": "안녕"}],
        tools, max_tokens=1, timeout=900)
    print(f"  (준비 {time.time() - t:.1f}초)", flush=True)


#: "대화만 하자" 같은 말로 도구를 끄고 켭니다
SWITCH_REPLY = {True: "좋아요, 이제 다시 움직이고 불빛도 바꿀 수 있어요!",
                False: "좋아요, 지금부터는 대화만 할게요."}


def tool_switch(text: str, on: bool) -> bool:
    """학생 말에 따라 바뀐 도구 사용 여부 (띄어쓰기는 보지 않음)."""
    t = re.sub(r"\s", "", text)
    if re.search(r"(쓰지|사용하지|움직이지)(마|말)|대화만|말로만|얘기만|이야기만|가만히있", t):
        return False
    if re.search(r"움직여도|(이제|다시)움직|도구.*(쓰|써|사용)", t):
        return True
    return on


def main(robot_class=Pibo):
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:50020")
    p.add_argument("--no-tools", action="store_true", help="도구 없이 대화만")
    p.add_argument("--fake", action="store_true", help="openpibo 없이 동작을 한 척만")
    p.add_argument("--history", type=int, default=HISTORY,
                   help="기억하는 주고받기 수 (지금 말 포함)")
    p.add_argument("--name", help=f"기기 이름 (시스템 글에 들어감, 기본: {robot_class.name})")
    p.add_argument("--about", help="기기 소개 (예: '코딩 로봇, 서큘러스에서 만듦, 하얀색')")
    args = p.parse_args()
    globals()["HISTORY"] = max(1, args.history)
    robot = robot_class(fake=args.fake)
    name = args.name or robot.name
    system = system_for(name, args.about)
    use_tools = not args.no_tools
    warm_up(args.url, system, use_tools)
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
        if not args.no_tools:
            was, use_tools = use_tools, tool_switch(text, use_tools)
            if was != use_tools:
                # 끄고 켜는 말은 모델에 보내지 않고 정해진 말로 대답하고 기억을 비웁니다
                history.clear()
                print(f"  (도구 {'켬' if use_tools else '끔'})", flush=True)
                speak(name, SWITCH_REPLY[use_tools])
                continue
        t = time.time()
        try:
            speak(name, turn(args.url, history, text, robot, system, use_tools))
        except OSError as exc:               # 서버 오류(HTTPError 포함)·연결 끊김: 앱은 계속
            body = exc.read().decode("utf-8", "ignore")[:300] if hasattr(exc, "read") else ""
            print(f"  (서버 오류: {exc} {body})", flush=True)
            history.clear()
            speak(name, "잠깐 문제가 생겼어요. 한 번만 다시 말해 줄래요?")
        print(f"  (모두 {time.time() - t:.1f}초)", flush=True)


if __name__ == "__main__":
    main()
