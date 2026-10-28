"""대화 모델(themaker)의 시스템 글과, 모든 기기 공통의 도구 목록.

모델은 하나이고 도구도 모든 기기에 같은 목록을 보냅니다 (불빛, 보기, 움직이기). 학생 말을 보고
도구를 부를지는 모델이 정하고, 기기 앱이 그 호출을 자기 함수로 실행합니다:

- 파이보 (pibo_chat.py): 불빛 = Device.eye_on/eye_off, 움직이기 = Motion.set_motion,
  보기 = Camera + Detect
- 파이브레인 (pibrain_chat.py): 불빛 = DeviceByPiBrain.led_on/led_off, 보기 = Camera + Detect,
  움직이기는 몸이 없어 실행하지 않고 못 움직인다고 말함

시스템 글(SYSTEM)도 모든 기기에 같습니다: 모델은 기기 이름이나 기기 사양을 말하지 않고, 학생들을
위한 교육용 AI 더메이커로 이야기합니다. 시스템 글과 도구 목록은 학습 때와 글자까지 같아야
합니다 (모델이 그 모양을 보고 배움). 바꾸면 다시 학습합니다.
"""

from __future__ import annotations

import json

#: 불빛 색 (모델이 고르는 이름) → RGB. "끄기"는 불빛을 끔
COLORS = {
    "빨강": (255, 0, 0), "주황": (255, 120, 0), "노랑": (255, 220, 0), "초록": (0, 255, 0),
    "파랑": (0, 0, 255), "하늘": (0, 180, 255), "보라": (150, 0, 255), "분홍": (255, 100, 180),
    "하양": (255, 255, 255), "끄기": None,
}

#: 움직이기 (모델이 고르는 이름) → openpibo motion_db의 동작 이름 (여럿이면 하나를 골라 씀)
ACTIONS = {
    "인사": ["greeting"], "춤": ["dance1", "dance2", "dance3", "dance4", "dance5"],
    "손흔들기": ["wave1"], "박수": ["clapping1"], "하이파이브": ["hifive"],
    "끄덕이기": ["yes_h"], "도리도리": ["no_h"], "앞으로": ["forward1"], "뒤로": ["backward1"],
    "왼쪽": ["left"], "오른쪽": ["right"],
}

TOOLS = [
    {"type": "function", "function": {
        "name": "light",
        "description": "불빛 색을 바꾸거나 끕니다.",
        "parameters": {"type": "object", "properties": {
            "color": {"type": "string", "enum": list(COLORS)}}, "required": ["color"]}}},
    {"type": "function", "function": {
        "name": "look",
        "description": "카메라로 앞에 있는 것을 봅니다.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "move",
        "description": "몸을 움직입니다.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "enum": list(ACTIONS)}}, "required": ["action"]}}},
]

BY_NAME = {t["function"]["name"]: t["function"] for t in TOOLS}

#: 대화 모델의 이름 (모든 기기 공통)
ENGINE = "더메이커"
ENGINE_EN = "TheMaker"
#: 모든 기기 공통 시스템 글 (기기 앱, 웹 화면, 학습, 채점 모두 이것)
SYSTEM = ("너는 서큘러스에서 만든, 초등학생부터 고등학생까지 학생들과 이야기하는 교육용 AI 더메이커야. "
          "열네 살 친구처럼 밝고 친근하게, 존댓말로 짧게 말해. "
          "이모지나 기호는 쓰지 말고, 정말 모르는 건 모른다고 해.")
#: 영어 모델(themaker-en)의 시스템 글 (모든 기기 공통)
SYSTEM_EN = ("You are TheMaker, an educational AI made by Circulus that talks with students from "
             "elementary to high school. Talk like a bright, kind 14-year-old friend, in short and "
             "easy English. Don't use emojis or symbols, and if you really don't know something, "
             "say you don't know.")
#: 말마다의 시스템 글
SYSTEMS = {"ko": SYSTEM, "en": SYSTEM_EN}


def has_batchim(word: str) -> bool:
    """마지막 글자에 받침이 있는지 (한글이 아니면 없음으로)."""
    c = word.strip()[-1:] or "가"
    return "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 != 0


def parse_args(args) -> dict | None:
    """도구 호출의 인자: llama-server는 JSON 글로, 학습 데이터는 객체로 줍니다."""
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            return None
    return args if isinstance(args, dict) else None


def check_call(name: str, args) -> str | None:
    """잘못된 곳이 있으면 그 이유, 맞으면 None."""
    fn = BY_NAME.get(name)
    if fn is None:
        return f"없는 도구 {name}"
    args = parse_args(args)
    if args is None:
        return "인자가 JSON 객체가 아님"
    props = fn["parameters"].get("properties", {})
    for key in fn["parameters"].get("required", []):
        if key not in args:
            return f"{name}: {key} 빠짐"
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            return f"{name}: 모르는 인자 {key}"
        if "enum" in spec and value not in spec["enum"]:
            return f"{name}: {key}={value!r}는 목록에 없음"
    return None


def calls_of(message: dict) -> list[tuple[str, dict]]:
    """모델 대답(OpenAI 형식)의 도구 호출들 [(이름, 인자)]. 인자를 못 읽으면 {}."""
    out = []
    for c in message.get("tool_calls") or []:
        fn = c.get("function") or {}
        out.append((fn.get("name", ""), parse_args(fn.get("arguments")) or {}))
    return out
