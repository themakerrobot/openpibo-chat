"""파이보가 부를 수 있는 도구(함수) 목록과 그 검사.

학습 데이터, 채점, 파이보에서의 실행이 모두 이 목록 하나를 씁니다.
이름과 인자는 openpibo의 함수에 그대로 대응합니다
(Motion.set_motion, Device.eye_on, Device.eye_off, Device.get_battery,
Weather.search, 카메라 물체 찾기).
"""

from __future__ import annotations

import json

#: 아이와 놀 때 쓸 만한 동작만 (openpibo motion_db의 이름 그대로)
MOTIONS = {
    "greeting": "인사하기",
    "welcome": "반갑게 맞이하기",
    "bow": "고개 숙여 인사",
    "wave1": "손 흔들기",
    "hifive": "하이파이브",
    "handshaking": "악수",
    "clapping1": "박수",
    "cheer1": "응원하기",
    "dance1": "춤 1",
    "dance2": "춤 2",
    "dance3": "춤 3",
    "dance4": "춤 4",
    "dance5": "춤 5",
    "happy1": "기뻐하기",
    "sad1": "슬퍼하기",
    "think1": "생각하기",
    "yes_h": "고개 끄덕이기",
    "no_h": "고개 젓기",
    "look_l": "왼쪽 보기",
    "look_r": "오른쪽 보기",
    "lookup": "위 보기",
    "forward1": "앞으로 가기",
    "backward1": "뒤로 가기",
    "left": "왼쪽으로 돌기",
    "right": "오른쪽으로 돌기",
    "sleep": "잠자기",
    "wake_up1": "잠에서 깨기",
    "stop": "멈추기",
}

#: 데이터를 만들 때 알려 주는 색 (모델은 RGB 숫자로 부릅니다)
COLORS = {
    "빨강": (255, 0, 0), "주황": (255, 120, 0), "노랑": (255, 220, 0), "초록": (0, 255, 0),
    "파랑": (0, 0, 255), "하늘": (0, 180, 255), "남색": (20, 30, 160), "보라": (150, 0, 255),
    "분홍": (255, 100, 180), "하양": (255, 255, 255), "연두": (150, 255, 50),
    "청록": (0, 200, 160), "갈색": (140, 70, 20),
}

TOOLS = [
    {"type": "function", "function": {
        "name": "set_motion",
        "description": "파이보가 몸을 움직입니다 (춤, 인사, 박수, 고개 끄덕이기, 이동 등).",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "enum": list(MOTIONS),
                     "description": "동작 이름: " + ", ".join(f"{k}({v})" for k, v in
                                                          MOTIONS.items())}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "eye_on",
        "description": "파이보 눈의 불빛 색을 바꿉니다. 색은 0~255 사이의 빨강, 초록, 파랑 값입니다.",
        "parameters": {"type": "object", "properties": {
            "r": {"type": "integer", "minimum": 0, "maximum": 255},
            "g": {"type": "integer", "minimum": 0, "maximum": 255},
            "b": {"type": "integer", "minimum": 0, "maximum": 255}},
            "required": ["r", "g", "b"]}}},
    {"type": "function", "function": {
        "name": "eye_off",
        "description": "파이보 눈의 불빛을 끕니다.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "find_objects",
        "description": "카메라로 파이보 앞에 무엇이 있는지 찾아봅니다.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_battery",
        "description": "파이보의 배터리가 얼마나 남았는지 봅니다.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "지역의 오늘 날씨를 알아봅니다 (인터넷이 필요합니다).",
        "parameters": {"type": "object", "properties": {
            "region": {"type": "string", "description": "지역 이름, 예: 서울, 부산, 전국"}},
            "required": ["region"]}}},
]

BY_NAME = {t["function"]["name"]: t["function"] for t in TOOLS}


def tool_list(names=None, motions=None, device: str = "파이보") -> list:
    """기기에 있는 도구만 고른 목록. names: 도구 이름들 (None이면 모두), motions: set_motion에서
    쓸 수 있는 동작들 (None이면 모두), device: 설명에 쓰는 기기 이름. 셋 다 기본값이면 TOOLS 그대로.
    학습 때 일부만 보여 줘서, 모델이 외운 목록이 아니라 앱이 보낸 목록을 따르게 합니다."""
    if names is None and motions is None and device == "파이보":
        return TOOLS
    out = []
    for t in TOOLS:
        f = t["function"]
        if names is not None and f["name"] not in names:
            continue
        f = json.loads(json.dumps(f, ensure_ascii=False).replace("파이보", device))
        if f["name"] == "set_motion" and motions is not None:
            keep = [m for m in MOTIONS if m in motions]
            spec = f["parameters"]["properties"]["name"]
            spec["enum"] = keep
            spec["description"] = "동작 이름: " + ", ".join(f"{k}({MOTIONS[k]})" for k in keep)
        out.append({"type": "function", "function": f})
    return out


def has_batchim(word: str) -> bool:
    """마지막 글자에 받침이 있는지 (한글이 아니면 없음으로)."""
    c = word.strip()[-1:] or "가"
    return "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 != 0


def system_for(name: str = "파이보", about: str | None = None) -> str:
    """기기 이름과 소개를 넣은 시스템 글. 파이보에 소개가 없으면 SYSTEM 그대로.
    about: 그 기기에 대해 모델이 알아야 할 것 (예: "교육용 코딩 로봇, 서큘러스에서 만듦, 하얀색")."""
    text = SYSTEM.replace("로봇 파이보야.", f"로봇 {name}{'이야' if has_batchim(name) else '야'}.")
    if about:
        text += f" {name}에 대해: {about}"
    return text

SYSTEM = ("너는 초등학생부터 고등학생까지와 이야기하는 교육용 로봇 파이보야. "
          "친구처럼 친근한 존댓말로 짧게 말해. 이모지나 기호는 쓰지 말고, 정말 모르는 건 모른다고 해.")
#: 처음 만든 어린이(미취학~초저) 데이터에 들어 있는 시스템 글 (그 데이터와 채점용 질문에 그대로 남음)
SYSTEM_KIDS = ("너는 어린이와 이야기하는 교육용 로봇 파이보야. 친근한 존댓말로 한두 문장만 말해. "
               "이모지나 기호는 쓰지 말고, 모르는 건 모른다고 해.")


def check_call(name: str, args) -> str | None:
    """잘못된 곳이 있으면 그 이유, 맞으면 None."""
    fn = BY_NAME.get(name)
    if fn is None:
        return f"없는 도구 {name}"
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return "인자가 JSON이 아님"
    if not isinstance(args, dict):
        return "인자가 객체가 아님"
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
        if spec["type"] == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                return f"{name}: {key}는 정수"
            if not spec.get("minimum", value) <= value <= spec.get("maximum", value):
                return f"{name}: {key}={value} 범위 밖"
        if spec["type"] == "string" and not (isinstance(value, str) and value.strip()):
            return f"{name}: {key}는 글자"
    return None


def same_call(a: dict, b: dict) -> bool:
    """두 호출이 같은 일을 하는지 (색은 조금 달라도 같은 색으로 봅니다)."""
    if a.get("name") != b.get("name"):
        return False
    x, y = a.get("arguments") or {}, b.get("arguments") or {}
    if a["name"] == "eye_on":
        try:
            return all(abs(int(x[k]) - int(y[k])) <= 60 for k in "rgb")
        except (KeyError, TypeError, ValueError):
            return False
    if a["name"] == "get_weather":
        return str(x.get("region", "")).strip() == str(y.get("region", "")).strip()
    if a["name"] == "set_motion" and all(str(c.get("name", "")).startswith("dance")
                                         for c in (x, y)):
        return True                          # 춤은 몇 번 춤이든 같은 일
    return x == y
