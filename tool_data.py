"""도구(불빛, 보기, 움직이기)를 부르는 학습 대화를 정해 둔 말 모양으로 만듭니다 (data/tools.jsonl).

    python tool_data.py --out data/tools.jsonl --n 4000 --chat data/pibo.jsonl
    python tool_data.py --out data/heldout-tools.jsonl --n 300 --seed 99 --chat data/heldout.jsonl

선생 모델 없이 사람이 쓴 말 모양을 조합합니다: 도구 호출은 틀리면 안 되고, 말 모양만 다양하면
되기 때문입니다. 대화마다 tools.TOOLS(모든 기기 공통)를 함께 보여 줍니다.

- 불빛 켜기·색 바꾸기, 끄기, 앞에서 불빛 이야기를 한 뒤 "이번엔 파란색으로" (이어서 시키기)
- 움직이기 (파이브레인에서도 모델은 부릅니다. 못 움직인다는 말은 앱이 함)
- 보기 ("앞에 뭐 있어?")
- 부르면 안 되는 말: "난 빨간색이 좋아", "어제 춤 학원 갔어" 같은 이야기 속 낱말 → 대화만
- 시스템 글은 모든 기기 공통(tools.SYSTEM, --lang en이면 tools.SYSTEM_EN과 영어 말 모양). 학생은 기기 이름(파이보, 파이브레인 …)으로 부르기도 함
- --chat을 주면 그 대화의 앞부분 한두 번을 붙여 대화 중간에 시키는 모양도 만듭니다

모델의 말은 호출 앞의 짧은 말("빨간색으로 바꿀게요!")이고, 앱이 그 말을 하고 도구를 실행합니다.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from persona import CALLS, rename_text
from tools import ACTIONS, SYSTEMS, has_batchim

#: 색 이름 → (꾸미는 말, 색 낱말, ~게 꼴 또는 None)
FORMS = {
    "빨강": ("빨간", "빨간색", "빨갛게"), "주황": ("주황", "주황색", None),
    "노랑": ("노란", "노란색", "노랗게"), "초록": ("초록", "초록색", None),
    "파랑": ("파란", "파란색", "파랗게"), "하늘": ("하늘색", "하늘색", None),
    "보라": ("보라", "보라색", None), "분홍": ("분홍", "분홍색", None),
    "하양": ("하얀", "하얀색", "하얗게"),
}
LIGHTS = ["불", "불빛", "LED", "눈"]
ENDS = ["줘", "줘", "줄래?", "주세요", "봐", "줄 수 있어?"]
PREFIXES = ["", "", "", "", "{call} ", "음 ", "있잖아 ", "저기 "]

LIGHT_ON = [
    "{adj} 불 켜 {end}", "{light}{eul} {noun}으로 바꿔 {end}", "{light} {noun}으로 해 {end}",
    "{noun}으로 {light} 켜 {end}", "{light} {advx} 해 {end}", "{noun} 불빛 보고 싶어",
    "{light} 색 {noun}으로 바꿔 {end}", "{adj} 불빛 켜 줄래?", "{noun} {light} 켜 {end}",
    "{light} 좀 {advx} 바꿔 봐", "{noun} 불 켜 볼래?",
]
LIGHT_ON_SAY = ["{noun}으로 바꿀게요!", "좋아요, {noun} 불빛이에요.", "{adj} 불빛을 켤게요.",
                "네, {noun}으로 해 볼게요!", "{noun} 어때요?"]
#: 색 없이 켜 달라는 말 → 하양
PLAIN_ON = ["불 켜 {end}", "{light} 켜 {end}", "너무 어두워, 불 좀 켜 {end}", "환하게 불 켜 {end}",
            "불빛 좀 켜 볼래?"]
PLAIN_ON_SAY = ["불빛을 켤게요!", "네, 환하게 켤게요.", "하얀 불빛을 켤게요."]
LIGHT_OFF = ["불 꺼 {end}", "{light} 꺼 {end}", "불빛 좀 꺼 {end}", "이제 불 꺼도 돼", "불 좀 끌래?",
             "{light} 그만 켜도 돼"]
LIGHT_OFF_SAY = ["불빛을 끌게요.", "네, 끌게요.", "알겠어요, 불빛을 꺼요."]
#: 앞에서 불빛을 바꾼 뒤 불빛이라는 말 없이 다시 시킴
FOLLOW = ["이번엔 {noun}으로", "{noun}으로 바꿔 {end}", "음 {noun}이 더 좋겠다", "아니 {noun}으로 해 {end}",
          "{noun}으로도 해 볼래?"]

MOVE = {
    "인사": ["인사해 {end}", "인사 좀 해 {end}", "친구한테 인사해 {end}", "안녕 하고 인사해 볼래?"],
    "춤": ["춤춰 {end}", "춤 좀 춰 {end}", "신나는 춤 보여 {end}", "춤 출 수 있어? 보여 줘",
          "노래 나오니까 춤춰 봐"],
    "손흔들기": ["손 흔들어 {end}", "손 좀 흔들어 봐", "바이바이 손 흔들어 줘", "나한테 손 흔들어 줄래?"],
    "박수": ["박수 쳐 {end}", "박수 좀 쳐 봐", "잘했으면 박수 쳐 줘", "짝짝짝 박수!"],
    "하이파이브": ["하이파이브 하자", "하이파이브 해 {end}", "우리 하이파이브!"],
    "끄덕이기": ["고개 끄덕여 {end}", "맞으면 고개 끄덕여 봐", "고개 끄덕끄덕 해 줘"],
    "도리도리": ["고개 저어 {end}", "도리도리 해 봐", "아니면 고개 저어 줘"],
    "앞으로": ["앞으로 가 {end}", "앞으로 와 {end}", "이리 와 봐", "조금 앞으로 와 줄래?"],
    "뒤로": ["뒤로 가 {end}", "뒤로 물러나 {end}", "조금 뒤로 가 줄래?"],
    "왼쪽": ["왼쪽으로 돌아 {end}", "왼쪽으로 돌아 볼래?", "왼쪽으로 돌자"],
    "오른쪽": ["오른쪽으로 돌아 {end}", "오른쪽으로 돌아 볼래?", "오른쪽으로 돌자"],
}
MOVE_SAY = {
    "인사": ["안녕하세요! 반가워요.", "인사할게요, 안녕하세요!"],
    "춤": ["신나게 춤춰 볼게요!", "좋아요, 춤출게요!", "제 춤 실력 보여 줄게요!"],
    "손흔들기": ["손 흔들게요, 안녕!", "네, 손을 흔들게요."],
    "박수": ["짝짝짝, 박수 칠게요!", "박수 쳐 줄게요!"],
    "하이파이브": ["하이파이브!", "좋아요, 하이파이브해요!"],
    "끄덕이기": ["끄덕끄덕할게요.", "네, 고개를 끄덕일게요."],
    "도리도리": ["도리도리할게요.", "고개를 저어 볼게요."],
    "앞으로": ["앞으로 갈게요.", "네, 앞으로 가 볼게요."],
    "뒤로": ["뒤로 갈게요.", "조금 뒤로 물러날게요."],
    "왼쪽": ["왼쪽으로 돌게요.", "네, 왼쪽으로 돌아 볼게요."],
    "오른쪽": ["오른쪽으로 돌게요.", "네, 오른쪽으로 돌아 볼게요."],
}
assert set(MOVE) == set(MOVE_SAY) == set(ACTIONS)

LOOK = ["앞에 뭐 있어?", "뭐가 보여?", "지금 뭐 보여?", "이거 뭔지 알아?", "내가 뭐 들고 있게?",
        "카메라로 봐 {end}", "주변에 뭐 있는지 봐 {end}", "이거 뭐야? 봐 봐", "내 앞에 뭐가 있는지 맞혀 봐"]
LOOK_SAY = ["한번 볼게요.", "잠깐만요, 볼게요.", "어디 볼까요?", "카메라로 볼게요."]

#: 도구를 부르면 안 되는 말 (이야기 속 낱말) → 대화로 대답
NOT_CALLS = [
    ("난 {noun}이 제일 좋아", ["{noun} 예쁘죠! {noun} 물건 중에 제일 아끼는 게 있어요?",
                            "저도 {noun} 좋아요. 왜 {noun}이 좋아요?"]),
    ("어제 {adj} 옷 입었어", ["{adj} 옷 멋졌겠어요! 어디 갈 때 입었어요?"]),
    ("내 동생은 맨날 {noun} 크레파스만 써", ["{noun}을 정말 좋아하나 봐요. 동생은 뭘 자주 그려요?"]),
    ("어제 춤 학원 갔어", ["춤 학원 재미있었겠어요! 어떤 춤 배웠어요?"]),
    ("친구들이랑 춤 연습했어", ["와, 무슨 노래로 연습했어요?"]),
    ("어제 본 영화에 로봇이 춤추더라", ["로봇이 춤추는 장면 재미있었겠어요! 무슨 영화였어요?"]),
    ("밤에 불 켜고 잤어", ["불 켜고 자면 잠이 잘 안 올 수도 있어요. 잘 잤어요?"]),
    ("눈이 너무 피곤해", ["눈이 피곤하면 잠깐 먼 곳을 보면서 쉬어 봐요."]),
    ("눈 감고 소원 빌었어", ["어떤 소원 빌었어요? 꼭 이뤄졌으면 좋겠어요."]),
    ("앞으로 열심히 공부할 거야", ["멋져요! 어떤 과목부터 해 볼 거예요?"]),
    ("체육 시간에 박수 치기 게임 했어", ["재미있었겠어요! 누가 이겼어요?"]),
    ("오늘 뭐 봤는지 알아?", ["음, 뭐 봤어요? 궁금해요."]),
    ("어두운 데서 핸드폰 보면 눈 나빠져?", ["어두운 데서 오래 보면 눈이 쉽게 피곤해져요. 불을 켜고 보는 게 좋아요."]),
    ("우리 반 친구가 인사를 잘 안 해", ["조금 서운했겠어요. 그 친구가 수줍음이 많은 걸까요?"]),
]

#: 만들 대화 종류와 비율
KINDS = [("light", 30), ("plain_on", 5), ("light_off", 7), ("follow", 8), ("move", 25), ("look", 12),
         ("not_call", 13)]


def eul(word: str) -> str:
    """을/를 (영어 낱말은 읽는 소리로: LED → 엘이디 → 를)."""
    return "를" if word == "LED" or not has_batchim(word) else "을"


def said(rng: random.Random, template: str, color: str | None = None) -> str:
    light = rng.choice(LIGHTS)
    slots = {"end": rng.choice(ENDS), "light": light, "eul": eul(light),
             "call": rng.choice([c.strip() for c in CALLS if c])}
    if color:
        adj, noun, adv = FORMS[color]
        slots.update(adj=adj, noun=noun, advx=adv or noun + "으로")
    text = template.format(**slots)
    return (rng.choice(PREFIXES).format(**slots) + text).strip()


def assistant(content: str, name: str | None = None, **args) -> dict:
    m = {"role": "assistant", "content": content}
    if name:
        m["tool_calls"] = [{"type": "function", "function": {"name": name, "arguments": args}}]
    return m


def turn_for(kind: str, rng: random.Random) -> tuple[list[dict], str]:
    """(그 종류의 마지막 주고받기 메시지들, 주제)."""
    if kind == "light":
        color = rng.choice(list(FORMS))
        return [{"role": "user", "content": said(rng, rng.choice(LIGHT_ON), color)},
                assistant(said_line(rng, LIGHT_ON_SAY, color), "light", color=color)], "불빛"
    if kind == "plain_on":
        return [{"role": "user", "content": said(rng, rng.choice(PLAIN_ON), None)},
                assistant(rng.choice(PLAIN_ON_SAY), "light", color="하양")], "불빛"
    if kind == "light_off":
        return [{"role": "user", "content": said(rng, rng.choice(LIGHT_OFF), None)},
                assistant(rng.choice(LIGHT_OFF_SAY), "light", color="끄기")], "불빛 끄기"
    if kind == "follow":
        first, second = rng.sample(list(FORMS), 2)
        before = [{"role": "user", "content": said(rng, rng.choice(LIGHT_ON), first)},
                  {"role": "assistant", "content": said_line(rng, LIGHT_ON_SAY, first)}]
        return before + [{"role": "user", "content": said(rng, rng.choice(FOLLOW), second)},
                         assistant(said_line(rng, LIGHT_ON_SAY, second), "light", color=second)], \
            "불빛 이어서"
    if kind == "move":
        action = rng.choice(list(MOVE))
        return [{"role": "user", "content": said(rng, rng.choice(MOVE[action]), None)},
                assistant(rng.choice(MOVE_SAY[action]), "move", action=action)], "움직이기"
    if kind == "look":
        return [{"role": "user", "content": said(rng, rng.choice(LOOK), None)},
                assistant(rng.choice(LOOK_SAY), "look")], "보기"
    user, replies = rng.choice(NOT_CALLS)
    color = rng.choice(list(FORMS))
    return [{"role": "user", "content": said(rng, user, color)},
            {"role": "assistant", "content": said_line(rng, replies, color)}], "부르지 않음"


def said_line(rng: random.Random, lines: list[str], color: str | None) -> str:
    line = rng.choice(lines)
    if color is None:
        return line
    adj, noun, _ = FORMS[color]
    return line.format(adj=adj, noun=noun)


def context(rng: random.Random, chats: list[dict]) -> list[dict]:
    """다른 대화의 앞부분 0~2번 (대화 중간에 시키는 모양)."""
    k = rng.choice([0, 0, 1, 1, 2])
    if not chats or not k:
        return []
    msgs = [m for m in rng.choice(chats)["messages"] if m["role"] in ("user", "assistant")
            and not m.get("tool_calls")][:2 * k]
    if len(msgs) % 2:
        msgs = msgs[:-1]
    return [dict(m, content=rename_text(m["content"])) if m["role"] == "assistant" else m
            for m in msgs]


# ------------------------------------------------------------- 영어 (themaker-en)
# 도구 목록은 한국어 모델과 같은 tools.TOOLS (색·동작 값도 같은 한국어 이름): 기기 앱을 그대로 씀

COLORS_EN = {"빨강": "red", "주황": "orange", "노랑": "yellow", "초록": "green", "파랑": "blue",
             "하늘": "sky blue", "보라": "purple", "분홍": "pink", "하양": "white"}
ENDS_EN = ["", "", " please", ", please", "!"]
PREFIXES_EN = ["", "", "", "", "{call}, ", "Hey, ", "Um, ", "Okay, "]
CALLS_EN = ["Pibo", "PiBrain", "TheMaker"]

LIGHT_ON_EN = ["Turn on the {color} light{end}", "Can you make the light {color}?",
               "Change the light to {color}{end}", "Make your light {color}{end}",
               "I want to see a {color} light", "Switch the light to {color}{end}",
               "Show me a {color} light{end}", "Can you turn on a {color} light?",
               "Could you change your light to {color}?"]
LIGHT_ON_SAY_EN = ["Okay, changing it to {color}!", "Here's a {color} light.", "Sure, {color} it is!",
                   "Turning on the {color} light."]
PLAIN_ON_EN = ["Turn on the light{end}", "It's too dark, turn on the light{end}",
               "Can you turn your light on?", "Light on{end}", "Make it bright{end}"]
PLAIN_ON_SAY_EN = ["Turning on my light!", "Sure, here's a bright light.", "Okay, white light on."]
LIGHT_OFF_EN = ["Turn off the light{end}", "Light off{end}", "You can turn off the light now",
                "Can you switch off your light?", "Turn your light off{end}"]
LIGHT_OFF_SAY_EN = ["Turning off the light.", "Okay, light off.", "Sure, I'll turn it off."]
FOLLOW_EN = ["Now make it {color}", "Change it to {color}{end}", "Hmm, {color} would be better",
             "No, make it {color}{end}", "Can you try {color} too?"]

MOVE_EN = {
    "인사": ["Say hello{end}", "Can you say hi to my friend?", "Say hi to everyone{end}"],
    "춤": ["Dance for me{end}", "Can you dance?", "Show me a fun dance{end}", "Music is on, let's dance!"],
    "손흔들기": ["Wave your hand{end}", "Wave goodbye to me", "Can you wave at me?"],
    "박수": ["Clap your hands{end}", "Give me a clap!", "Can you clap for me?"],
    "하이파이브": ["High five!", "Give me a high five{end}", "Let's do a high five"],
    "끄덕이기": ["Nod your head{end}", "Nod if you agree", "Can you nod for me?"],
    "도리도리": ["Shake your head{end}", "Shake your head if you don't agree", "Can you shake your head?"],
    "앞으로": ["Move forward{end}", "Come here{end}", "Come a little closer", "Can you move forward?"],
    "뒤로": ["Move back{end}", "Step back a little{end}", "Can you go backward?"],
    "왼쪽": ["Turn left{end}", "Can you turn to the left?", "Let's turn left"],
    "오른쪽": ["Turn right{end}", "Can you turn to the right?", "Let's turn right"],
}
MOVE_SAY_EN = {
    "인사": ["Hello, everyone! Nice to meet you.", "Hi there!"],
    "춤": ["Let's dance!", "Okay, watch my dance!"],
    "손흔들기": ["Bye bye!", "Sure, waving now!"],
    "박수": ["Clap clap clap!", "Great job, here's a clap!"],
    "하이파이브": ["High five!", "Yay, high five!"],
    "끄덕이기": ["Okay, I'm nodding.", "Nod nod!"],
    "도리도리": ["Okay, shaking my head.", "Shake shake!"],
    "앞으로": ["Moving forward.", "Okay, coming closer."],
    "뒤로": ["Moving back.", "Okay, stepping back."],
    "왼쪽": ["Turning left.", "Okay, turning to the left."],
    "오른쪽": ["Turning right.", "Okay, turning to the right."],
}
assert set(MOVE_EN) == set(MOVE_SAY_EN) == set(ACTIONS)
LOOK_EN = ["What's in front of you?", "What do you see?", "What can you see right now?",
           "Do you know what this is?", "Guess what I'm holding", "Look with your camera{end}",
           "Look around{end}", "What is this? Take a look"]
LOOK_SAY_EN = ["Let me take a look.", "Hold on, let me see.", "Let me look with my camera."]
NOT_CALLS_EN = [
    ("I like {color} the most", ["{Color} is pretty! Do you have a favorite {color} thing?",
                                 "I like {color} too. Why do you like it?"]),
    ("I wore a {color} shirt yesterday", ["That sounds nice! Where did you go?"]),
    ("My little brother only uses {color} crayons", ["He must really love {color}. What does he like to draw?"]),
    ("I went to dance class yesterday", ["That sounds fun! What dance did you learn?"]),
    ("I practiced dancing with my friends", ["Cool! What song did you dance to?"]),
    ("A robot danced in the movie I saw", ["That sounds fun! What movie was it?"]),
    ("I slept with the light on last night", ["It can be hard to sleep with the light on. Did you sleep well?"]),
    ("My eyes are so tired", ["Try looking far away for a little while to rest your eyes."]),
    ("I closed my eyes and made a wish", ["What did you wish for? I hope it comes true."]),
    ("I'm going to study hard from now on", ["That's great! Which subject will you start with?"]),
    ("We played a clapping game in P.E. class", ["Sounds fun! Who won?"]),
    ("Do you know what I saw today?", ["Hmm, what did you see? I'm curious."]),
    ("My friend in class doesn't say hello to me", ["That must feel a little sad. Maybe your friend is just shy?"]),
    ("I sat in the back of the class today", ["How was class today? Did anything fun happen?"]),
]


def said_en(rng: random.Random, template: str, color: str | None = None) -> str:
    slots = {"end": rng.choice(ENDS_EN), "call": rng.choice(CALLS_EN)}
    if color:
        slots.update(color=COLORS_EN[color], Color=COLORS_EN[color].capitalize())
    text = template.format(**slots)
    prefix = rng.choice(PREFIXES_EN).format(**slots)
    return prefix + (text[0].lower() + text[1:] if prefix and not text.startswith("I") else text)


def turn_for_en(kind: str, rng: random.Random) -> tuple[list[dict], str]:
    """turn_for와 같은 종류를 영어로."""
    if kind == "light":
        color = rng.choice(list(COLORS_EN))
        return [{"role": "user", "content": said_en(rng, rng.choice(LIGHT_ON_EN), color)},
                assistant(rng.choice(LIGHT_ON_SAY_EN).format(color=COLORS_EN[color]), "light",
                          color=color)], "불빛"
    if kind == "plain_on":
        return [{"role": "user", "content": said_en(rng, rng.choice(PLAIN_ON_EN))},
                assistant(rng.choice(PLAIN_ON_SAY_EN), "light", color="하양")], "불빛"
    if kind == "light_off":
        return [{"role": "user", "content": said_en(rng, rng.choice(LIGHT_OFF_EN))},
                assistant(rng.choice(LIGHT_OFF_SAY_EN), "light", color="끄기")], "불빛 끄기"
    if kind == "follow":
        first, second = rng.sample(list(COLORS_EN), 2)
        before = [{"role": "user", "content": said_en(rng, rng.choice(LIGHT_ON_EN), first)},
                  {"role": "assistant", "content": rng.choice(LIGHT_ON_SAY_EN).format(color=COLORS_EN[first])}]
        return before + [{"role": "user", "content": said_en(rng, rng.choice(FOLLOW_EN), second)},
                         assistant(rng.choice(LIGHT_ON_SAY_EN).format(color=COLORS_EN[second]),
                                   "light", color=second)], "불빛 이어서"
    if kind == "move":
        action = rng.choice(list(MOVE_EN))
        return [{"role": "user", "content": said_en(rng, rng.choice(MOVE_EN[action]))},
                assistant(rng.choice(MOVE_SAY_EN[action]), "move", action=action)], "움직이기"
    if kind == "look":
        return [{"role": "user", "content": said_en(rng, rng.choice(LOOK_EN))},
                assistant(rng.choice(LOOK_SAY_EN), "look")], "보기"
    user, replies = rng.choice(NOT_CALLS_EN)
    color = rng.choice(list(COLORS_EN))
    word = COLORS_EN[color]
    return [{"role": "user", "content": said_en(rng, user, color)},
            {"role": "assistant", "content": rng.choice(replies).format(color=word, Color=word.capitalize())}], \
        "부르지 않음"


def make(n: int, seed: int, chats: list[dict], lang: str = "ko") -> list[dict]:
    rng = random.Random(seed)
    kinds = [k for k, _ in KINDS]
    weights = [w for _, w in KINDS]
    out, seen = [], set()
    while len(out) < n:
        kind = rng.choices(kinds, weights)[0]
        last, topic = turn_for_en(kind, rng) if lang == "en" else turn_for(kind, rng)
        msgs = [{"role": "system", "content": SYSTEMS[lang]}] + context(rng, chats) + last
        key = json.dumps(msgs, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append({"kind": "tool", "source": "tools", "topic": topic, "tools": True,
                    "messages": msgs, **({"lang": lang} if lang != "ko" else {})})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chat", help="앞부분을 빌려 올 대화 (없으면 도구 말만)")
    p.add_argument("--lang", choices=["ko", "en"], default="ko", help="en이면 영어 모델용")
    args = p.parse_args()
    chats = []
    if args.chat and Path(args.chat).exists():
        chats = [json.loads(x) for x in Path(args.chat).read_text().splitlines() if x.strip()]
    rows = make(args.n, args.seed, chats, args.lang)
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    count: dict = {}
    for r in rows:
        count[r["topic"]] = count.get(r["topic"], 0) + 1
    print(f"{args.out}: {len(rows)}개 (" + ", ".join(f"{k} {v}" for k, v in count.items()) + ")",
          flush=True)


if __name__ == "__main__":
    main()
