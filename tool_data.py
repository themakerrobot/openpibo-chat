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
    # 아이들 말투 (claude_gen.py로 쓴 초·중 대화에 맞춤)
    "{noun}으로 해조", "{adj} 불 ㄱㄱ", "{noun} 불 켜봐 빨리", "{light} {noun}으로 바꿔줘 ㅋㅋ",
]
LIGHT_ON_SAY = ["{noun}으로 바꿀게요!", "좋아요, {noun} 불빛이에요.", "{adj} 불빛을 켤게요.",
                "네, {noun}으로 해 볼게요!", "{noun} 어때요?"]
#: 색 없이 켜 달라는 말 → 하양
PLAIN_ON = ["불 켜 {end}", "{light} 켜 {end}", "너무 어두워, 불 좀 켜 {end}", "환하게 불 켜 {end}",
            "불빛 좀 켜 볼래?", "깜깜해 불 켜줘", "불 켜조"]
PLAIN_ON_SAY = ["불빛을 켤게요!", "네, 환하게 켤게요.", "하얀 불빛을 켤게요."]
LIGHT_OFF = ["불 꺼 {end}", "{light} 꺼 {end}", "불빛 좀 꺼 {end}", "이제 불 꺼도 돼", "불 좀 끌래?",
             "{light} 그만 켜도 돼", "불 꺼조", "눈부셔 불 꺼줘"]
LIGHT_OFF_SAY = ["불빛을 끌게요.", "네, 끌게요.", "알겠어요, 불빛을 꺼요."]
#: 앞에서 불빛을 바꾼 뒤 불빛이라는 말 없이 다시 시킴
FOLLOW = ["이번엔 {noun}으로", "{noun}으로 바꿔 {end}", "음 {noun}이 더 좋겠다", "아니 {noun}으로 해 {end}",
          "{noun}으로도 해 볼래?", "이번엔 {noun} ㄱㄱ"]

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
        "카메라로 봐 {end}", "주변에 뭐 있는지 봐 {end}", "이거 뭐야? 봐 봐", "내 앞에 뭐가 있는지 맞혀 봐",
        "이거 뭐게?", "나 지금 뭐 들고있게 ㅋㅋ"]
LOOK_SAY = ["한번 볼게요.", "잠깐만요, 볼게요.", "어디 볼까요?", "카메라로 볼게요."]

#: 도구를 부르면 안 되는 말 (이야기 속 낱말) → 대화로 대답
NOT_CALLS = [
    ("난 {noun}이 제일 좋아", ["{noun} 예쁘죠! {noun} 물건 중에 제일 아끼는 게 있어요?",
                            "{noun}이 왜 좋아요? 제일 좋아하는 {noun} 물건도 알려 줘요."]),
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
    ("오늘 뭐 봤는지 알아?", ["뭐 봤는지 맞혀 볼게요. 힌트를 하나 줘요."]),
    ("어두운 데서 핸드폰 보면 눈 나빠져?", ["어두운 데서 오래 보면 눈이 쉽게 피곤해져요. 불을 켜고 보는 게 좋아요."]),
    ("우리 반 친구가 인사를 잘 안 해", ["조금 서운했겠어요. 그 친구가 수줍음이 많은 걸까요?"]),
    ("{noun} 색연필 잃어버렸어", ["아끼던 색연필이었나 봐요. 가방 구석도 한번 찾아봐요."]),
    ("체육대회 때 우리 반이 {noun} 팀이었어", ["{noun} 팀이었군요! 몇 등 했어요?"]),
    ("어제 불꽃놀이 봤어 완전 예뻤어", ["불꽃놀이라니 멋졌겠어요! 무슨 색이 제일 예뻤어요?"]),
    ("나 춤 진짜 못 춰 ㅋㅋ", ["못 춰도 신나게 추면 그게 제일 멋져요. 좋아하는 노래 있어요?"]),
    ("손 들고 발표했어", ["용기 내서 발표했군요! 떨리지 않았어요?"]),
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
                                 "What do you like most about {color}? Tell me more."]),
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
    ("Do you know what I saw today?", ["Hmm, let me guess. Give me one hint."]),
    ("My friend in class doesn't say hello to me", ["That must feel a little sad. Maybe your friend is just shy?"]),
    ("I sat in the back of the class today", ["How was class today? Did anything fun happen?"]),
]


# 아이 말투 (claude_gen.py로 쓴 초·중 대화에 맞춤): 줄임말과 소문자, 짧은 말
LIGHT_ON_EN += ["make it {color} pls", "{color} light pls", "can u make the light {color}?",
                "light {color} go go", "i want the {color} light", "turn the light {color} real quick"]
PLAIN_ON_EN += ["its dark turn the light on", "lights on pls", "it's too dark can u turn the light on?"]
LIGHT_OFF_EN += ["ok lights off now", "turn the light off pls", "ok u can turn it off", "too bright, light off"]
FOLLOW_EN += ["ok now {color}", "{color} next", "ooh try {color}"]
LOOK_EN += ["what do u see?", "guess what i have lol", "can u look and tell me what this is?"]
MOVE_EN["춤"] += ["dance dance dance", "can u dance for me pls"]
MOVE_EN["하이파이브"] += ["high five pls", "high five go"]
MOVE_EN["손흔들기"] += ["wave bye bye", "wave at my friend"]
MOVE_EN["박수"] += ["clap for me pls", "clap clap clap"]
MOVE_EN["앞으로"] += ["come here pls", "come closer"]
NOT_CALLS_EN += [
    ("my {color} pencil case broke", ["Oh no, that's sad. Can you fix it or get a new one?"]),
    ("our team color on field day was {color}", ["Fun! Did your team win?"]),
    ("i saw fireworks yesterday, they were so pretty", ["Fireworks are awesome! What colors did you see?"]),
    ("i cant dance at all lol", ["Dancing is fun even if you're not perfect. What song do you like?"]),
    ("i raised my hand in class today", ["Nice, that takes courage! Did you get to answer?"]),
    ("the light in my room is broken", ["That's annoying. Maybe tell a grown-up so they can fix it."]),
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


# ---- 도구 부르는 말 다음의 대답을 다양하게: 앞말 x 몸말 x 끝말을 곱해서 만든다 ----
def _mix(openers, bodies, closers):
    return [o + b + c for o in openers for b in bodies for c in closers]


_O = ["", "네, ", "좋아요, ", "알겠어요, ", "자, "]
_C = ["!", ".", "! 잘 봐요.", ". 다 하면 알려 줘요."]
LIGHT_ON_SAY = _mix(_O, ["{noun}으로 바꿀게요", "{adj} 불빛을 켤게요", "{noun} 불빛이에요", "{noun}으로 해 볼게요",
                         "불빛을 {noun}으로 바꿔요", "{adj} 불이 들어와요", "{noun}으로 켜 볼게요"],
                    ["!", ".", "! 어때요?", ". 잘 보이는지 봐 줘요.", "! 마음에 들면 말해 줘요."])
PLAIN_ON_SAY = _mix(_O, ["불빛을 켤게요", "환하게 켤게요", "하얀 불빛을 켤게요", "불을 켜요", "밝게 켜 볼게요",
                         "하얀 불이 들어와요"], ["!", ".", "! 어때요?", ". 이제 잘 보여요?"])
LIGHT_OFF_SAY = _mix(_O, ["불빛을 끌게요", "불을 꺼요", "불빛을 꺼 볼게요", "이제 불을 끌게요", "불을 끄고 쉬어요"],
                     [".", "!", ". 다시 켜고 싶으면 말해 줘요.", ". 필요하면 또 말해 줘요."])
_MB = {
    "인사": ["인사할게요", "손을 들어 인사할게요", "반갑게 인사할게요", "인사 동작을 해 볼게요"],
    "춤": ["춤출게요", "신나게 춤춰 볼게요", "몸을 흔들며 춤출게요", "춤 동작을 보여 줄게요", "음악에 맞춰 춤춰 볼게요"],
    "손흔들기": ["손을 흔들게요", "손을 흔들어 볼게요", "신나게 손을 흔들게요", "손 흔들며 인사할게요"],
    "박수": ["박수 칠게요", "짝짝짝 박수 쳐 볼게요", "힘차게 박수 칠게요", "박수를 쳐 볼게요"],
    "하이파이브": ["하이파이브해요", "손을 내밀어 하이파이브해 볼게요", "하이파이브 할게요", "같이 하이파이브해요"],
    "끄덕이기": ["고개를 끄덕일게요", "끄덕끄덕해 볼게요", "고개를 끄덕여 볼게요", "천천히 끄덕일게요"],
    "도리도리": ["고개를 저을게요", "도리도리해 볼게요", "고개를 저어 볼게요", "천천히 고개를 저어요"],
    "앞으로": ["앞으로 갈게요", "앞으로 가 볼게요", "조금 앞으로 나아갈게요", "천천히 앞으로 갈게요"],
    "뒤로": ["뒤로 갈게요", "뒤로 물러날게요", "조금 뒤로 가 볼게요", "천천히 뒤로 갈게요"],
    "왼쪽": ["왼쪽으로 돌게요", "왼쪽으로 돌아 볼게요", "몸을 왼쪽으로 돌릴게요", "왼쪽으로 방향을 바꿀게요"],
    "오른쪽": ["오른쪽으로 돌게요", "오른쪽으로 돌아 볼게요", "몸을 오른쪽으로 돌릴게요", "오른쪽으로 방향을 바꿀게요"],
}
MOVE_SAY = {a: _mix(_O, b, _C) for a, b in _MB.items()}
LOOK_SAY = _mix(["", "네, ", "잠깐만요, ", "좋아요, ", "자, "],
                ["볼게요", "한번 볼게요", "카메라로 볼게요", "앞을 살펴볼게요", "잠깐 둘러볼게요", "어디 한번 볼게요"],
                [".", "!", ". 조금만 기다려 줘요.", ". 잠깐만 가만히 있어 줘요."])
NOT_CALLS += [
    ("{noun} 풍선 터져서 울 뻔했어", ["그럴 만했어요. 풍선은 터지면 깜짝 놀라죠. 다시 하나 받았어요?"]),
    ("오늘 급식에 {noun} 젤리 나왔어", ["{noun} 젤리라니 눈에 띄었겠어요! 맛은 어땠어요?"]),
    ("우리 반 인사 당번이야", ["인사 당번이군요! 오늘 인사는 누구한테 제일 먼저 했어요?"]),
    ("나 어제 박수 많이 받았어", ["박수를 많이 받았다니 뿌듯했겠어요. 무슨 일로 받았어요?"]),
    ("춤 대회 나가기로 했어", ["대회에 나가기로 했군요! 어떤 노래로 나갈 거예요?"]),
    ("손 흔드는 인형 샀어", ["인형이 손을 흔들어 주는군요. 어디서 샀어요?"]),
    ("눈 오는 날 좋아", ["눈 오는 날은 하얀 세상이 되죠. 눈 오면 제일 먼저 뭐 하고 싶어요?"]),
    ("불조심 포스터 그렸어", ["불조심 포스터라니 좋은 숙제예요! 어떤 그림을 그렸어요?"]),
    ("앞으로 의사 되고 싶어", ["의사가 되고 싶군요! 왜 의사가 좋아 보였어요?"]),
    ("우리 집 강아지가 앞으로 달려 나왔어", ["강아지가 반겨 줬군요! 이름이 뭐예요?"]),
    ("뒤로 걷기 놀이 했어", ["뒤로 걸으면 균형 잡기가 어려워요. 몇 걸음까지 갔어요?"]),
    ("내 방 불이 나갔어", ["불이 나갔다니 불편하겠어요. 부모님께 말씀드려서 바꿔 달라고 해요."]),
    ("머리 위에서 전등이 깜빡거려", ["깜빡이는 전등은 눈이 피곤해져요. 어른께 말해서 고쳐 달라고 해요."]),
    ("{adj} 풍선 갖고 싶어", ["{adj} 풍선 예쁘겠어요! 어디에 달아 두고 싶어요?"]),
    ("고개 끄덕이다가 졸았어", ["졸릴 만큼 피곤했군요. 오늘은 일찍 자는 게 좋겠어요."]),
]
LOOK_SAY_EN = [o + b + c for o in ["", "Okay, ", "Sure, ", "Hold on, "] for b in
               ["let me take a look", "let me see", "let me look with my camera", "I'll take a quick look", "let me look around"]
               for c in [".", "!", ". Stay still for a second."]]
LOOK_SAY_EN = [x[0].upper() + x[1:] for x in LOOK_SAY_EN]


def _mix_en(openers, bodies, closers):
    return [(o + b if o else b[0].upper() + b[1:]) + c for o in openers for b in bodies for c in closers]


_OE = ["", "Okay, ", "Sure, ", "Got it, ", "Alright, "]
LIGHT_ON_SAY_EN = _mix_en(_OE, ["changing it to {color}", "here's a {color} light", "{color} it is",
                                "turning on the {color} light", "the light is {color} now", "let me make it {color}"],
                          ["!", ".", "! How does it look?", ". Tell me if you want another color."])
PLAIN_ON_SAY_EN = _mix_en(_OE, ["turning on the light", "here's a bright light", "white light on", "making it bright",
                                "the light is on now"], ["!", ".", "! Can you see better now?", ". Is that bright enough?"])
LIGHT_OFF_SAY_EN = _mix_en(_OE, ["turning off the light", "light off", "I'll turn it off", "the light is off now",
                                 "switching it off"], [".", "!", ". Tell me if you want it back on."])
_MBE = {
    "인사": ["saying hello", "I'll say hi", "waving hello", "let me say hello"],
    "춤": ["let's dance", "watch me dance", "I'll show you a dance", "dancing now", "time to dance"],
    "손흔들기": ["waving now", "I'll wave", "waving my hand", "bye bye, waving"],
    "박수": ["clapping now", "clap clap clap", "I'll clap for you", "here comes a clap"],
    "하이파이브": ["high five", "let's do a high five", "I'll give you a high five", "here's a high five"],
    "끄덕이기": ["I'm nodding", "nodding my head", "nod nod", "I'll nod"],
    "도리도리": ["I'm shaking my head", "shake shake", "shaking my head", "I'll shake my head"],
    "앞으로": ["moving forward", "coming closer", "I'll move forward", "stepping forward"],
    "뒤로": ["moving back", "stepping back", "I'll go back a little", "backing up"],
    "왼쪽": ["turning left", "I'll turn left", "turning to the left", "going left"],
    "오른쪽": ["turning right", "I'll turn right", "turning to the right", "going right"],
}
MOVE_SAY_EN = {a: _mix_en(_OE, b, [".", "!", "! Watch me.", ". Tell me when you want to stop."]) for a, b in _MBE.items()}
NOT_CALLS_EN += [
    ("my {color} balloon popped and i almost cried", ["That would surprise anyone. Did you get a new one?"]),
    ("they gave us {color} jelly at lunch", ["{Color} jelly sounds fun! How did it taste?"]),
    ("i'm the greeting helper in class this week", ["Nice job! Who did you greet first today?"]),
    ("i got a lot of claps yesterday", ["That sounds great! What was it for?"]),
    ("i'm joining a dance contest", ["Wow, good luck! What song will you use?"]),
    ("i bought a doll that waves", ["A waving doll sounds cute. Where did you buy it?"]),
    ("i love snowy days", ["Snow turns everything white. What do you do first when it snows?"]),
    ("i drew a fire safety poster", ["A fire safety poster is a great project! What did you draw?"]),
    ("i want to be a doctor someday", ["That's a big dream! What made you pick a doctor?"]),
    ("we walked backwards in P.E.", ["Walking backwards is tricky. How many steps could you take?"]),
    ("the light flickers over my desk", ["Flickering lights can tire your eyes. Tell a grown-up so they can fix it."]),
    ("i nodded off in class lol", ["You must have been tired. Try to go to bed a little earlier tonight."]),
]


def make(n: int, seed: int, chats: list[dict], lang: str = "ko") -> list[dict]:
    rng = random.Random(seed)
    kinds = [k for k, _ in KINDS]
    weights = [w for _, w in KINDS]
    out, seen = [], set()
    while len(out) < n:
        kind = rng.choices(kinds, weights)[0]
        last, topic = turn_for_en(kind, rng) if lang == "en" else turn_for(kind, rng)
        ctx = context(rng, chats)
        msgs = [{"role": "system", "content": SYSTEMS[lang]}] + ctx + last
        key = json.dumps(msgs, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        row = {"kind": "tool", "source": "tools", "topic": topic, "tools": True,
               "messages": msgs, **({"lang": lang} if lang != "ko" else {})}
        if topic == "불빛 이어서":
            # 앞 대답("주황색으로 바꿀게요!")은 앱이 기억에 남기는 모양(도구 호출 없이 말만)이라 읽기만 하고
            # 학습은 하지 않음: 학습하면 "불빛 요청에 말만 하고 부르지 않기"를 가르치게 됨 (train.py no_loss)
            row["no_loss"] = [1 + len(ctx) + 1]
        out.append(row)
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
