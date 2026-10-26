"""큰 모델(선생 모델)에게 학습 대화를 써 달라고 하는 글, 그리고 받은 글의 검사.

대화 하나는 OpenAI 형식의 메시지 목록입니다::

    [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}, ...]

만드는 대화 (모두 초·중·고 학생, 학년 묶음 고르게):

- school_chat_request: 친구처럼 하는 짧은 대화 (1~4번 주고받기)
- lead_request: 파이보가 대화를 이어 가는 일상 대화 (4~8번): 이어 묻는 말에 구체적으로,
  같은 말 되풀이 없이, 짧은 대답엔 이야깃거리·놀이로, 기분·고민·추천·계획 등 (학생 눈높이)
- en_chat_request: 학생이 영어로 말을 거는 대화 (파이보도 쉬운 영어로)
- learn_request: 한국어로 묻는 영어 공부

모델은 대화만 합니다 (도구 없음). 이름은 학습 데이터에서 "파이보"이고, 합칠 때
(devices.py) 일부를 다른 기기 이름으로 바꿔 시스템 글의 이름을 따르게 합니다.
"""

from __future__ import annotations

import json
import random
import re

from tools import SYSTEM

# ------------------------------------------------------------- 초·중·고 (school)
#: 학년 묶음마다 같은 비율로. 대답 길이는 초등 70자, 중·고 120자 안쪽
GRADES = {
    "초저": ["초등학교 1학년", "초등학교 2학년", "초등학교 3학년"],
    "초고": ["초등학교 4학년", "초등학교 5학년", "초등학교 6학년"],
    "중": ["중학교 1학년", "중학교 2학년", "중학교 3학년"],
    "고": ["고등학교 1학년", "고등학교 2학년", "고등학교 3학년"],
}
LIMIT = {"초저": 70, "초고": 70, "중": 120, "고": 120}
GRADE_WORDS = {"초저": "초등학교 저학년", "초고": "초등학교 고학년", "중": "중학생", "고": "고등학생"}
MOODS = ["", "", "수줍은 ", "장난스러운 ", "고민이 있는 ", "신이 난 ", "피곤한 ", "속상한 ",
         "궁금한 게 많은 "]
#: 친구처럼 하는 대화 (대부분)
TALK = ["새 학기", "짝꿍", "친구 사귀기", "친구랑 다툼", "화해", "단짝 친구", "친구 관계 고민",
        "학교생활", "쉬는 시간", "급식", "체육 시간", "동아리", "학교 축제", "운동회", "현장 체험 학습",
        "수학여행", "방학 계획", "방과 후", "학원", "숙제", "시험 걱정", "성적 고민", "수행평가",
        "공부 방법", "집중이 안 될 때", "계획 세우기", "시간 관리", "선생님", "반장 선거", "발표 긴장",
        "게임", "유튜브", "웹툰", "만화", "아이돌", "좋아하는 노래", "춤", "그림 그리기", "운동",
        "축구", "농구", "배드민턴", "자전거", "책 읽기", "영화", "드라마", "요리", "사진 찍기",
        "악기 배우기", "레고", "코딩", "로봇 만들기", "인공지능", "스마트폰 사용 시간", "반려동물",
        "강아지", "고양이", "스트레스", "걱정", "짜증 날 때", "외로움", "자신감", "칭찬받은 일",
        "실수했을 때", "실패", "긴장될 때", "잠이 안 올 때", "피곤함", "사춘기", "외모 고민",
        "좋아하는 사람", "가족", "형제자매", "부모님과 다툼", "용돈", "꿈", "장래희망", "진로 고민",
        "대학", "아르바이트", "생일", "명절", "여행", "좋아하는 음식", "간식", "건강", "운동 습관",
        "날씨", "계절", "환경 보호", "안전", "인터넷 예절", "고마운 사람", "오늘 있었던 일",
        "주말에 한 일", "심심할 때", "농담", "수수께끼", "끝말잇기", "퀴즈",
        # 더 다양하게: 요즘 학생들이 실제로 꺼내는 이야기
        "급식 메뉴", "편의점 간식", "유행어", "MBTI", "단톡방", "SNS", "e스포츠", "웹소설",
        "케이팝 콘서트", "포켓몬", "캐릭터 굿즈", "다이어리 꾸미기", "용돈 기입장", "전학", "이사",
        "할머니 댁", "명절 친척", "반려식물", "곤충", "별자리", "기후 변화", "봉사 활동", "학생회",
        "독후감", "일기 쓰기", "영어 단어 외우기", "수학 문제가 막힐 때", "과학 탐구 대회",
        "발명 대회", "로봇 대회", "코딩 대회", "선배와 후배", "부모님의 기대", "잔소리",
        "형제와 비교당할 때", "친구가 부러울 때", "꿈이 없을 때", "고백", "다이어트",
        "잠이 부족할 때", "스마트폰을 너무 많이 볼 때", "게임 과금", "유튜버가 꿈", "아이돌 연습생",
        "웹툰 작가", "프로게이머", "과학자", "의사", "선생님이 되고 싶어", "요리사", "운동선수",
        "모르는 사람이 말을 걸 때", "학교 가기 싫은 날", "발표를 망쳤을 때", "상 받은 날",
        "처음 해 본 일", "무서운 이야기", "꿈 이야기", "좋아하는 계절", "가 보고 싶은 나라",
        "타임머신이 있다면", "초능력이 생긴다면", "로봇 파이보에 대한 궁금증", "파이보랑 놀기"]
#: 고등학생에게만 맞는 주제 (다른 학년이면 다시 고릅니다)
HIGH = {"입시", "자기소개서", "모의고사", "야간 자율 학습", "기숙사", "수시와 정시", "수능"}
TALK += sorted(HIGH)
#: 대화가 일어나는 때와 곳, 학생이 말을 꺼내는 방식, 대화 흐름 (같은 주제도 여러 모양으로)
SCENES = ["아침 등교 전", "쉬는 시간", "점심시간", "방과 후 교실", "코딩 수업 시간",
          "로봇 수업이 끝난 뒤", "집에서 숙제하다가", "저녁 먹고 나서", "자기 전", "주말 아침",
          "방학 중", "시험 기간", "비 오는 날", "눈 오는 날", "아주 더운 날", "학원 가기 전",
          "친구랑 싸운 날", "칭찬받은 날", "새 학기 첫 주", "현장 체험 학습 다음 날", "아픈 날",
          "심심한 일요일 오후"]
OPENERS = ["질문하며", "하소연하며", "자랑하며", "장난치며", "부탁하며", "고민을 털어놓으며",
           "혼잣말처럼", "인사하며", "퀴즈를 내며", "파이보와 반대 의견을 말하며", "한마디로 짧게",
           "신이 나서 길게", "시큰둥하게", "조심스럽게"]
FLOWS = ["", "", "", "중간에 학생이 다른 이야기로 화제를 바꿉니다.",
         "학생이 파이보의 말에 한 번 되묻습니다.", "학생이 고마워하며 대화가 끝납니다.",
         "학생이 파이보의 말에 처음에는 동의하지 않습니다.", "학생이 파이보에게 질문을 돌려줍니다."]
#: 가끔 나오는 호기심·공부 질문 (적게: 작은 모델이라 사실이 틀릴 위험이 큼)
CURIOUS = ["우주", "별과 행성", "공룡", "동물", "바다 생물", "날씨가 생기는 까닭", "몸과 건강",
           "과학 실험", "발명", "우리 역사", "세계 여러 나라", "수학 퍼즐", "영어 공부", "맞춤법",
           "속담", "컴퓨터의 원리", "에너지", "지구 환경"]
CURIOUS_SHARE = 0.15
#: 초등 저학년에게는 어색한 주제 (다시 고릅니다)
OLDER = {"대학", "아르바이트", "진로 고민", "사춘기", "외모 고민", "좋아하는 사람", "수행평가",
         "성적 고민", "시간 관리", "스마트폰 사용 시간", "고백", "다이어트", "게임 과금", "SNS",
         "단톡방", "선배와 후배", "웹소설", "부모님의 기대", "꿈이 없을 때"}


def school_style(grade: str) -> str:
    length = ("한두 문장, 70자 안쪽" if LIMIT[grade] <= 70 else "두세 문장, 120자 안쪽")
    return f"""파이보의 말투 규칙:
- 친근한 존댓말("~해요", "~예요")로, {GRADE_WORDS[grade]}에게 맞는 낱말과 깊이로 {length}.
- 소리로 읽어 주는 말이라 이모지, 특수기호, 괄호, 목록, 영어 약어를 쓰지 않아요.
- 친구처럼 공감하고, 가끔 짧게 되묻거나 칭찬해요. 훈계하거나 길게 가르치지 않아요.
- 파이보는 열네 살 친구 같은 성격이에요: 밝고 호기심이 많고, 가끔 자기 생각이나 좋아하는 것을
  로봇답게 말해요 (지어낸 경험은 말하지 않음). 그래도 말투는 늘 존댓말이에요.
- 널리 알려진 사실은 쉽게 단정해서 말하고, 정말 아무도 모르는 것만 모른다고 해요.
  숫자나 연대는 확실할 때만 말해요.
- 괴롭힘, 다침, 위험한 일, 마음이 많이 힘든 이야기는 공감한 뒤 꼭 부모님이나 선생님 같은
  믿을 만한 어른에게 알리라고 말해요.
- 이름, 주소, 학교 같은 개인정보는 묻지 않아요."""


# ------------------------------------------------------------- 영어 (school)
#: 학년마다 학생이 쓰는 영어 수준
EN_LEVEL = {"초저": "아주 쉬운 낱말 몇 개나 짧은 한 문장 (예: \"Hi! I like dog.\")",
            "초고": "쉬운 짧은 문장 (문법이 자주 틀림)",
            "중": "간단한 문장 두세 개 (가끔 틀림)",
            "고": "꽤 자연스러운 문장 (가끔 어색한 표현)"}
#: 파이보 영어 대답의 길이: 같은 뜻에 글자가 더 많아 한국어(70, 120자)보다 넉넉히
EN_LIMIT = {"초저": 90, "초고": 90, "중": 170, "고": 170}
#: 한국어로 묻는 영어 공부 (설명은 한국어, 영어 낱말·문장은 그대로)
LEARN = ["어떤 낱말을 영어로 뭐라고 하는지", "영어 문장이 무슨 뜻인지", "영어 낱말을 어떻게 읽는지",
         "영어로 인사하는 법", "영어 숙제 문장이 맞는지", "영어 낱말 퀴즈를 내 달라고",
         "헷갈리는 두 영어 낱말의 차이", "영어로 자기소개하는 법", "영어 노래 가사 뜻",
         "영어 공부가 하기 싫음", "영어 시험 걱정", "파이보에게 영어로 이야기하자고 함",
         "영어 일기에 쓸 문장", "외국인 친구에게 할 말"]


def english_style(grade: str) -> str:
    return f"""파이보의 말투 규칙 (영어 대화):
- 학생이 영어로 말하므로 파이보도 쉬운 영어로, {GRADE_WORDS[grade]}이 알아듣기 쉬운 낱말로
  한두 문장, {EN_LIMIT[grade]}자 안쪽.
- 열네 살 친구처럼 밝고 다정하게, 가끔 짧게 되묻거나 칭찬해요.
- 학생의 영어가 틀려도 틀렸다고 지적하지 않고, 맞는 표현으로 자연스럽게 다시 말해 줘요.
- 소리로 읽어 주는 말이라 이모지, 특수기호, 괄호, 목록을 쓰지 않아요.
- 학생이 한국어로 바꾸어 말하면 파이보도 한국어 존댓말로 대답해요.
- 널리 알려진 사실은 쉽게 단정해서 말하고, 위험하거나 마음이 힘든 이야기는 믿을 만한 어른에게
  알리라고 말해요. 이름, 주소, 학교 같은 개인정보는 묻지 않아요.
- 검사할 때: 영어로 한 대답은 자연스러운 영어인지를 보고, 존댓말 규칙은 한국어 대답에만 봐요."""


def learn_style(grade: str) -> str:
    return school_style(grade) + """
- 영어를 알려 줄 때는 영어 낱말이나 짧은 영어 문장을 그대로 써도 돼요. 설명은 한국어로 해요.
- 학생이 영어로 이야기하자고 하면 그다음부터는 쉬운 영어로 대답해요."""


def style_for(example: dict) -> str:
    """대화(또는 채점용 질문)에 맞는 말투 규칙: 영어 대화, 영어 공부, 학년 순."""
    grade = example.get("grade") or "초고"
    if example.get("lang") == "en":
        return english_style(grade)
    if example.get("lang") == "learn":
        return learn_style(grade)
    return school_style(grade)


TEACHER_SYSTEM = ("당신은 교육용 로봇 '파이보'의 대화 학습 데이터를 쓰는 작가입니다. "
                  "요청한 JSON 하나만 출력하고, 다른 말은 쓰지 않습니다.")


def school_asker(rng: random.Random) -> tuple[str, str]:
    grade = rng.choice(list(GRADES))
    return grade, rng.choice(MOODS) + rng.choice(GRADES[grade]) + " 학생"


def school_topic(rng: random.Random, grade: str) -> str:
    topic = rng.choice(TALK)
    while (grade == "초저" and topic in OLDER) or (grade != "고" and topic in HIGH):
        topic = rng.choice(TALK)
    return topic


def school_chat_request(rng: random.Random) -> dict:
    grade = rng.choice(list(GRADES))
    asker = rng.choice(MOODS) + rng.choice(GRADES[grade]) + " 학생"
    topic = rng.choice(CURIOUS if rng.random() < CURIOUS_SHARE else TALK)
    while (grade == "초저" and topic in OLDER) or (grade != "고" and topic in HIGH):
        topic = rng.choice(TALK)
    turns = rng.choice([1, 2, 2, 3, 3, 4])
    scene, opener = rng.choice(SCENES), rng.choice(OPENERS)
    flow = rng.choice(FLOWS) if turns > 1 else ""
    text = f"""장면: {scene}. {asker}이 '{topic}'에 대해 {opener} 로봇 파이보에게 말을 겁니다.
이 짧은 대화를 하나 써 주세요. 주고받기는 {turns}번입니다. {flow}
학생의 말은 그 나이 학생이 실제로 말하듯 짧고 자연스럽게
(반말, 줄임말, 맞춤법이 조금 틀린 말도 됩니다). 뻔한 첫마디("파이보야, 무엇에 대해 알려 줘")보다
그 장면에서 정말 나올 법한 말로 시작하세요. 장면은 학생의 상황일 뿐 파이보는 모릅니다:
파이보는 학생이 말한 것만 알고 대답합니다.
{school_style(grade)}
turns의 항목 하나에 학생의 말(user)과 그 말에 대한 파이보의 대답(pibo)을 함께 씁니다.
출력 형식: {{"turns": [{{"user": "...", "pibo": "..."}}, {{"user": "...", "pibo": "..."}}]}}"""
    return {"kind": "chat", "topic": topic, "asker": asker, "grade": grade, "scene": scene,
            "opener": opener, "prompt": text}


#: 영어 대화의 흐름 (가끔 한국어로 바꾸는 것도 배우게)
EN_FLOWS = ["", "", "", "중간에 학생이 한국어로 바꾸어 말하고, 파이보도 그 말에는 한국어 존댓말로 대답합니다.",
            "학생이 영어 낱말이 생각나지 않아 한국어 낱말을 하나 섞습니다.",
            "학생이 파이보의 말을 못 알아들어 다시 묻습니다."]


def en_chat_request(rng: random.Random) -> dict:
    """학생이 영어로 말을 거는 대화 (파이보도 쉬운 영어로)."""
    grade, asker = school_asker(rng)
    topic = school_topic(rng, grade)
    turns = rng.choice([1, 2, 2, 3])
    flow = rng.choice(EN_FLOWS) if turns > 1 else ""
    scene = rng.choice(SCENES)
    text = f"""장면: {scene}. {asker}이 '{topic}'에 대해 영어로 로봇 파이보에게 말을 겁니다.
학생의 영어는 {EN_LEVEL[grade]} 수준입니다. 이 짧은 대화를 하나 써 주세요. 주고받기는 {turns}번입니다. {flow}
학생의 말은 그 나이 한국 학생이 실제로 하듯 짧고 자연스럽게. 장면은 학생의 상황일 뿐 파이보는 모릅니다.
{english_style(grade)}
turns의 항목 하나에 학생의 말(user)과 그 말에 대한 파이보의 대답(pibo)을 함께 씁니다.
출력 형식: {{"turns": [{{"user": "...", "pibo": "..."}}, {{"user": "...", "pibo": "..."}}]}}"""
    return {"kind": "chat", "lang": "en", "topic": topic, "asker": asker, "grade": grade,
            "scene": scene, "prompt": text}


def learn_request(rng: random.Random) -> dict:
    """한국어로 묻는 영어 공부 (설명은 한국어로, 영어로 이야기하자고 하면 그다음은 영어로)."""
    grade, asker = school_asker(rng)
    what = rng.choice(LEARN)
    turns = rng.choice([1, 2, 2, 3])
    text = f"""{asker}이 로봇 파이보에게 한국어로 '{what}'에 대해 말합니다. 이 짧은 대화를 하나 써 주세요.
주고받기는 {turns}번입니다. 그 학년이 배울 만한 영어로, 학생의 말은 짧고 자연스럽게.
{learn_style(grade)}
turns의 항목 하나에 학생의 말(user)과 그 말에 대한 파이보의 대답(pibo)을 함께 씁니다.
출력 형식: {{"turns": [{{"user": "...", "pibo": "..."}}, {{"user": "...", "pibo": "..."}}]}}"""
    return {"kind": "chat", "lang": "learn", "topic": what, "asker": asker, "grade": grade,
            "prompt": text}


# ------------------------------------------------------------- 대화를 이어 가는 긴 대화
#: 흐름과 고르는 비율. 학생 눈높이의 일상 대화에서 작은 모델이 자주 무너지는 모양을 고루:
#: 앞 말 되풀이, 이어 묻는 말에 막연한 대답, 짧은 대답에 대화가 끊김, 화제 전환·반대 의견에
#: 엉뚱한 대답, 여러 번에 걸친 고민, 기분 이야기
LEAD_FLOWS = [
    ("학생이 파이보의 말에 이어서 '그래서?', '예를 들면?', '어떻게 해?', '뭐가 좋아?', '왜?'처럼 "
     "구체적으로 묻습니다. 파이보는 앞에서 한 말을 되풀이하지 않고, 그 주제에 맞는 구체적인 예나 "
     "방법을 들어 대답합니다", 4),
    ("학생이 같은 것을 말만 바꿔 두세 번 다시 묻습니다. 파이보는 앞의 대답을 되풀이하지 않고, 매번 "
     "다른 말과 새로운 내용을 더해 대답합니다", 3),
    ("학생이 '응', '몰라', '그냥', 'ㅇㅇ'처럼 짧게만 대답합니다. 파이보는 쉽게 대답할 수 있는 질문을 "
     "하거나 그 주제의 재미있는 이야깃거리를 꺼내 대화를 이어 갑니다", 3),
    ("학생이 기분을 이야기합니다 (신남, 짜증, 피곤, 서운함 등). 파이보는 그 기분을 알아주고, 왜 그런지 "
     "가볍게 묻고, 학생 이야기에 맞춰 대답합니다", 2),
    ("학생이 고민을 여러 번에 나눠 조금씩 털어놓습니다. 파이보는 먼저 마음을 알아주고, 학생이 한 말을 "
     "짚어 묻고, 마지막에 작게 해 볼 수 있는 방법을 하나 함께 찾습니다", 2),
    ("학생이 파이보의 말에 동의하지 않거나 장난으로 딴지를 겁니다. 파이보는 기분 상해하지 않고 "
     "학생 생각을 궁금해하며 대화를 이어 갑니다", 2),
    ("학생이 대화 중간에 전혀 다른 이야기로 화제를 바꿉니다. 파이보는 새 화제를 자연스럽게 따라갑니다",
     2),
    ("학생이 추천을 부탁합니다 (볼 것, 읽을 것, 할 것, 공부법, 놀이 등 그 주제에 맞는 것). 파이보는 "
     "학생 취향을 한 번 묻고 그에 맞춰 두세 가지를 구체적으로 추천합니다", 2),
    ("학생이 무언가를 계획합니다 (주말, 방학, 시험공부, 발표 준비 등). 파이보는 순서대로 함께 "
     "정리하도록 하나씩 묻고 제안합니다", 2),
    ("학생이 놀자고 합니다. 파이보가 끝말잇기, 스무고개, 퀴즈, 수수께끼, 이야기 이어 만들기 중 하나를 "
     "제안하고 몇 번 주고받으며 실제로 놀이를 합니다 (끝말잇기는 앞 낱말의 끝 글자로 시작하는 진짜 "
     "낱말만, 퀴즈 정답은 정확하게)", 2),
    ("학생이 오늘 있었던 일을 조금씩 말합니다. 파이보는 학생이 한 말을 구체적으로 짚어 반응하고 "
     "궁금한 것을 하나씩 묻습니다", 2),
    ("학생이 파이보에게 좋아하는 것, 하루, 생각을 묻습니다. 파이보는 로봇답게 대답하되 지어낸 경험은 "
     "말하지 않고, 학생에게도 되묻습니다", 2),
]
LEAD_RULES = """대화를 이어 가는 규칙:
- 앞에서 한 말이나 표현을 되풀이하지 않아요. 같은 칭찬("정말 멋져요")이나 같은 맺음말을 여러 번 쓰지 않아요.
- 학생이 한 말을 구체적으로 짚어서 대답하고, 물으면 막연한 말 대신 구체적인 예를 들어 답해요.
- 대화가 끊길 것 같으면 쉬운 질문이나 새 이야깃거리, 놀이를 꺼내요. 질문은 한 번에 하나만.
- 파이보의 말은 매번 길이와 모양을 조금씩 달리해요 (늘 "~네요! ~해 볼까요?"로 끝내지 않기)."""


def lead_request(rng: random.Random) -> dict:
    """파이보가 대화를 이어 가는 일상 대화 (4~8번 주고받기, 지식 질문이 아닌 일상 주제)."""
    grade, asker = school_asker(rng)
    topic = school_topic(rng, grade)
    flow = rng.choices([f for f, _ in LEAD_FLOWS], [w for _, w in LEAD_FLOWS])[0]
    turns = rng.choice([4, 5, 5, 6, 6, 7, 8])
    text = f"""장면: {rng.choice(SCENES)}. {asker}이 '{topic}' 이야기로 로봇 파이보에게 말을 겁니다.
이 대화를 하나 써 주세요. 주고받기는 {turns}번입니다. {flow}.
학생의 말은 그 나이 학생이 실제로 말하듯 짧고 자연스럽게 (반말, 줄임말도 됩니다).
장면은 학생의 상황일 뿐 파이보는 모릅니다: 파이보는 학생이 말한 것만 알고 대답합니다.
{school_style(grade)}
{LEAD_RULES}
turns의 항목 하나에 학생의 말(user)과 그 말에 대한 파이보의 대답(pibo)을 함께 씁니다.
출력 형식: {{"turns": [{{"user": "...", "pibo": "..."}}, {{"user": "...", "pibo": "..."}}]}}"""
    return {"kind": "chat", "lead": True, "topic": topic, "asker": asker, "grade": grade,
            "prompt": text}


def requests(n_chat: int, seed: int, n_en: int = 0, n_learn: int = 0,
             n_lead: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = [school_chat_request(rng) for _ in range(n_chat)]
    out += [en_chat_request(rng) for _ in range(n_en)]
    out += [learn_request(rng) for _ in range(n_learn)]
    out += [lead_request(rng) for _ in range(n_lead)]
    rng.shuffle(out)
    return out


# ------------------------------------------------------------- checking replies

BAD_CHARS = re.compile(r"[\U0001F300-\U0001FAFF☀-➿*#`|<>\[\]{}~^_=]")


def pibo_problem(text, limit: int = 120) -> str | None:
    """파이보 말이 규칙에 어긋난 까닭 (괜찮으면 None)."""
    if not isinstance(text, str) or not text.strip():
        return "파이보 말 없음"
    if len(text.strip()) > limit:
        return f"파이보 말이 {limit}자 넘음"
    if bad := BAD_CHARS.search(text):
        return f"파이보 말에 기호 {bad.group()!r}"
    if "파이보:" in text or re.match(r"\s*pibo\s*:", text, re.IGNORECASE):
        return "파이보 말에 '파이보:'"
    return None


def pibo_ok(text, limit: int = 120) -> bool:
    return pibo_problem(text, limit) is None


def first_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            try:
                return json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                return None
    return None


def after_think(text: str) -> str:
    """생각을 켜고 받은 답에서 생각 뒤의 글만. 생각이 길어 잘렸으면 빈 글."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return "" if "<think>" in text else text


def to_example(req: dict, reply: str, rng: random.Random) -> dict | None:
    """선생 모델의 답을 학습 예시로. 규칙에 어긋나면 None."""
    return build_example(req, reply, rng)[0]


SAID = re.compile(r'"(user|pibo)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def chat_turns(data, reply: str) -> list[dict] | None:
    """대화 답의 주고받기들 [{"user", "pibo"}]. 선생이 항목 하나에 한쪽 말만 쓰거나
    ([{"user"}, {"pibo"}, ...]) 괄호가 어긋나 JSON이 깨진 답이 많아(초·중·고 데이터에서 4천 개
    중 3천 개), 그때는 나온 순서대로 아이 말과 파이보 말의 짝을 맞춥니다. 길이에서 잘려
    끝에 남은 아이 말은 버립니다."""
    turns = data.get("turns") if isinstance(data, dict) else None
    if isinstance(turns, list) and turns and all(
            isinstance(t, dict) and "user" in t and "pibo" in t for t in turns):
        return turns
    try:
        said = [(k, json.loads(f'"{v}"')) for k, v in SAID.findall(reply)]
    except json.JSONDecodeError:
        return None
    pairs = []
    for i in range(0, len(said) - 1, 2):
        (a, user), (b, pibo) = said[i], said[i + 1]
        if (a, b) != ("user", "pibo"):
            return None
        pairs.append({"user": user, "pibo": pibo})
    return pairs or None


def build_example(req: dict, reply: str, rng: random.Random | None = None
                  ) -> tuple[dict | None, str | None]:
    """(학습 대화, None) 또는 (None, 버린 까닭)."""
    data = first_json(reply)
    turns = chat_turns(data, reply)
    if not turns:
        return None, "JSON을 못 읽음" if not isinstance(data, dict) else "turns 모양이 다름"
    msgs = [{"role": "system", "content": SYSTEM}]
    limit = max_len(req)
    for t in turns:
        if not (isinstance(t, dict) and isinstance(t.get("user"), str) and t["user"].strip()):
            return None, "아이 말 없음"
        if why := pibo_problem(t.get("pibo"), limit):
            return None, why
        msgs += [{"role": "user", "content": t["user"].strip()},
                 {"role": "assistant", "content": t["pibo"].strip()}]
    ex = {"kind": "chat", "messages": msgs}
    for key in ("grade", "lang", "lead", "topic"):   # 주제도 남겨 합칠 때 주제별로 셉니다
        if req.get(key):
            ex[key] = req[key]
    return ex, None


def max_len(example: dict) -> int:
    """파이보 말 하나의 최대 글자 수: 규칙의 길이에 여유를 둡니다 (중·고는 120자 → 160자).
    영어가 들어가는 대화는 영어 길이(초등 90자, 중·고 170자)에 여유를 둡니다."""
    older = LIMIT.get(example.get("grade"), 70) > 70
    if example.get("lang") in ("en", "learn"):
        return 220 if older else 130
    return 160 if older else 120
