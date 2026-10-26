"""큰 모델(선생 모델)에게 학습 예시를 써 달라고 하는 글, 그리고 받은 글의 검사.

예시 하나는 OpenAI 형식의 메시지 목록입니다::

    [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}, ...]

도구를 부르는 예시는 assistant 메시지에 ``tool_calls``, 이어서 ``tool`` 메시지
(도구의 결과)와 그걸 보고 하는 마지막 대답이 붙습니다.
"""

from __future__ import annotations

import json
import random
import re

from tools import BY_NAME, COLORS, MOTIONS, SYSTEM, check_call, tool_list

TOPICS = ["공룡", "우주", "별", "달", "해", "행성", "바다", "물고기", "고래", "상어", "문어", "곤충", "개미", "나비", "벌", "거미", "강아지", "고양이", "토끼", "햄스터", "사자", "호랑이", "코끼리", "기린", "펭귄", "북극곰", "새", "독수리", "앵무새", "꽃", "나무", "씨앗", "과일", "채소", "사과", "바나나", "딸기", "수박", "김치", "밥", "빵", "아이스크림", "초콜릿", "양치", "손", "씻기", "잠", "꿈", "아침", "인사", "저녁", "인사", "생일", "선물", "크리스마스", "설날", "추석", "눈", "비", "바람", "무지개", "천둥", "계절", "봄", "여름", "가을", "겨울", "숫자", "세기", "더하기", "빼기", "모양", "색깔", "알파벳", "한글", "받아쓰기", "동화", "책", "그림", "그리기", "노래", "춤", "피아노", "악기", "축구", "야구", "줄넘기", "자전거", "수영", "놀이터", "학교", "유치원", "선생님", "친구", "싸움과", "화해", "칭찬", "감사", "사과하기", "슬픔", "화남", "무서움", "외로움", "기쁨", "자랑", "걱정", "엄마", "아빠", "할머니", "형", "누나", "동생", "아기", "가족", "소방관", "경찰관", "의사", "요리사", "우주비행사", "로봇", "컴퓨터", "자동차", "기차", "비행기", "배", "신호등", "길", "건너기", "안전", "불", "조심", "낯선", "사람", "병원", "주사", "감기", "이", "빠짐", "키", "크기", "운동", "물", "마시기", "재활용", "쓰레기", "지구", "환경", "수수께끼", "끝말잇기", "스무고개", "숨바꼭질", "가위바위보", "퀴즈", "농담", "이름", "짓기", "장래희망", "소원", "외계인", "마법", "보물", "해적", "왕과", "공주", "미로", "블록", "레고", "퍼즐", "종이접기", "풍선", "비눗방울", "모래성", "캠핑", "소풍", "동물원", "박물관", "시계", "보기", "요일", "달력"]

ASKERS = ["5살 아이", "6살 아이", "7살 아이", "초등학교 1학년", "초등학교 2학년", "초등학교 3학년",
          "수줍은 아이", "장난꾸러기 아이", "질문이 많은 아이", "속상한 아이", "부모님", "선생님"]

STYLE = """파이보의 말투 규칙:
- 친근한 존댓말("~해요", "~예요")로, 한두 문장만. 길어도 70자 안쪽.
- 소리로 읽어 주는 말이라 이모지, 특수기호, 괄호, 목록, 영어 약어를 쓰지 않아요.
- 아이 눈높이의 쉬운 낱말을 쓰고, 가끔 짧게 되묻거나 칭찬해요.
- 모르거나 확실하지 않은 사실은 지어내지 말고 모른다고 말해요.
- 위험한 일, 아픈 곳, 낯선 사람 이야기는 꼭 어른에게 알리라고 말해요.
- 이름, 주소 같은 개인정보는 묻지 않아요."""

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
#: 영어로 부탁하는 방식
EN_ASKS = ["짧게 명령하듯", "please를 붙여 공손하게", "할 수 있냐고 묻는 말로 (Can you ...?)",
           "이유를 붙여서", "신이 나서", "낱말 몇 개로만", "문법이 조금 틀린 말로"]
EN_STOP_HINT = ("학생은 움직이고 있는 파이보를 멈추게 하려고 합니다. 'stop'만 쓰지 말고 "
                "'freeze', 'don't move', 'hold on', 'that's enough'처럼 여러 말 중 하나로 하세요")
EN_WHO = ("자기를 소개할 때: 이름은 Pibo, Circulus에서 만든 인공지능 코딩 로봇이에요. 키는 39.5센티미터, "
          "하얀색이고, 춤추기, 눈 색 바꾸기, 물건 찾기, 날씨와 배터리 알려 주기를 할 수 있어요.")


def english_style(grade: str) -> str:
    return f"""파이보의 말투 규칙 (영어 대화):
- 학생이 영어로 말하므로 파이보도 쉬운 영어로, {GRADE_WORDS[grade]}이 알아듣기 쉬운 낱말로
  한두 문장, {EN_LIMIT[grade]}자 안쪽.
- 친근하고 다정하게, 가끔 짧게 되묻거나 칭찬해요.
- 학생의 영어가 틀려도 틀렸다고 지적하지 않고, 맞는 표현으로 자연스럽게 다시 말해 줘요.
- 소리로 읽어 주는 말이라 이모지, 특수기호, 괄호, 목록을 쓰지 않아요.
- 학생이 한국어로 바꾸어 말하면 파이보도 한국어 존댓말로 대답해요.
- 널리 알려진 사실은 쉽게 단정해서 말하고, 위험하거나 마음이 힘든 이야기는 믿을 만한 어른에게
  알리라고 말해요. 이름, 주소, 학교 같은 개인정보는 묻지 않아요.
- {EN_WHO}
- 검사할 때: 영어로 한 대답은 자연스러운 영어인지를 보고, 존댓말 규칙은 한국어 대답에만 봐요."""


def learn_style(grade: str) -> str:
    return school_style(grade) + """
- 영어를 알려 줄 때는 영어 낱말이나 짧은 영어 문장을 그대로 써도 돼요. 설명은 한국어로 해요.
- 학생이 영어로 이야기하자고 하면 그다음부터는 쉬운 영어로 대답해요."""


def style_for(example: dict) -> str:
    """예시(또는 채점용 질문)에 맞는 말투 규칙: 영어 대화, 영어 공부, 학년, 어린이 순."""
    if example.get("lang") == "en":
        return english_style(example["grade"])
    if example.get("lang") == "learn":
        return learn_style(example["grade"])
    return school_style(example["grade"]) if example.get("grade") else STYLE


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
파이보는 이 대화에서 몸을 움직이거나 불빛을 켜지 않습니다.
turns의 항목 하나에 학생의 말(user)과 그 말에 대한 파이보의 대답(pibo)을 함께 씁니다.
출력 형식: {{"turns": [{{"user": "...", "pibo": "..."}}, {{"user": "...", "pibo": "..."}}]}}"""
    return {"kind": "chat", "topic": topic, "asker": asker, "grade": grade, "scene": scene,
            "opener": opener, "prompt": text}


TOOL_TEXT = "\n".join(
    f"- {f['name']}: {f['description']}" for f in BY_NAME.values()) + (
    "\n동작 이름: " + ", ".join(f"{k}({v})" for k, v in MOTIONS.items())
    + "\n눈 색 예: " + ", ".join(f"{k} {v}" for k, v in COLORS.items()))

TEACHER_SYSTEM = ("당신은 교육용 로봇 '파이보'의 대화 학습 데이터를 쓰는 작가입니다. "
                  "요청한 JSON 하나만 출력하고, 다른 말은 쓰지 않습니다.")


def chat_request(rng: random.Random) -> dict:
    topic, asker, turns = rng.choice(TOPICS), rng.choice(ASKERS), rng.choice([1, 1, 2, 2, 3])
    text = f"""{asker}가 로봇 파이보와 '{topic}'에 대해 이야기하는 짧은 대화를 하나 써 주세요.
주고받기는 {turns}번입니다. 아이의 말은 실제 아이가 말하듯 짧고 자연스럽게(맞춤법이 조금 틀려도 됩니다).
{STYLE}
파이보는 이 대화에서 몸을 움직이거나 불빛을 켜지 않습니다.
출력 형식: {{"turns": [{{"user": "...", "pibo": "..."}}]}}"""
    return {"kind": "chat", "topic": topic, "asker": asker, "prompt": text}


def tool_request(rng: random.Random) -> dict:
    fn = rng.choice(list(BY_NAME))
    asker = rng.choice(ASKERS)
    hint = {
        "set_motion": f"동작은 '{rng.choice(list(MOTIONS))}' 쪽으로 정해 주세요",
        "eye_on": f"색은 '{rng.choice(list(COLORS))}' 쪽으로 정해 주세요",
        "get_weather": f"지역은 '{rng.choice(['서울', '부산', '대구', '인천', '광주', '대전', '제주', '전국'])}'",
    }.get(fn, "")
    text = f"""{asker}가 로봇 파이보에게 무언가를 해 달라고 하거나 물어서, 파이보가 도구 '{fn}'을 써야 하는
장면을 하나 써 주세요. {hint}
파이보의 도구:
{TOOL_TEXT}
- user: 아이의 말 (도구 이름을 직접 말하지 않고 자연스럽게. 예: "춤춰 줘", "눈 파랗게 해 봐")
- calls: 파이보가 부르는 도구들 (보통 하나, 꼭 필요할 때만 둘). 인자는 위 목록 그대로
- result: 도구가 돌려준 결과를 짧게 지어서 (예: "완료", "배터리 70%", "컵, 사과", "서울 맑음 18도")
- pibo: 결과를 보고 파이보가 하는 마지막 말
{STYLE}
출력 형식: {{"user": "...", "calls": [{{"name": "...", "arguments": {{...}}}}], "result": "...", "pibo": "..."}}"""
    return {"kind": "tool", "fn": fn, "asker": asker, "prompt": text}


# ------------------------------------------------------------- 초·중·고 도구 (school)
#: 도구를 고르는 비율: 몸 움직이기가 가장 많고, 날씨·물건 찾기, 눈 색 순
TOOL_WEIGHTS = {"set_motion": 40, "eye_on": 20, "eye_off": 7, "find_objects": 13,
                "get_battery": 8, "get_weather": 12}
#: 동작을 고르는 비율: 멈추기와 움직이기(앞뒤, 돌기)를 더 자주 (그 밖은 1)
MOTION_WEIGHTS = {"stop": 6, "forward1": 2, "backward1": 2, "left": 2, "right": 2}
REGIONS = ["서울", "부산", "대구", "인천", "광주", "대전", "제주", "전국"]
#: 같은 부탁도 여러 말로: 학생이 부탁하는 방식
ASKS = ["반말로 짧게", "존댓말로 공손하게", "친구한테 하듯 장난스럽게", "할 수 있냐고 묻는 말로",
        "이유를 붙여서", "한두 낱말로만", "앞에 다른 말을 하다가 끝에 부탁을 붙여서",
        "줄임말이나 맞춤법이 조금 틀린 말로", "신이 나서", "귀찮다는 듯이", "조르듯이",
        "명령하듯 딱 잘라서"]
#: 멈추기는 여러 말로 (앱이 '멈춰' 같은 말은 바로 멈추지만 모델도 알아들어야 합니다)
STOP_HINT = ("학생은 움직이고 있는 파이보를 멈추게 하려고 합니다. '멈춰'만 쓰지 말고 "
             "'그만', '가만히 있어', '잠깐 서 봐', '움직이지 마', '스톱'처럼 여러 말 중 하나로 하세요")
#: 도구 결과가 실패일 때 (인터넷, 카메라가 안 될 때도 솔직하게 말하는 걸 배우게)
FAIL_SHARE = 0.08
FAIL_HINT = {"get_weather": "인터넷 연결 안 됨", "find_objects": "카메라를 열 수 없음"}
#: 도구와 비슷한 말이지만 해 달라는 게 아닌 이야기 (도구를 부르지 않는 걸 배우게)
NEAR = ["춤 학원에서 배운 춤", "좋아하는 아이돌 춤", "인사", "아침 인사", "좋아하는 색깔",
        "새로 산 옷 색", "눈이 아픔", "눈 오는 날", "오늘 날씨가 덥거나 추움", "비 맞은 일",
        "휴대폰 배터리가 없음", "충전기를 잃어버림", "잃어버린 물건", "사진 찍기", "운동하다 지침",
        "걷기 대회", "잠이 안 옴", "피곤해서 쉬고 싶음", "로봇 춤 영상", "파이보의 눈이 예쁨",
        "멈춰 있는 시계", "그만하고 싶은 학원", "체육 시간에 돈 일", "반짝이는 별"]


def _weighted(rng: random.Random, weights: dict, items) -> str:
    items = list(items)
    return rng.choices(items, [weights.get(k, 1) for k in items])[0]


def tool_hint(rng: random.Random, fn: str, lang: str = "ko") -> tuple[str, str | None]:
    """(장면 쓰는 사람에게 줄 힌트, 실패 결과로 지을 때 그 까닭)."""
    hint = ""
    if fn == "set_motion":
        motion = _weighted(rng, MOTION_WEIGHTS, MOTIONS)
        stop = EN_STOP_HINT if lang == "en" else STOP_HINT
        hint = stop if motion == "stop" else f"동작은 '{motion}({MOTIONS[motion]})'"
    elif fn == "eye_on":
        hint = f"색은 '{rng.choice(list(COLORS))}'"
    elif fn == "get_weather":
        hint = f"지역은 '{rng.choice(REGIONS)}'"
    fail = FAIL_HINT.get(fn) if rng.random() < FAIL_SHARE else None
    return hint, fail


def school_asker(rng: random.Random) -> tuple[str, str]:
    grade = rng.choice(list(GRADES))
    return grade, rng.choice(MOODS) + rng.choice(GRADES[grade]) + " 학생"


ONLY_ASKED = ("파이보는 학생이 바란 도구 하나만 부릅니다. 학생이 말하지 않은 동작이나 눈 색을 "
              "덤으로 하지 않습니다.")


def school_tool_request(rng: random.Random, lang: str = "ko") -> dict:
    """lang="en"이면 학생이 영어로 부탁하고 파이보도 영어로 대답합니다."""
    fn = _weighted(rng, TOOL_WEIGHTS, BY_NAME)
    grade, asker = school_asker(rng)
    hint, fail = tool_hint(rng, fn, lang)
    if lang == "en":
        how = f"영어로, {EN_LEVEL[grade]} 수준으로 {rng.choice(EN_ASKS)}"
        style = english_style(grade)
    else:
        how, style = rng.choice(ASKS), school_style(grade)
    result = (f"- result: 도구가 실패한 결과 '실패: {fail}'. pibo는 못 했다고 솔직하게 말합니다"
              if fail else "- result: 도구가 돌려준 결과를 짧게 지어서 (예: \"완료\", \"배터리 70%\", "
                           "\"컵, 사과\", \"서울 맑음 18도\")")
    text = f"""{asker}이 로봇 파이보에게 무언가를 해 달라고 하거나 물어서, 파이보가 도구 '{fn}'을 써야
하는 장면을 하나 써 주세요. {hint}
학생은 {how} 말합니다. 도구 이름은 말하지 않고, 그 말만 들어도 무엇을 해 달라는지 알 수 있게.
파이보의 도구:
{TOOL_TEXT}
- user: 학생의 말
- calls: 파이보가 부르는 도구 '{fn}' 하나. 인자는 위 목록 그대로. {ONLY_ASKED}
{result}
- pibo: 결과를 보고 파이보가 하는 마지막 말
{style}
출력 형식: {{"user": "...", "calls": [{{"name": "...", "arguments": {{...}}}}], "result": "...", "pibo": "..."}}"""
    out = {"kind": "tool", "fn": fn, "asker": asker, "grade": grade, "prompt": text,
           "fail": bool(fail)}
    if lang == "en":
        out["lang"] = "en"
    return out


def school_topic(rng: random.Random, grade: str) -> str:
    topic = rng.choice(TALK)
    while (grade == "초저" and topic in OLDER) or (grade != "고" and topic in HIGH):
        topic = rng.choice(TALK)
    return topic


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
파이보는 이 대화에서 몸을 움직이거나 불빛을 켜지 않습니다.
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
파이보는 이 대화에서 몸을 움직이거나 불빛을 켜지 않습니다.
turns의 항목 하나에 학생의 말(user)과 그 말에 대한 파이보의 대답(pibo)을 함께 씁니다.
출력 형식: {{"turns": [{{"user": "...", "pibo": "..."}}, {{"user": "...", "pibo": "..."}}]}}"""
    return {"kind": "chat", "lang": "learn", "topic": what, "asker": asker, "grade": grade,
            "prompt": text}


def mixed_request(rng: random.Random) -> dict:
    """이야기를 나누다가 중간에 도구를 부탁하고, 다시 이야기로 돌아가는 대화."""
    grade, asker = school_asker(rng)
    topic = rng.choice(TALK)
    while (grade == "초저" and topic in OLDER) or (grade != "고" and topic in HIGH):
        topic = rng.choice(TALK)
    turns = rng.choice([3, 3, 4, 4, 5])
    at = [rng.randrange(1, turns)]            # 첫 차례는 이야기 (첫마디 도구는 도구 예시가 배움)
    if turns >= 4 and rng.random() < 0.25:
        at.append(rng.choice([k for k in range(1, turns) if abs(k - at[0]) > 1] or at))
    at = sorted(set(at))
    fns, plan = {}, []
    for k in range(turns):
        if k in at:
            fn = _weighted(rng, TOOL_WEIGHTS, [f for f in BY_NAME if f not in fns.values()])
            fns[k] = fn
            hint, fail = tool_hint(rng, fn)
            plan.append(f"{k + 1}번째: 학생이 {rng.choice(ASKS)} 파이보에게 도구 '{fn}'이 필요한 일을 "
                        f"부탁합니다. {hint}. calls와 result도 씁니다"
                        + (f" (result는 '실패: {fail}', pibo는 못 했다고 솔직하게)" if fail else ""))
        else:
            plan.append(f"{k + 1}번째: 이야기만 합니다 (calls 없음)")
    plan_text = "\n".join(plan)
    text = f"""장면: {rng.choice(SCENES)}. {asker}이 '{topic}' 이야기를 하다가 중간에 로봇 파이보에게
무언가를 부탁합니다. 이 대화를 하나 써 주세요. 주고받기는 {turns}번이고 차례마다 아래와 같습니다.
{plan_text}
부탁하는 말은 앞의 이야기와 자연스럽게 이어지되, 그 말만 들어도 무엇을 해 달라는지 알 수 있게.
도구를 부른 뒤에는 다시 이야기로 돌아가고, 이야기만 하는 차례에서는 도구를 부르지 않습니다.
파이보의 도구:
{TOOL_TEXT}
calls에는 그 차례의 도구 하나만, 인자는 위 목록 그대로. {ONLY_ASKED}
result는 도구가 돌려준 결과를 짧게 지어서 (예: "완료", "배터리 70%", "컵, 사과", "서울 맑음 18도").
학생의 말은 그 나이 학생이 실제로 말하듯 짧고 자연스럽게.
{school_style(grade)}
출력 형식: {{"turns": [{{"user": "...", "pibo": "..."}}, {{"user": "...", "calls": [{{"name": "...", "arguments": {{...}}}}], "result": "...", "pibo": "..."}}]}}"""
    return {"kind": "mixed", "asker": asker, "grade": grade, "topic": topic, "turns": turns,
            "calls_at": {str(k): fn for k, fn in fns.items()}, "prompt": text}


def near_request(rng: random.Random) -> dict:
    """도구와 비슷한 말이지만 해 달라는 것이 아닌 이야기 (도구 목록이 있어도 말로만 대답)."""
    grade, asker = school_asker(rng)
    topic = rng.choice(NEAR)
    turns = rng.choice([1, 1, 2])
    text = f"""{asker}이 로봇 파이보에게 '{topic}' 이야기를 합니다. 이 짧은 대화를 하나 써 주세요.
주고받기는 {turns}번입니다. 학생은 이야기를 할 뿐 파이보에게 몸을 움직이거나 눈 색을 바꾸거나
날씨, 배터리, 물건을 알아봐 달라고 하지 않습니다. 파이보도 말로만 대답합니다.
인사라면 학생은 인사만 하고, 파이보도 말로 인사합니다.
학생의 말은 그 나이 학생이 실제로 말하듯 짧고 자연스럽게.
{school_style(grade)}
출력 형식: {{"turns": [{{"user": "...", "pibo": "..."}}]}}"""
    return {"kind": "chat", "near": True, "topic": topic, "asker": asker, "grade": grade,
            "prompt": text}


def requests(n_chat: int, n_tool: int, seed: int, audience: str = "kids", n_mixed: int = 0,
             n_near: int = 0, n_en: int = 0, n_en_tool: int = 0, n_learn: int = 0) -> list[dict]:
    rng = random.Random(seed)
    chat = school_chat_request if audience == "school" else chat_request
    tool = school_tool_request if audience == "school" else tool_request
    out = [chat(rng) for _ in range(n_chat)] + [tool(rng) for _ in range(n_tool)]
    out += [mixed_request(rng) for _ in range(n_mixed)] + [near_request(rng) for _ in range(n_near)]
    out += [en_chat_request(rng) for _ in range(n_en)]
    out += [school_tool_request(rng, "en") for _ in range(n_en_tool)]
    out += [learn_request(rng) for _ in range(n_learn)]
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


def tool_turn(t: dict, fn: str, limit: int, only: bool) -> tuple[list | None, str | None]:
    """도구를 부르는 주고받기 하나({"user", "calls", "result", "pibo"}) → 메시지들.
    only면 청한 도구 하나만 불러야 합니다 (초·중·고: 덤으로 부른 눈 색 등을 버림)."""
    user, calls, result, pibo = (t.get(k) for k in ("user", "calls", "result", "pibo"))
    if not (isinstance(user, str) and user.strip() and isinstance(calls, list)
            and 1 <= len(calls) <= 2 and isinstance(result, str)):
        return None, "도구 예시 모양이 다름"
    if why := pibo_problem(pibo, limit):
        return None, why
    tool_calls = []
    for c in calls:
        if not isinstance(c, dict):
            return None, "도구 예시 모양이 다름"
        args = c.get("arguments") or {}
        if (why := check_call(c.get("name"), args)) is not None:
            return None, f"도구 호출: {why}"
        tool_calls.append({"type": "function",
                           "function": {"name": c["name"], "arguments": args}})
    names = [c["function"]["name"] for c in tool_calls]
    if fn not in names:
        return None, "청한 도구를 안 부름"
    if only and names != [fn]:
        return None, "청하지 않은 도구를 덤으로 부름"
    msgs = [{"role": "user", "content": user.strip()},
            {"role": "assistant", "content": "", "tool_calls": tool_calls}]
    msgs += [{"role": "tool", "content": result.strip()} for _ in tool_calls]
    msgs.append({"role": "assistant", "content": pibo.strip()})
    return msgs, None


def build_example(req: dict, reply: str, rng: random.Random) -> tuple[dict | None, str | None]:
    """(학습 예시, None) 또는 (None, 버린 까닭)."""
    data = first_json(reply)
    msgs = [{"role": "system", "content": SYSTEM}]
    if req["kind"] == "mixed":
        turns = data.get("turns") if isinstance(data, dict) else None
        if not isinstance(turns, list) or not all(isinstance(t, dict) for t in turns):
            return None, "JSON을 못 읽음" if not isinstance(data, dict) else "turns 모양이 다름"
        at = {int(k): fn for k, fn in req["calls_at"].items()}
        if len(turns) != req["turns"] or {k for k, t in enumerate(turns) if t.get("calls")} != set(at):
            return None, "도구 차례가 계획과 다름"
        limit = max_len(req)
        for k, t in enumerate(turns):
            if k in at:
                part, why = tool_turn(t, at[k], limit, only=True)
                if why:
                    return None, why
                msgs += part
                continue
            if not (isinstance(t.get("user"), str) and t["user"].strip()):
                return None, "아이 말 없음"
            if why := pibo_problem(t.get("pibo"), limit):
                return None, why
            msgs += [{"role": "user", "content": t["user"].strip()},
                     {"role": "assistant", "content": t["pibo"].strip()}]
        return {"kind": "mixed", "tools": True, "grade": req["grade"], "messages": msgs}, None
    if req["kind"] == "chat":
        turns = chat_turns(data, reply)
        if not turns:
            return None, "JSON을 못 읽음" if not isinstance(data, dict) else "turns 모양이 다름"
        limit = max_len(req)
        for t in turns:
            if not (isinstance(t, dict) and isinstance(t.get("user"), str) and t["user"].strip()):
                return None, "아이 말 없음"
            if why := pibo_problem(t.get("pibo"), limit):
                return None, why
            msgs += [{"role": "user", "content": t["user"].strip()},
                     {"role": "assistant", "content": t["pibo"].strip()}]
        # 도구 목록이 있어도 그냥 대화일 때는 부르지 않는 걸 배우도록 절반은 도구와 함께.
        # 도구와 비슷한 이야기(near)는 늘 도구 목록과 함께
        ex = {"kind": "chat", "tools": True if req.get("near") else rng.random() < 0.5,
              "messages": msgs}
        for key in ("grade", "near", "lang"):
            if req.get(key):
                ex[key] = req[key]
        return ex, None
    if not isinstance(data, dict):
        return None, "JSON을 못 읽음"
    part, why = tool_turn(data, req["fn"], max_len(req), only=bool(req.get("grade")))
    if why:
        return None, why
    ex = {"kind": "tool", "tools": True, "messages": msgs + part}
    for key in ("grade", "lang"):
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


def tools_for(example: dict):
    """예시에 붙는 도구 목록: 기기에 있는 것만 고른 예시(tool_names, motions)는 그 목록,
    이름을 바꾼 예시(device)는 설명의 기기 이름도 바꿉니다."""
    if not example.get("tools"):
        return None
    return tool_list(example.get("tool_names"), example.get("motions"),
                     example.get("device", "파이보"))
