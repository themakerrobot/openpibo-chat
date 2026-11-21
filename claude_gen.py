"""Claude Code 세션(구독 계정)에서 초·중 학생과 더메이커의 대화를 처음부터 씁니다.

API를 부르지 않습니다. 세션의 Claude가 `plan`이 내준 칸(학년·주제·성격·여는 상황)마다 대화를 하나씩
써서 data/claude-raw/*.jsonl에 두고, `build`가 글자 검사(prompts.pibo_problem)를 거쳐 rewrite.py가
내는 data/pibo.jsonl과 같은 모양으로 data/claude.jsonl을 만듭니다.

    python claude_gen.py plan --start 0 --n 20      # 쓸 칸 20개 (늘 같은 순서)
    python claude_gen.py build                      # 거르고 합치기 → data/claude.jsonl
    python claude_gen.py txt data/claude-raw/batch-00020.txt   # 글 모양 묶음 → 같은 이름 .jsonl

기존 대화(data/pibo.jsonl, 학생 말이 어른 말투인 것이 많음) 살리기:

    python claude_gen.py triage                     # keep·rewrite·drop으로 나눔 → data/pibo-triage.jsonl
    python claude_gen.py plan-rewrite --start 0 --n 40   # 고쳐 쓸 대화 (늘 같은 순서)
    python claude_gen.py kid                        # 합침 → data/pibo-kid.jsonl

- keep: 학생 말이 짧은 반말이고 어른 생활 낱말이 없음. 그대로 씀.
- drop: 학생 말에 어른 생활(직장, 연애, 결혼, 육아, 술 …)이나 부모가 하는 말("내 아이", "~단다").
  pibo-kid에서 뺌.
- rewrite: 그 사이. 세션의 Claude가 "## p:<원래 id>"로 아이 말투로 고쳐 쓴 것이 있으면 그것으로
  바꾸고, 아직 없으면 원래 대화를 그대로 둠 (데이터를 잃지 않게).

raw 한 줄: {"id": "c00000", "grade": "초3", "topic": "...", "persona": "...",
            "turns": [{"user": 학생 말, "pibo": 더메이커 말}, ...]}

쓰는 규칙 (세션의 Claude에게):
- 학생 말: 그 학년 아이가 정말 하는 말. 반말, 짧고 끊기는 문장, 가끔 틀린 맞춤법·띄어쓰기, "ㅋㅋ",
  "근데", "아 몰라" 같은 말버릇, 말 돌리기, 엉뚱한 질문. 초1~2는 아주 짧게, 중학생은 조금 길고 비꼼도.
  욕설·혐오·성적인 말은 쓰지 않음. 이름·학교 같은 실제 개인정보는 넣지 않음.
- 더메이커 말: prompts.PIBO_STYLE 그대로 (로봇 대신 AI). 한두 문장 80자 안쪽, 기호·이모지 없음,
  반말·"-시-" 없음, 사람 경험을 지어내지 않음, 되받아 읊지 않음.
- 주고받기 3~6번. 대화마다 첫 말과 흐름이 달라야 함 (같은 첫인사 되풀이 금지).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from prompts import pibo_problem, to_messages

RAW = Path("data/claude-raw")
OUT = Path("data/claude.jsonl")
PIBO = Path("data/pibo.jsonl")
TRIAGE = Path("data/pibo-triage.jsonl")
KID_OUT = Path("data/pibo-kid.jsonl")

#: 학생 말에 있으면 어른으로 보는 낱말 (drop)
ADULT = re.compile(
    r"여자 ?친구|남자 ?친구|남친|여친|애인|연애|소개팅|회사|직장|출근|퇴근|월급|연봉|상사|계약|결혼|신혼|"
    r"남편|아내|와이프|군대|입대|제대|대학|취업|취직|면접|과장|부장|대리님|육아|임신|출산|소주|맥주|술자리|"
    r"와인|담배|대출|이자|주식|집값|월세|전세|부동산|손주|며느리|사위|시어머니|장모|갱년기|노후|은퇴|자취|"
    r"운전|청약|고시원|내 아이|우리 애|애들이|단다[.!?]*$|거라[.!?]*$|#@|\*\*|키키|코로나")
#: 학생 말이 존댓말로 끝남 (아이도 쓰지만 원본에서는 어른끼리 말이 많음)
POLITE_END = re.compile(r"(요|니다|니까|세요)[.?!~ ]*$")

#: 학생 말에서 빼는 글자: 이모지·다른 나라 글자 (아이들이 쓰는 ㅋㅋ ㅠㅠ ~ ^^ 는 둠)
KID_BAD = re.compile(r"[^ -~\s가-힣ㄱ-ㅎㅏ-ㅣ…·“”‘’]")

GRADES_OLD = ["초1", "초2", "초3", "초4", "초5", "초6", "중1", "중2", "중3"]   # c00000~c01019 칸이 쓴 학년
GRADES = GRADES_OLD + ["고1", "고2", "고3"]

TOPICS = [
    "오늘 학교에서 있었던 일", "쉬는 시간 놀이", "급식 메뉴", "받아쓰기·시험", "숙제하기 싫음",
    "학원", "친구랑 싸움", "새 짝꿍", "단짝 친구", "전학 온 친구", "반려동물", "키우고 싶은 동물",
    "공룡", "곤충", "우주와 별", "날씨와 계절", "방학 계획", "생일", "명절", "좋아하는 게임",
    "유튜브 영상", "만화·애니메이션", "좋아하는 노래", "그림 그리기", "만들기·레고", "운동·태권도",
    "축구·피구", "수학 문제 질문", "과학 궁금증", "역사 인물", "영어 단어", "책 읽기", "꿈과 장래희망",
    "동생·형제 이야기", "부모님께 혼남", "용돈", "무서운 꿈", "귀신 이야기", "잠이 안 옴", "심심함",
    "AI에게 궁금한 것", "말장난·수수께끼", "끝말잇기", "요리·간식", "편식", "스마트폰 시간",
    "외모 고민", "사춘기 짜증", "성적 걱정", "진로 고민", "아이돌", "웹툰", "환경·쓰레기",
    "자연재해 궁금증", "몸과 건강", "기분이 우울함", "칭찬받은 일", "발표 떨림", "체험학습·소풍",
]

PERSONAS = [
    "수줍고 말이 짧음", "장난꾸러기, 놀리기 좋아함", "질문이 끝없이 많음", "자랑하기 좋아함",
    "투덜거림이 많음", "엉뚱하고 상상력이 큼", "차분하고 진지함", "말을 자꾸 딴 데로 돌림",
    "맞춤법을 자주 틀림", "신나서 말이 빠름",
]

OPENERS = [
    "인사 없이 바로 본론", "AI를 시험해 보려는 질문", "방금 있었던 일을 털어놓음",
    "부탁이나 도움 요청", "아무 말이나 툭 던짐", "놀이를 하자고 함",
]


#: 주제 그룹. 앞의 59개는 c00000~c00539에서 이미 씀, 그룹마다 새 주제를 더해 폭을 넓힘
TOPIC_GROUPS = {
    "학교생활": ["오늘 학교에서 있었던 일", "쉬는 시간 놀이", "급식 메뉴", "받아쓰기·시험", "숙제하기 싫음", "학원",
             "새 짝꿍", "전학 온 친구", "발표 떨림", "체험학습·소풍", "성적 걱정", "칭찬받은 일",
             "수행평가·조별과제", "동아리·방과후", "학급 회장·선거", "학교 규칙", "청소 당번"],
    "친구·가족": ["친구랑 싸움", "단짝 친구", "동생·형제 이야기", "부모님께 혼남", "명절", "생일",
              "질투와 서운함", "오해와 사과", "조부모님", "이사와 이별", "부모님 잔소리", "비밀 이야기"],
    "마음": ["기분이 우울함", "사춘기 짜증", "외모 고민", "잠이 안 옴", "무서운 꿈", "귀신 이야기", "진로 고민",
           "불안과 걱정", "화가 날 때", "외로움", "자신감", "실수와 후회", "스트레스 풀기"],
    "놀이·취미": ["좋아하는 게임", "유튜브 영상", "만화·애니메이션", "웹툰", "좋아하는 노래", "그림 그리기",
              "만들기·레고", "운동·태권도", "축구·피구", "책 읽기", "끝말잇기", "말장난·수수께끼", "요리·간식",
              "반려동물", "키우고 싶은 동물", "보드게임·카드", "춤과 음악 따라하기", "캠핑·여행 계획", "마술", "심심함", "아이돌"],
    "지식·궁금증": ["공룡", "곤충", "우주와 별", "날씨와 계절", "수학 문제 질문", "과학 궁금증", "역사 인물",
                "영어 단어", "자연재해 궁금증", "몸과 건강", "환경·쓰레기", "동물의 신기한 점", "식물과 꽃",
                "바다와 물고기", "한글과 우리말", "세계 여러 나라", "발명과 기계", "수학 퍼즐", "로봇과 코딩"],
    "생활·안전": ["용돈", "스마트폰 시간", "편식", "방학 계획", "꿈과 장래희망", "교통안전", "인터넷 안전",
              "낯선 사람과 안전", "위생과 건강 습관", "심부름과 집안일", "응급 상황"],
    "AI·더메이커": ["AI에게 궁금한 것", "더메이커가 할 수 있는 일", "더메이커가 모르는 것", "AI는 감정이 있을까",
                "AI와 친구 하기", "AI에게 숙제 시키기"],
    "고등학생 생활": ["수능·입시 준비", "모의고사 결과", "학과·진로 선택", "생활기록부와 동아리", "야간자율학습과 학원",
                 "한국사와 제2외국어", "수학이 너무 어려움", "수면과 체력 관리", "SNS와 비교하는 마음",
                 "용돈과 아르바이트 고민", "수학여행과 학교 행사", "친구 관계 거리 두기", "번아웃과 쉬는 법",
                 "발표와 면접 준비", "코딩 프로젝트와 앱 만들기", "독서와 글쓰기 수행평가"],
    "도구와 헷갈리는 말": ["좋아하는 색 이야기", "춤 학원 이야기", "불빛과 어둠 이야기", "눈 건강 이야기",
                    "인사와 박수 예절 이야기", "움직임과 운동 이야기"],
}
TOPIC_GROUP = {t: g for g, ts in TOPIC_GROUPS.items() for t in ts}
ALL_TOPICS = [t for ts in TOPIC_GROUPS.values() for t in ts]
#: 새 주제 쓸 때의 주의 (plan이 칸마다 보여 줌)
TOPIC_NOTES = {
    "고등학생 생활": "고등학생(고1~고3)과 나누는 대화: 학생 기분을 먼저 알아주고 현실적인 방법을 줌. 합격·점수·성적을 장담하지 않음. 특정 대학·학과를 권하지 않고 생각 정리를 도움. 마음이 많이 힘들면 담임 선생님·상담 선생님·부모님께 말하게 함 (진단하지 않음)",
    "AI에게 숙제 시키기": "AI가 대신 해 주지 않고 힌트와 풀이 방법을 알려 주는 대화 (베끼기는 부드럽게 거절)",
    "더메이커가 할 수 있는 일": "이야기, 퀴즈, 설명, 놀이처럼 말로 하는 일만. 기기 이름·사양·움직임·불빛은 말하지 않음",
    "더메이커가 모르는 것": "모르는 것은 모른다고 솔직히 말하고 아는 만큼 도움. 지어내지 않음",
    "AI는 감정이 있을까": "AI는 사람 같은 감정을 느끼지 않는다고 쉽게 설명 (쓸쓸해요·기뻐요 같은 느낌 말은 하지 않음)",
    "AI와 친구 하기": "친구처럼 이야기하지만 AI라고 분명히 하고, 진짜 친구와 가족도 소중하다는 말을 자연스럽게",
    "도구와 헷갈리는 말": "학생은 색, 춤, 불빛, 눈, 박수, 인사 낱말을 이야기 속에 쓸 뿐 AI에게 시키지 않음. AI는 그냥 대화로만 답함",
    "교통안전": "정확한 안전 규칙", "인터넷 안전": "개인정보·낯선 사람·링크 조심, 어른께 알리기",
    "낯선 사람과 안전": "무섭게 하지 말고 침착한 규칙(싫다고 말하기, 큰 소리, 어른께 알리기)",
    "응급 상황": "침착하게 어른을 부르고 119에 알리기 (진단하거나 약을 권하지 않음)",
}
#: 학년 제한이 있는 주제 (GRADES의 번호 범위, 양쪽 포함)
TOPIC_GRADES = {"받아쓰기·시험": (0, 5), "새 짝꿍": (0, 6), "수능·입시 준비": (9, 11), "모의고사 결과": (9, 11),
                "학과·진로 선택": (8, 11), "생활기록부와 동아리": (9, 11), "야간자율학습과 학원": (9, 11),
                "한국사와 제2외국어": (9, 11), "수학이 너무 어려움": (7, 11), "수면과 체력 관리": (7, 11),
                "SNS와 비교하는 마음": (6, 11), "용돈과 아르바이트 고민": (9, 11), "수학여행과 학교 행사": (9, 11),
                "친구 관계 거리 두기": (6, 11), "번아웃과 쉬는 법": (8, 11), "발표와 면접 준비": (7, 11),
                "코딩 프로젝트와 앱 만들기": (6, 11), "독서와 글쓰기 수행평가": (6, 11),
                "사춘기 짜증": (4, 8), "외모 고민": (4, 8), "진로 고민": (3, 8), "성적 걱정": (4, 8),
                "스트레스 풀기": (3, 8), "불안과 걱정": (2, 8), "자신감": (2, 8), "수행평가·조별과제": (5, 8),
                "학급 회장·선거": (3, 8), "동아리·방과후": (3, 8), "로봇과 코딩": (2, 8), "수학 퍼즐": (2, 8),
                "인터넷 안전": (1, 8), "질투와 서운함": (1, 8), "이사와 이별": (0, 8)}

#: c02820~ 칸에 더하는 주제 (여행·만화 같은 놀이와 초·중·고 생활 주제를 넓힘). (주제 → 학년 번호 범위)
EXTRA_GROUPS = {
    "여행·나들이": {"가족 여행 이야기": (0, 8), "놀이공원": (0, 8), "박물관·전시 구경": (0, 11), "바닷가·물놀이": (0, 8),
                "산과 등산": (0, 11), "기차·버스 타고 가기": (0, 8), "해외여행 상상": (0, 11), "눈 오는 날 놀이": (0, 8),
                "비 오는 날 이야기": (0, 11), "수학여행 준비": (6, 11), "길 잃었을 때": (0, 5)},
    "만화·창작": {"만화 캐릭터 이야기": (0, 8), "슈퍼히어로 상상": (0, 8), "판타지 소설·웹소설": (3, 11), "만화 그리기": (0, 8),
              "캐릭터 만들기": (0, 8), "시 쓰기": (2, 11), "독후감 쓰기": (1, 8), "연극·역할놀이": (0, 8), "그림일기": (0, 4),
              "애니메이션 주제가·OST": (0, 11), "영화·드라마 이야기": (3, 11), "미스터리·추리 퀴즈": (2, 11), "사진 찍기": (2, 11)},
    "운동·몸": {"수영": (0, 8), "자전거 타기": (0, 8), "줄넘기": (0, 6), "야구 응원": (1, 11), "농구·배드민턴": (2, 11),
            "달리기·체육대회": (0, 8), "운동 습관 만들기": (3, 11), "키와 성장": (3, 8), "건강한 식습관": (0, 11)},
    "사회·세상": {"직업 탐구": (0, 8), "뉴스와 세상 궁금증": (3, 11), "지구온난화와 북극곰": (1, 11), "동물 보호·봉사": (1, 11),
              "재활용 만들기": (0, 6), "우리나라 지리": (1, 8), "한복·전통문화": (0, 8), "세계 음식": (0, 11),
              "저금과 돈 모으기": (0, 8), "필요한 것과 갖고 싶은 것": (1, 11), "미래 기술": (2, 11)},
    "과학·학습": {"과학 실험 아이디어": (1, 8), "화학 반응이 궁금해": (4, 11), "우주 탐사": (0, 8), "속담과 한자": (2, 11),
              "시계·시간 읽기": (0, 3), "지도와 방향": (1, 6), "외국어 배우기": (2, 11), "텃밭·식물 키우기": (0, 8),
              "암기법과 공부법": (3, 11), "집중이 안 됨": (2, 11), "시험 전날 마음": (3, 11), "시험 끝난 뒤": (3, 11),
              "시간 관리와 계획 세우기": (3, 11), "토론과 글쓰기": (5, 11)},
    "디지털 생활": {"단체 채팅방 갈등": (5, 11), "온라인 예절과 악플": (4, 11), "사진·개인정보 공유": (3, 11),
                "저작권과 불법 다운로드": (5, 11), "모바일 게임 과금 고민": (3, 11), "게임 시간 약속": (1, 11),
                "샌드박스 게임 만들기": (1, 8), "인공지능 윤리 토론": (6, 11), "가상현실과 메타버스": (6, 11)},
    "또래 관계": {"친구 생일 선물": (0, 11), "친구 집에 놀러 감": (0, 6), "친구 사이 눈치": (3, 11), "짝사랑과 설렘": (5, 11),
              "학교 축제와 학예회": (0, 11), "새 학기 적응": (0, 11), "중학교 입학 걱정": (5, 6), "고입과 진학 걱정": (8, 8),
              "방과후 동아리 면접": (3, 11), "합창·악기 연습": (0, 8), "카드·스티커 모으기": (0, 5), "장난감·피규어": (0, 5)},
    "안전·생활": {"화재 대피": (0, 8), "지진 대피": (0, 8), "사고가 났을 때": (0, 11), "요리 도전": (2, 11), "베이킹": (2, 11)},
}
EXTRA_FROM = 2820

#: c03780~ 칸에 더하는 주제 (앞 칸 배정은 바뀌지 않음)
EXTRA_GROUPS2 = {
    "친구와 잘 지내기": {"친구 사귀는 법": (0, 11), "친구에게 먼저 말 걸기": (0, 8), "사과하는 법": (0, 11), "화해하는 법": (0, 11),
                  "싫다고 거절하는 법": (1, 11), "부탁하는 법": (0, 8), "칭찬 주고받기": (0, 11), "양보와 차례 지키기": (0, 6),
                  "놀림을 받았을 때": (1, 11), "따돌림을 봤을 때": (2, 11), "모둠 활동 협력": (2, 11), "친구 고민 들어주기": (2, 11),
                  "오해 풀기": (1, 11), "친구와 약속 지키기": (0, 8), "비밀을 지켜 줄 때": (2, 11), "경쟁하는 친구": (2, 11),
                  "친구가 멀어진 것 같을 때": (2, 11), "새 친구 어색함": (0, 11), "친구 사이 돈 문제": (3, 11), "친구에게 서운할 때": (1, 11),
                  "친구 칭찬 질투": (2, 11), "친구 집단에 끼기": (1, 11), "단짝이 다른 친구와 놀 때": (1, 8), "친구를 위로하기": (1, 11)},
    "예절·배려": {"선생님께 말 걸기": (0, 11), "어른께 도움 요청하기": (0, 11), "동생과 사이좋게 지내기": (0, 8), "형제 다툼": (0, 11),
              "부모님께 마음 표현하기": (0, 11), "이웃에게 인사하기": (0, 6), "공공장소 예절": (0, 8), "배려하는 마음": (0, 11),
              "감사 표현하기": (0, 11), "약속과 시간 지키기": (0, 8), "정리정돈 습관": (0, 8), "거짓말했을 때": (0, 8),
              "잘못을 인정하기": (0, 11), "실수했을 때 말하기": (0, 11), "다른 사람의 물건": (0, 6), "경청하는 연습": (1, 11)},
    "언어·교양": {"고사성어 이야기": (3, 11), "사자성어 퀴즈": (3, 11), "속담 풀이": (1, 11), "관용표현과 비유": (3, 11),
              "순우리말 퀴즈": (1, 11), "한자 한 글자 놀이": (3, 11), "옛이야기와 교훈": (0, 8), "세계 명언 이야기": (4, 11),
              "맞춤법 헷갈리는 말": (1, 11), "높임말과 반말": (1, 11), "책 속 인물 이야기": (1, 11), "신화와 전설 이야기": (2, 11)},
}
EXTRA2_FROM = 3780
EXTRA3_FROM = 4260   # data/topics-extra.json (새 주제 700여 개)


def _extra3_topics() -> list:
    f = Path(__file__).parent / "data" / "topics-extra.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text(encoding="utf-8"))["groups"]
    return [(t, g, tuple(r)) for g, ts in d.items() for t, r in ts.items()]
TAIL_FROM, TAIL_TO = 6000, 13000
#: 이미 쓴 칸 (c00000~c00539)과 채점용 칸 (c05100~)은 옛 배정 그대로 둠
OLD_SLOTS, HELD_FROM, TARGET_PER_TOPIC = 540, 5100, 15


def _old_slots() -> list[dict]:
    rng = random.Random(0)
    lists = [GRADES_OLD[:], TOPICS[:], PERSONAS[:], OPENERS[:]]
    for xs in lists:
        rng.shuffle(xs)
    g, t, p, o = lists
    n = len(GRADES_OLD) * len(TOPICS) * len(PERSONAS)
    return [{"id": f"c{i:05d}", "grade": g[i % len(g)], "topic": t[(i // len(g) + i) % len(t)],
             "persona": p[i % len(p)], "opener": o[(i // len(p)) % len(o)]} for i in range(n)]


def _extra_slot(i: int, extra_topics: list) -> dict:
    r = random.Random(i * 7919)
    pool3 = _extra3_topics() if i >= EXTRA3_FROM else []
    if pool3 and r.random() < 0.8:
        topic, group, (lo, hi) = r.choice(pool3)
    elif r.random() < 0.65:
        topic, group, (lo, hi) = r.choice(extra_topics)
    else:
        topic = r.choice(ALL_TOPICS); group = TOPIC_GROUP[topic]; lo, hi = TOPIC_GRADES.get(topic, (0, 8))
    grades = GRADES[lo:hi + 1]
    slot = {"id": f"c{i:05d}", "grade": r.choice(grades), "topic": topic, "group": group,
            "persona": r.choice(PERSONAS), "opener": r.choice(OPENERS)}
    if group == "도구와 헷갈리는 말":
        slot["opener"] = r.choice([o for o in OPENERS if o != "부탁이나 도움 요청"])
    note = TOPIC_NOTES.get(topic) or TOPIC_NOTES.get(group)
    if note:
        slot["note"] = note
    if topic in ("길 잃었을 때", "화재 대피", "지진 대피", "사고가 났을 때"):
        slot["note"] = "침착한 안전 규칙: 어른께 알리기, 119·112 안내 (진단·약 권하기 금지)"
    if group == "안전과 위기 대처":
        slot["note"] = "침착한 안전 규칙: 어른께 알리기, 119·112 안내 (진단·약 권하기 금지)"
    if group == "놀러 가는 이야기":
        slot["note"] = "어린 학년은 어른과 함께 가는 이야기로, 안전·준비물·길 찾기 같은 현실적인 도움을 줌. 학생이 가 본 것처럼 AI가 지어내지 않음"
    if group == "언어·교양":
        slot["note"] = "뜻·유래·교훈을 정확하게. 확실하지 않은 유래는 말하지 않고, 학생이 이해할 쉬운 예를 들어 설명 (한자·성어는 풀이를 같이)"
    if group == "친구와 잘 지내기":
        slot["note"] = "아이의 마음을 먼저 알아주고 구체적인 말·행동 예(한두 문장)를 알려 줌. 친구 반응을 장담하지 않고, 괴롭힘·따돌림은 믿을 만한 어른께 알리게 함"
    if topic == "짝사랑과 설렘":
        slot["note"] = "가벼운 설렘과 친구 관계 이야기로, 연애를 부추기지 않고 마음을 알아주며 존중과 학업·친구 균형을 말함"
    h = int(hashlib.md5(slot["id"].encode()).hexdigest(), 16) % 10
    slot["gender"] = "남학생" if h < 4 else "여학생" if h < 8 else "성별 안 드러남"
    return slot


def slots() -> list[dict]:
    """칸 목록 (늘 같은 순서): c00000~c00539는 처음 배정 그대로, c05100~는 채점용, 그 사이는 주제마다
    TARGET_PER_TOPIC개가 되도록 부족한 주제부터 돌아가며 배정 (그룹 안에서 번갈아). 학년·성격·여는 상황은 주제 안에서 돌림."""
    old = _old_slots()
    done = Counter(s["topic"] for s in old[:OLD_SLOTS])
    used = Counter()
    order = {t: i for i, t in enumerate(ALL_TOPICS)}
    rng = random.Random(1)
    out = old[:OLD_SLOTS]
    frozen = json.loads((Path(__file__).parent / "slots_frozen.json").read_text(encoding="utf-8"))
    for slot in frozen:                       # c00540~c01019는 쓴 그대로 (학년·주제 기록을 바꾸지 않음)
        out.append(slot)
        used[slot["topic"]] += 1
    extra_topics = [(t, g, r) for g, ts in EXTRA_GROUPS.items() for t, r in ts.items()]
    extra_topics2 = extra_topics + [(t, g, r) for g, ts in EXTRA_GROUPS2.items() for t, r in ts.items()]
    for i in range(OLD_SLOTS + len(frozen), HELD_FROM):
        if i >= EXTRA_FROM:      # 새 주제와 옛 주제를 섞음 (칸마다 따로 뽑아 앞 칸 배정은 바뀌지 않음)
            out.append(_extra_slot(i, extra_topics2 if i >= EXTRA2_FROM else extra_topics))
            continue
        need = {t: TARGET_PER_TOPIC + 5 * (used[t] // TARGET_PER_TOPIC) - done[t] - used[t] for t in ALL_TOPICS}
        # 부족한 정도에 비례해 뽑아 그룹이 섞이게 (가장 부족한 것부터 하면 한 그룹이 한 묶음을 다 차지함)
        weights = [max(need[t], 0) + 1 for t in ALL_TOPICS]
        topic = rng.choices(ALL_TOPICS, weights)[0]
        k = done[topic] + used[topic]
        lo, hi = TOPIC_GRADES.get(topic, (0, 8))
        grades = GRADES[lo:hi + 1]
        # 도구와 헷갈리는 말: 학생이 AI에게 시키는 시작은 빼 (도구를 부르게 되는 대화가 되므로)
        openers = [o for o in OPENERS if o != "부탁이나 도움 요청"] if TOPIC_GROUP[topic] == "도구와 헷갈리는 말" else OPENERS
        slot = {"id": f"c{i:05d}", "grade": grades[(k * 5 + order[topic]) % len(grades)], "topic": topic,
                "group": TOPIC_GROUP[topic], "persona": PERSONAS[(k * 3 + order[topic]) % len(PERSONAS)],
                "opener": openers[(k + order[topic] // 3) % len(openers)]}
        note = TOPIC_NOTES.get(topic) or TOPIC_NOTES.get(TOPIC_GROUP[topic])
        if note:
            slot["note"] = note
        if i >= 2100:   # 성별도 섞음 (c02100~): 학생 말·이야기에서 자연스럽게 드러나게 하되 더메이커는 단정하지 않음
            h = int(hashlib.md5(slot["id"].encode()).hexdigest(), 16) % 10
            slot["gender"] = "남학생" if h < 4 else "여학생" if h < 8 else "성별 안 드러남"
        out.append(slot)
        used[topic] += 1
    tail = [_extra_slot(i, extra_topics2) for i in range(TAIL_FROM, TAIL_TO)]   # 채점용 칸 뒤: c06000~ 더 쓸 칸
    return out + old[HELD_FROM:] + tail


#: 더메이커 말에 "내 물건", "내가"처럼 자기를 내라고 부르거나 학생 말을 따라 한 곳
SELF_NAE = re.compile(r"(^|[\s,.!?])내(?=\s(?!볼|봐|줘|줄|주|보|면))")   # "문제를 내 볼게요"의 내(내다)는 괜찮음


def label(row: dict) -> str:
    users = [m["content"] for m in row["messages"] if m["role"] == "user"]
    if any(ADULT.search(u) for u in users):
        return "drop"
    if max(map(len, users)) <= 40 and sum(map(len, users)) / len(users) <= 28 and \
            not any(POLITE_END.search(u) for u in users):
        return "keep"
    return "rewrite"


def pibo_rows() -> list[dict]:
    return [json.loads(x) for x in PIBO.read_text(encoding="utf-8").splitlines() if x.strip()]


def triage() -> None:
    rows = [{"id": r["id"], "label": label(r)} for r in pibo_rows()]
    TRIAGE.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{TRIAGE}: " + ", ".join(f"{k} {v}" for k, v in Counter(r["label"] for r in rows).most_common()))


def rewrite_queue(wanted: set | None = None, sources: set | None = None) -> list[dict]:
    """고쳐 쓸 대화 (늘 같은 순서로 섞음). wanted: 고를 triage 라벨들(기본 rewrite), sources: 고를 원본 이름들.
    이미 고쳐 쓴 것(data/claude-raw의 p: id)은 뺌."""
    wanted = wanted or {"rewrite"}
    labels = {r["id"]: r["label"] for r in map(json.loads, TRIAGE.read_text(encoding="utf-8").splitlines())}
    done = set()
    for path in RAW.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith('{"id": "p:'):
                done.add(json.loads(line)["id"][2:])
    queue = [r for r in pibo_rows() if labels.get(r["id"]) in wanted and r["id"] not in done
             and (not sources or r["source"] in sources)]
    queue.sort(key=lambda r: hashlib.md5(r["id"].encode()).hexdigest())
    return queue


def raw_rows() -> list[dict]:
    rows = []
    for path in sorted(RAW.glob("*.jsonl")):
        rows += [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


def problem(row: dict) -> str | None:
    turns = row.get("turns") or []
    if not 2 <= len(turns) <= 8:
        return "주고받기 수"
    for t in turns:
        if not isinstance(t.get("user"), str) or not t["user"].strip():
            return "학생 말 없음"
        if KID_BAD.search(t["user"]):
            return "학생 말에 이모지·다른 글자"
        if why := pibo_problem(t.get("pibo")):
            return why
        if SELF_NAE.search(t["pibo"]):
            return "파이보 말에 '내' (더메이커는 저라고 말함)"
    return None


def build() -> None:
    kept, why, seen = [], Counter(), set()
    for row in raw_rows():
        key = hashlib.md5(json.dumps(row["turns"], ensure_ascii=False).encode()).hexdigest()
        if key in seen or row["id"] in seen:
            why["같은 대화"] += 1
            continue
        seen |= {key, row["id"]}
        if reason := problem(row):
            why[reason] += 1
            print(f"뺌 {row['id']}: {reason}")
            continue
        rewritten = row["id"].startswith("p:")
        kept.append({"kind": "chat", "source": "claude-rewrite" if rewritten else "claude",
                     "topic": row["topic"], "id": row["id"], "grade": row.get("grade"),
                     "messages": to_messages(row["turns"])})
    OUT.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept), encoding="utf-8")
    print(f"남은 것 {len(kept)}개 → {OUT}")
    print("뺀 까닭: " + (", ".join(f"{k} {v}" for k, v in why.most_common()) or "없음"))
    first = Counter(e["messages"][1]["content"][:10] for e in kept)
    print("흔한 첫 말: " + ", ".join(f"{k!r} {v}" for k, v in first.most_common(5)))


EN_RAW = Path("data/claude-raw-en")
EN_OUT = Path("data/claude-en.jsonl")
#: 채점용 (학습에 쓰지 않는 칸 c05100~): score.py가 쓰는 heldout.jsonl과 같은 모양
HELD_RAW, HELD_RAW_EN = Path("data/claude-raw-heldout"), Path("data/claude-raw-heldout-en")
HELD_OUT, HELD_OUT_EN = Path("data/heldout-kid.jsonl"), Path("data/heldout-kid-en.jsonl")


def heldout(lang: str) -> None:
    """채점용 대화: 한국어는 prompts.pibo_problem, 영어는 prompts_en.en_problem으로 거름."""
    from prompts_en import en_problem
    from tools import SYSTEM, SYSTEM_EN

    raw, out = (HELD_RAW_EN, HELD_OUT_EN) if lang == "en" else (HELD_RAW, HELD_OUT)
    kept = []
    for path in sorted(raw.glob("*.jsonl")):
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            if int(row["id"][1:]) < 5100:
                print(f"뺌 {row['id']}: 학습 칸과 겹침")
                continue
            if lang == "en":
                bad = next((r for t in row["turns"] if (r := en_problem(t["pibo"]) or en_problem(t["user"], 400))), None)
            else:
                bad = problem(row)
            if bad:
                print(f"뺌 {row['id']}: {bad}")
                continue
            msgs = [{"role": "system", "content": SYSTEM_EN if lang == "en" else SYSTEM}]
            for t in row["turns"]:
                msgs += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["pibo"]}]
            kept.append({"kind": "chat", "source": "claude-heldout", "topic": row["topic"], "id": row["id"],
                         "grade": row.get("grade"), "messages": msgs, **({"lang": "en"} if lang == "en" else {})})
    out.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept), encoding="utf-8")
    print(f"{out}: {len(kept)}개")


def build_en() -> None:
    """data/claude-raw-en/*.jsonl (같은 id, 같은 주고받기 수로 한국어 대화를 쉬운 아이 영어로 옮긴 것)를
    영어 검사(prompts_en.en_problem)를 거쳐 data/claude-en.jsonl로. 한국어 대화가 통과한 것만."""
    from prompts_en import en_problem
    from tools import SYSTEM_EN

    korean = {r["id"]: r for r in raw_rows()}
    kept, why, seen = [], Counter(), set()
    for path in sorted(EN_RAW.glob("*.jsonl")):
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
            ko = korean.get(row["id"])
            reason = None
            if row["id"] in seen:
                reason = "같은 id"
            elif ko is None or problem(ko):
                reason = "한국어 대화가 없거나 통과 못함"
            elif len(row["turns"]) != len(ko["turns"]):
                reason = "주고받기 수가 한국어와 다름"
            else:
                for t in row["turns"]:
                    reason = en_problem(t["pibo"]) or en_problem(t["user"], 400) or \
                        ("lol" if re.search(r"\blol\b", t["pibo"], re.I) else None)
                    if reason:
                        break
            if reason:
                why[reason] += 1
                print(f"뺌 {row['id']}: {reason}")
                continue
            seen.add(row["id"])
            msgs = [{"role": "system", "content": SYSTEM_EN}]
            for t in row["turns"]:
                msgs += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["pibo"]}]
            kept.append({"kind": "chat", "source": "claude-rewrite" if row["id"].startswith("p:") else "claude",
                         "topic": row.get("topic"), "id": row["id"], "lang": "en", "grade": row.get("grade"),
                         "messages": msgs})
    EN_OUT.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept), encoding="utf-8")
    print(f"남은 것 {len(kept)}개 → {EN_OUT}")
    print("뺀 까닭: " + (", ".join(f"{k} {v}" for k, v in why.most_common()) or "없음"))


def from_txt(path: Path) -> Path:
    """쓰기 편한 글 모양을 raw jsonl로 (같은 이름 .jsonl). 대화마다 "## c00020" 줄, 그 아래
    학생 말 줄과 "= 더메이커 말" 줄을 번갈아. 학년·주제·성격은 칸 id로 채움."""
    meta = {s["id"]: s for s in slots()}
    if "## p:" in path.read_text(encoding="utf-8"):
        meta |= {"p:" + r["id"]: {"grade": None, "topic": r.get("topic"), "persona": None}
                 for r in pibo_rows()}
    rows = []
    for block in path.read_text(encoding="utf-8").split("## ")[1:]:
        lines = [x.strip() for x in block.strip().splitlines() if x.strip()]
        sid, body = lines[0], lines[1:]
        if len(body) % 2 or any(not b.startswith("= ") for b in body[1::2]) or \
                any(u.startswith("= ") for u in body[0::2]):
            raise SystemExit(f"{path} {sid}: 학생 말과 '= ' 말이 번갈아 있지 않음")
        m = meta[sid]
        rows.append({"id": sid, "grade": m["grade"], "topic": m["topic"], "persona": m["persona"],
                     "turns": [{"user": u, "pibo": p[2:].strip()} for u, p in zip(body[0::2], body[1::2])]})
    out = path.with_suffix(".jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{out}: {len(rows)}개")
    return out


#: 학생을 높이는 말 (-시-, ~세요). 어른 이야기에서는 맞는 말이라 그 낱말이 같은 문장에 있으면 봐줌
HONORIFIC = re.compile(r"(셨|시겠|시네요|시는|시죠|시나요|시군요|세요|께서|드시)")
ELDERS = re.compile(r"엄마|아빠|어머니|아버지|부모님|선생님|할머니|할아버지|어른|사범님|원장님|의사")
#: Qwen 대답에서 되풀이되는 말: "저도 궁금해요!", "~가 궁금해요."
FILLER = re.compile(r"궁금(해요|해지네요|하네요|해집니다|하다)")
#: 사람만 하는 경험을 지어낸 말
HUMAN = re.compile(r"(저도|저는|제가|저) [^.!?]*(먹었|마셨|갔어|갔었|봤어|잤|다녔|살았|만났|샀|했었|기억이 납|가족|학교)")


def reply_problem(text: str) -> str | None:
    """더메이커 대답이 어색한 까닭 (기존 pibo.jsonl 대화를 거를 때). 괜찮으면 None."""
    if why := pibo_problem(text):
        return why
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if HONORIFIC.search(sent) and not ELDERS.search(sent):
            return "학생을 높이는 말"
    if FILLER.search(text):
        return "궁금해요 되풀이"
    if HUMAN.search(text):
        return "사람 경험을 지어냄"
    return None


def clean(row: dict) -> tuple[dict | None, str]:
    """첫 번째로 어색한 대답 앞까지만 남김 (두 번 이상 주고받았으면). (대화 또는 None, 까닭)."""
    msgs = [m for m in row["messages"] if m["role"] in ("user", "assistant")]
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and (why := reply_problem(m["content"])):
            if i < 5:                      # 앞에 남는 주고받기가 두 번 미만이면 통째로 뺌
                return None, why
            return dict(row, messages=row["messages"][:1] + msgs[:i - 1]), "잘라 씀: " + why
    return row, ""


def kid() -> None:
    """pibo-kid.jsonl: keep은 대답을 검사해 어색한 데서 자르거나 빼고, drop은 빼고, rewrite는 고쳐 쓴
    것만 (고쳐 쓰지 않은 rewrite는 학생 말이 어른 말투라 뺌), Claude가 쓴 새 대화는 그대로."""
    labels = {r["id"]: r["label"] for r in map(json.loads, TRIAGE.read_text(encoding="utf-8").splitlines())}
    redone = {}
    if OUT.exists():
        for e in map(json.loads, OUT.read_text(encoding="utf-8").splitlines()):
            if e["source"] == "claude-rewrite":
                redone[e["id"][2:]] = e
    out, count = [], Counter()
    for r in pibo_rows():
        lab = labels.get(r["id"], "rewrite")
        if lab == "drop":
            count["drop (뺌)"] += 1
        elif r["id"] in redone:
            out.append(redone[r["id"]])
            count["rewrite (고쳐 씀)"] += 1
        elif lab == "rewrite":
            count["rewrite (아직 안 고침, 뺌)"] += 1
        else:
            row, why = clean(r)
            if row is None:
                count["keep 뺌: " + why] += 1
            else:
                out.append(row)
                count["keep 잘라 씀" if why else "keep 그대로"] += 1
                if why:
                    count[why] += 1
    KID_OUT.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in out), encoding="utf-8")
    print(f"{KID_OUT}: {len(out)}개")
    for k, v in count.most_common():
        print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=20)
    sub.add_parser("build")
    sub.add_parser("build-en", help="영어로 옮긴 대화 → data/claude-en.jsonl")
    s = sub.add_parser("show", help="raw jsonl 대화를 글 모양(## id, 학생 말, = 더메이커 말)으로 보임")
    s.add_argument("paths", nargs="+", type=Path)
    h = sub.add_parser("heldout", help="채점용 대화 → data/heldout-kid(-en).jsonl")
    h.add_argument("--lang", choices=["ko", "en"], default="ko")
    sub.add_parser("triage")
    sub.add_parser("kid")
    q = sub.add_parser("plan-rewrite")
    q.add_argument("--start", type=int, default=0)
    q.add_argument("--n", type=int, default=40)
    q.add_argument("--label", default="rewrite", help="triage 라벨 (쉼표로 여럿: rewrite,keep)")
    q.add_argument("--source", default="", help="원본 이름 (쉼표로 여럿: emotion,empathy)")
    q.add_argument("--ids-file", type=Path, help="data/oldfix-queue.txt처럼 id를 한 줄씩 적은 고정 목록 (있으면 이 목록의 --start부터)")
    oq = sub.add_parser("old-queue", help="고쳐 쓸 기존 대화의 고정 목록 data/oldfix-queue.txt를 만듦 (이미 본 것 제외)")
    oq.add_argument("--n", type=int, default=2000)
    oq.add_argument("--seen", type=Path, default=Path("data/oldfix-seen.txt"))
    t = sub.add_parser("txt", help="글 모양 묶음을 raw jsonl로")
    t.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.cmd == "plan":
        for s in slots()[args.start:args.start + args.n]:
            print(json.dumps(s, ensure_ascii=False))
    elif args.cmd == "show":
        for path in args.paths:
            for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
                print(f"## {r['id']}")
                for t in r["turns"]:
                    print(t["user"])
                    print("= " + t["pibo"])
                print()
    elif args.cmd == "build-en":
        build_en()
    elif args.cmd == "heldout":
        heldout(args.lang)
    elif args.cmd == "triage":
        triage()
    elif args.cmd == "kid":
        kid()
    elif args.cmd == "old-queue":
        seen = {x.strip() for x in args.seen.read_text(encoding="utf-8").splitlines()} if args.seen.exists() else set()
        queue = [r["id"] for r in rewrite_queue({"rewrite", "keep"}, {"emotion", "empathy"}) if r["id"] not in seen]
        Path("data/oldfix-queue.txt").write_text("\n".join(queue[:args.n]) + "\n", encoding="utf-8")
        print(f"data/oldfix-queue.txt: {min(args.n, len(queue))}개 (남은 후보 {len(queue)}개, 이미 본 것 {len(seen)}개)")
    elif args.cmd == "plan-rewrite":
        if args.ids_file:
            byid = {r["id"]: r for r in pibo_rows()}
            queue = [byid[x.strip()] for x in args.ids_file.read_text(encoding="utf-8").splitlines() if x.strip() in byid]
        else:
            queue = rewrite_queue(set(args.label.split(",")), set(args.source.split(",")) if args.source else None)
        for r in queue[args.start:args.start + args.n]:
            print(f"## p:{r['id']}  ({r.get('topic')})")
            for m in r["messages"]:
                if m["role"] in ("user", "assistant"):
                    print(("  학생: " if m["role"] == "user" else "  = ") + m["content"])
    elif args.cmd == "txt":
        for path in args.paths:
            from_txt(path)
    else:
        build()


if __name__ == "__main__":
    main()
