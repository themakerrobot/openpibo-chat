# 파이보 대화 모델 적용 지시서 (파이보·파이브레인 세션용)

이 문서를 받은 세션이 할 일: 파이보에 새 대화 모델(Qwen3.5-0.8B 파인튜닝, Q4_K_M)을 띄워
시험하고, 기존 파이보 앱이 이 모델(대화)과 앱의 동작 명령을 쓰도록 붙일 방법을 조사·제안합니다.
**서비스 설정 변경과 기존 파일 덮어쓰기는 사용자 확인 뒤에만** 합니다.

대답과 보고는 한국어 존댓말, 결론 먼저. 부품명·라이브러리·핀 이름은 영문 그대로.
전압·전류·핀맵 등 하드웨어 사실은 추측하지 말고, 확인 안 되면 "확인 필요"라고 적습니다.

---

## 1. 지금 확인된 사실

| 항목 | 값 |
| --- | --- |
| 파이보 | Raspberry Pi 4B, 램 2GB (`free -m` 전체 1796MB), IP `192.168.68.121`, 사용자 `pi` |
| 학습 PC | Ubuntu, IP `192.168.68.118`, 사용자 `circulus`, 저장소 `~/openpibo-chat` (브랜치 `claude/inspiring-curie-s3a7wn`) |
| 기존 llama.cpp | `/home/pi/.llama.cpp` (빌드 5218, 2025-04). **Qwen3.5를 못 읽음** |
| 새 llama.cpp | `/home/pi/llama.cpp-new` (빌드 11889, 커밋 `d854d98`, PC와 같은 커밋) |
| 기존 서버 | 서비스(systemd)가 `/home/pi/.llama.cpp/build/bin/llama-server -m /home/pi/.model/llm/llm-model.gguf --ctx-size 2048 --log-disable --host 0.0.0.0 --port 50020`로 젬마3 1B를 띄움. 서비스 파일은 `~/openpibo-os/system/` 아래 (이름 확인 필요) |
| 새 모델 | 최종 `themaker-0.8b-Q4_K_M.gguf` (PC `out/`, 약 320MB). 그 전에는 `~/pibo-take/pibo-qwen3.5-0.8b-school-small-Q4_K_M.gguf` |
| 앱 | `~/pibo-take/pibo_chat.py`, `~/pibo-take/tools.py` (표준 라이브러리만 씀) |
| 파이썬 | `sudo ~/.pyenv/bin/python3` 로 실행 (openpibo가 여기에 설치됨) |

### 파이보 실측 (`pi_bench.sh`, 새 llama.cpp, `-c 2048 -t 4 -np 1`)

| 모델 | 최대 메모리 | 대화 한 번 | 도구 첫 호출 | 도구 다음 호출 | 읽기/말하기 (토큰/초) |
| --- | --- | --- | --- | --- | --- |
| 젬마3 1B (지금) | 872MB | 21.5초 | 불가 | 불가 | 11.4 / 4.2 |
| Qwen3.5 파이보 | 515MB | 11.5초 | 67.3초 (도구 목록 1천 토큰 처음 읽기) | 11.4초 | 16.7 / 6.3 |

PC 채점 (초·중·고 질문 700개, 선생 모델 채점): 대화 4.83/5, 도구 정답률 96%, 틀린 사실 6%.
젬마3 1B는 대화 3.27, 도구 없음.

### 알려진 함정 (이미 고침)

- 새 llama-server는 대화 기록 속 도구 호출에 `"type": "function"`이 없으면 **500 오류**.
  `pibo_chat.py`의 `run_calls`가 붙입니다. 다른 앱에서 직접 부를 때도 꼭 붙일 것.
- llama-server 웹 화면은 **도구 목록과 시스템 글을 보내지 않습니다**: 웹 화면에서는 도구가
  절대 안 되고, 옛 GGUF는 엉뚱한 말("파이썬으로 웹소설을 만들었어요")을 했습니다. 새 GGUF
  템플릿은 시스템 글이 없으면 학습 때의 것을 넣습니다 (`export.py --retemplate`).
- 서버를 셸에서 `&`로만 띄우면 그 창의 신호에 같이 꺼질 수 있어 `setsid nohup`으로 띄웁니다.
- 예전 방식(모델이 도구를 고름)은 도구 목록을 읽느라 시작에 40~70초 걸렸습니다.
  지금은 대화 모델이 도구를 모르고 앱이 동작을 정해, 시작이 몇 초입니다.

---

## 2. 하지 말 것

- `/home/pi/.model/llm/llm-model.gguf`(젬마)와 기존 서비스 파일을 백업 없이 덮어쓰지 않습니다.
  서비스는 `stop`만 하고 `disable` 하지 않습니다 (재부팅하면 원래대로).
- llama-server를 두 개 동시에 띄우지 않습니다 (램 2GB). 50020 포트가 겹치면 새 서버는 안 뜹니다.
- 파이보에서 `-j4`로 빌드하지 않습니다 (메모리 부족으로 멈춤, `-j2`).
- 모델이나 데이터를 파이보에서 다시 만들지 않습니다. 학습·변환은 PC 세션(openpibo-chat)이 합니다.

---

## 3. 할 일

### A. 최신 파일 받기

최종 모델은 PC에서 만드는 중입니다: `~/openpibo-chat/out/themaker-0.8b-Q4_K_M.gguf`
(모든 기기 공통 대화 모델, 도구 없음). 나오기 전까지는 파이보에 이미 있는 예전 모델
(`~/pibo-take/pibo-qwen3.5-0.8b-school-small-Q4_K_M.gguf`)로 앱을 시험합니다 (대화와 동작 모두 됨).

파이보에서:

```bash
cd ~/pibo-take
scp circulus@192.168.68.118:openpibo-chat/{pibo_chat.py,tools.py,pi_bench.sh} .
# 최종 모델이 나오면
scp circulus@192.168.68.118:openpibo-chat/out/themaker-0.8b-Q4_K_M.gguf .
```

### B. 서버 띄우기

```bash
pgrep -af llama-server          # 떠 있으면 기존 서비스: sudo systemctl stop <서비스 이름>
setsid nohup /home/pi/llama.cpp-new/build/bin/llama-server \
  -m /home/pi/pibo-take/themaker-0.8b-Q4_K_M.gguf \
  --jinja --ctx-size 2048 -np 1 --cache-ram 0 -t 4 -ub 256 --host 0.0.0.0 --port 50020 \
  > ~/qwen-server.log 2>&1 < /dev/null &
for i in $(seq 1 30); do curl -s http://127.0.0.1:50020/health | grep -q ok && { echo 준비됨; break; }; sleep 2; done
free -m | head -2
```

옵션 뜻: `--jinja` 도구 호출, `-np 1` 자리 하나, `--cache-ram 0` 지난 질문을 램에 쌓지 않음
(새 llama.cpp 기본은 8GB까지), `-t 4` 코어 4개, `-ub 256` 프롬프트 읽을 때 메모리 절약.

### C. 로봇으로 시험 (`pibo_chat.py`)

**구조가 바뀌었습니다 (2026-10-26).** 대화 모델은 도구를 모르고 모든 기기 공통입니다. 도구 목록
(약 1천 토큰) 때문에 파이보에서 너무 느려서 뺐습니다. 동작은 앱이 말 끝을 보고 바로 실행합니다
(`pibo_chat.COMMANDS`). 지금 파이보에 있는 GGUF(도구를 배운 예전 모델)로도 이 방식이 됩니다.

```bash
cd ~/pibo-take
sudo ~/.pyenv/bin/python3 pibo_chat.py --url http://127.0.0.1:50020
```

`(준비 …초)`(몇 초) 뒤 `나:`가 나오면 아래를 차례로 입력하고, 화면 전체(시간 줄 포함)를 기록합니다.

| 말 | 기대 |
| --- | --- |
| 안녕 / 너는 누구야? | 모델이 말로 대답 |
| 춤춰 줘 / 춤 좀 춰 봐 | `[동작] set_motion(dance…)` 실제로 춤, "신나게 춤춰 볼게요!" |
| 어제 춤 학원 갔어 | 동작 없이 모델이 대답 (이야기 속 낱말은 동작이 아님) |
| 앞으로 가 / 뒤로 가 / 왼쪽으로 돌아 | `forward1` / `backward1` / `left` |
| 인사해 줘 / 손 흔들어 / 박수 쳐 / 하이파이브 | `greeting` / `wave1` / `clapping1` / `hifive` |
| 눈 빨갛게 해 줘 → 눈 꺼 줘 | `eye_on` 빨강 → `eye_off` |
| 멈춰 / 그만 움직여 / stop | 바로 `stop` |
| 앞에 뭐 있어? | `find_objects` (카메라), "앞에 컵, 사과가 보여요" |
| 배터리 얼마나 남았어? | `get_battery`, "배터리가 …% 남았어요" |
| 오늘 서울 날씨 어때? | `get_weather` 결과를 모델이 말로 (인터넷 필요) |
| 대화만 하자 → 춤춰 줘 | 움직이지 않고 "지금은 대화만 하기로 했어요…" |
| 이제 움직여도 돼 → 춤춰 줘 | 다시 춤 |
| 학교에서 친구랑 싸웠어 | 공감하는 짧은 존댓말 |

동작 명령이 안 먹힌 말, 동작이 아닌데 움직인 말이 있으면 그 문장을 그대로 보고합니다
(`COMMANDS`의 말 모양을 고칩니다). 시간 줄: `[동작] … (…초)` 실행, `(읽기 N토큰, 말하기 M토큰)`
모델, `(모두 …초)` 전체.

동작이 `실패: …`로 나오면: 기존 openpibo 프로세스(서비스, 웹 앱)가 모터·카메라를 쥐고 있는지
`ps aux | grep -i pibo`로 확인하고 그 내용을 보고합니다 (임의로 죽이지 말 것).

### D. 하드웨어 확인 (보고만)

- 춤처럼 서보 여러 채널이 동시에 움직일 때 리셋, 화면 꺼짐, 소리 끊김이 있는지.
  어댑터 연결 상태와 배터리만 쓸 때 둘 다. 전원 전압 강하가 의심되면 측정값 없이 단정하지
  말고 "확인 필요"로 보고.
- 실행 중 `free -m`, `vcgencmd measure_temp`(있으면) 값.

### E. 기존 앱 조사 (코드 수정 전에 보고)

```bash
grep -rn "50020\|llama\|chat/completions\|/completion" ~/openpibo-os 2>/dev/null | head -50
grep -rln "llama-server" ~/openpibo-os/system/ /etc/systemd/system/ 2>/dev/null
```

알아낼 것:
1. 50020 포트를 부르는 앱과 파일, 부르는 방식 (`/v1/chat/completions` 메시지 목록인지,
   젬마 형식 글을 직접 만들어 `/completion`으로 보내는지). 후자면 새 모델에서 깨집니다.
2. 그 앱이 보내는 시스템 글, 대화 기록 처리.
3. STT·TTS를 부르는 곳, STT가 언어를 스스로 알아듣는지 미리 정해야 하는지, TTS 목소리
   (한국어·영어) 고르는 방법.
4. openpibo `Motion`, `Device`, 카메라를 그 앱이 이미 잡고 있는지 (같이 열면 충돌 가능).

### F. 통합 방법 (제안 → 사용자 확인 뒤 적용)

`pibo_chat.py`는 그대로 import해서 씁니다. 앱이 이미 `Motion`/`Device`를 갖고 있으면
`pibo_chat._parts`에 넣어 같은 객체를 쓰게 합니다.

```python
import pibo_chat
from tools import system_for, tool_list

URL = "http://127.0.0.1:50020"
pibo_chat.SYSTEM = system_for("파이보")   # 다른 기기: system_for("파이브레인", "코딩 로봇, 서큘러스에서 만듦")
pibo_chat.TOOLS = tool_list()             # 다른 기기: tool_list(["eye_on", "eye_off"]) 처럼 있는 것만
pibo_chat.warm_up(URL, True)              # 앱 시작 때 한 번 (몇 초)
history: list = []
use_tools = True
# 앱이 이미 연 객체가 있으면: pibo_chat._parts["motion"] = motion; pibo_chat._parts["device"] = device


def reply_to(text: str) -> str:
    """STT 결과 한 줄 → 할 말 (동작은 안에서 실행됨)."""
    global use_tools
    was, use_tools = use_tools, pibo_chat.tool_switch(text, use_tools)
    if was != use_tools:                 # "대화만 하자" / "움직여도 돼"
        history.clear()
        return pibo_chat.SWITCH_REPLY[use_tools]
    try:
        return pibo_chat.turn(URL, history, text, True, use_tools)
    except OSError:                      # 서버 오류·연결 끊김
        history.clear()
        return "잠깐 문제가 생겼어요. 한 번만 다시 말해 줄래요?"


# 사용: say = reply_to(stt_text); tts(say, lang=pibo_chat.lang_of(say))  # "ko" 또는 "en"
```

`turn()`이 하는 일: "멈춰/stop/freeze"는 바로 `stop`, 동작 명령(`COMMANDS`, 말 끝이 명령일 때만)은
바로 실행하고 정해진 말, 나머지는 모델에 최근 3번의 주고받기와 함께 묻습니다 (도구 목록 없이).

### F-2. 다른 기기(파이브레인 등)

모델은 하나입니다. 기기 이름·소개는 시스템 글로, 동작은 그 기기 앱이 처리합니다. 이름을 시스템
글에서 읽는 건 최종 모델(themaker)부터입니다 (예전 파이보 모델은 이름을 바꿔도 "파이보"라고 할 수 있음).
시스템 글을 보내지 않으면 "더메이커"로 대답합니다.

```bash
python3 pibo_chat.py --url http://127.0.0.1:50020 --name 파이브레인 \
  --about "코딩 로봇, 서큘러스에서 만듦, 하얀색" --tools eye_on,eye_off,get_weather
```

`--tools`/`--motions`로 그 기기에 있는 동작만 켭니다 (없는 동작 명령은 그냥 대화로 넘어감).
파이브레인에만 있는 동작은 `COMMANDS`에 말 모양과 실행 함수를 더하면 됩니다 (모델 재학습 불필요).

### G. 서비스 전환 (사용자가 하라고 할 때만)

서비스 파일과 젬마 모델을 백업한 뒤 `ExecStart`만 바꿉니다.

```
ExecStart=/home/pi/llama.cpp-new/build/bin/llama-server -m /home/pi/.model/llm/llm-model.gguf --jinja --ctx-size 2048 -np 1 --cache-ram 0 -t 4 -ub 256 --log-disable --host 0.0.0.0 --port 50020
```

`sudo systemctl daemon-reload && sudo systemctl restart <서비스>` 후 `/health`와 C의 시험을
다시 합니다. 되돌리기: 백업한 서비스 파일과 `llm-model-gemma.gguf`를 원래 이름으로.

---

## 4. 보고할 것

1. C의 화면 전체 (시간 줄 포함), 실패한 도구와 그 메시지
2. `free -m` (서버만 / 앱까지 띄운 상태), 온도
3. D의 하드웨어 관찰
4. E의 조사 결과 (파일 경로와 부르는 방식), F를 적용하려면 바꿀 파일과 줄
5. 확인 필요로 남은 것

## 5. 아직 확인 필요

- 50020 포트를 쓰는 기존 앱의 호출 방식
- STT 언어 자동 인식 여부, TTS 영어 목소리
- `openpibo.collect.Weather().search()`가 받는 지역 이름 (학습 데이터: 서울, 부산, 대구, 인천,
  광주, 대전, 제주, 전국)
- 카메라 모듈 경로 (`openpibo.vision_camera` / `openpibo.vision`), 검출기 출력 형식
- `tools.py`의 동작 28가지가 이 파이보의 motion_db에 모두 있는지
- 도구 실행에 `sudo`가 꼭 필요한지
