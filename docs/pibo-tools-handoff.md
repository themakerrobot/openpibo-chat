# 파이보 도구 지시서 (파이보 세션용, 최종)

이 문서를 받은 세션이 할 일: 대화 모델 themaker(한국어 `themaker-ko`, 영어 `themaker-en`)가 학생 말을 보고 부르는
**도구 3개(불빛·보기·움직이기)**를 파이보의 openpibo 함수로 실행하게 붙입니다.

- 먼저 참고용 앱 `pibo_chat.py`로 파이보에서 시험합니다. 그다음 기존 음성 대화 앱에 붙일 곳을 찾아 **설계안을 보고**합니다.
- 서비스 설정 변경과 기존 파일 덮어쓰기는 **사용자 확인 뒤에만** 합니다.
- 모델 서버를 고르고 띄우는 것은 `docs/model-select-handoff.md`(모델 고르기 블록)의 일입니다. 이 문서는 도구만 다룹니다.

대답과 보고는 한국어 존댓말, 결론 먼저. 부품명·라이브러리·핀 이름은 영문 그대로. 전압·전류·핀맵 등 하드웨어 사실은
추측하지 말고, 확인 안 되면 "확인 필요"라고 적습니다.

---

## 1. 모델과 앱의 약속

앱은 매번 아래 두 가지를 학습 때와 **글자까지 똑같이** 보냅니다. 바꾸면 모델을 다시 학습해야 합니다.

| 보내는 것 | 한국어 모델 (`themaker-ko`) | 영어 모델 (`themaker-en`) |
| --- | --- | --- |
| 시스템 글 | `tools.SYSTEM` | `tools.SYSTEM_EN` |
| 도구 목록 | `tools.TOOLS` | `tools.TOOLS` (같음. 색·동작 값도 같은 한국어 이름) |
| 참고 앱 | `pibo_chat.py` | `pibo_chat.py --lang en` |

모델은 어느 기기에서든 "서큘러스에서 만든 학생들을 위한 교육용 AI 더메이커(TheMaker)"로 이야기하고, 기기 이름이나
사양(키, 색, 모터, 배터리 …)은 말하지 않습니다 (물으면 잘 모른다고 함).

| 도구 | 인자 | 파이보 함수 |
| --- | --- | --- |
| `light` | `color`: 빨강·주황·노랑·초록·파랑·하늘·보라·분홍·하양·끄기 | `Device().eye_on(r, g, b)` / `eye_off()` (`openpibo/device.py` 302행). 색 → RGB는 `tools.COLORS` |
| `look` | 없음 | `Camera().read()` + `Detect().detect_object(img)` (`openpibo/vision_detect.py` 220행) |
| `move` | `action`: 인사·춤·손흔들기·박수·하이파이브·끄덕이기·도리도리·앞으로·뒤로·왼쪽·오른쪽 | `Motion().set_motion(이름)` (`openpibo/motion.py` 375행). 동작 → motion_db 이름은 `tools.ACTIONS` (여럿이면 하나를 골라 씀) |

**앱이 지킬 것** (`pibo_chat.py`에 이미 그렇게 되어 있음)
- 모델이 도구를 부를 때는 호출 앞에 짧은 말("빨간색으로 바꿀게요!")을 함께 냅니다. 앱은 그 말을 하고 실행합니다.
  `look`은 본 것을 덧붙입니다 ("한번 볼게요. 앞에 컵이 보여요." / "Let me take a look. I can see a cup.").
  대답 끝의 빈 줄은 `strip()`으로 지웁니다.
- 대화 기억(history)에는 학생 말과 실제로 한 말만 남깁니다 (도구 호출·결과 메시지는 넣지 않음). 최근 3번 주고받기.
- **"멈춰"·"정지"·"stop"·"freeze"**는 안전을 위해 모델에 묻지 않고 앱이 바로 멈춥니다 (`STOP`).
- **자리를 옮기는 동작**(앞으로·뒤로·왼쪽·오른쪽)은 학생 말에 방향 낱말(`MOVE_WORDS`: 앞·이리·다가·뒤·물러·왼·오른,
  forward·come·back·left·right)이 있을 때만 실행합니다 (`blocked()`). 채점에서 "우리 반 친구가 인사를 잘 안 해"에
  `move(뒤로)`를 부른 일이 있었습니다. 책상에서 떨어지지 않게 하는 장치이니 기존 앱에 옮길 때 꼭 같이 옮깁니다.
- 도구 이름·인자가 목록에 없으면 실행하지 않습니다 (`check_call()`).
- 시스템 글 + 도구 목록은 약 530토큰(파이보에서 읽는 데 약 30초)입니다. 앱을 켤 때 `warm_up`으로 한 번 읽혀 두면
  서버가 앞부분을 기억해 대화 때는 새로 한 말만 읽습니다.
- "대화만 하자"·"움직이지 마"는 도구를 끄고, "이제 움직여도 돼"는 다시 켭니다 (`tool_switch()`).

---

## 2. 지금 확인된 사실

| 항목 | 값 |
| --- | --- |
| 파이보 | Raspberry Pi 4B, 램 2GB, IP `192.168.68.121`, 사용자 `pi` |
| 학습 PC | Ubuntu, IP `192.168.68.118`, 사용자 `circulus`, 저장소 `~/openpibo-chat` |
| llama.cpp | `/home/pi/.llama.cpp` (그 자리에서 다시 빌드한 새 빌드). `--jinja`가 있어야 도구 호출을 읽음 |
| 모델 파일 | 사용자가 `/home/pi/.model/llm/`에 따로 옮겨 둠. `themaker-0.8b-Q4_K_M.gguf` sha256 `2a8799ec40d2f9a3facc035d7ab7eabc0fcc61b5e5637d7ddbb9ea453b21970c`. 영어판 `themaker-en-0.8b-Q4_K_M.gguf`는 PC 학습 뒤 |
| 파이썬 | `sudo ~/.pyenv/bin/python3` (openpibo가 여기 있고, Device·카메라는 root 필요) |
| 실측 (같은 크기 모델) | 서버 최대 515MB, 대화 한 번 11.5초, 읽기 초당 16.7토큰, 말하기 6.3토큰 (젬마3 1B: 872MB, 21.5초) |
| 채점 (PC) | 한국어 대화 4.61점(5점 만점), 도구를 맞게 부름 99% (불빛 99%, 보기 100%, 움직이기 100%), 부르면 안 될 때 안 부름 94% |

---

## 3. 할 일

### A. 앱 파일 받기 (모델 파일은 받지 않음)

```bash
mkdir -p ~/pibo-take && cd ~/pibo-take
scp circulus@192.168.68.118:openpibo-chat/{pibo_chat.py,tools.py,pi_bench.sh} .
ls -la /home/pi/.model/llm/          # 모델 파일은 사용자가 옮겨 둠: 없으면 멈추고 알림
```

### B. 서버 띄우기 (시험용)

모델 고르기 작업(`llama-server@.service`)이 끝났으면 그걸로, 아직이면 손으로 띄웁니다. 둘 다 동시에 띄우지 않습니다.

```bash
pgrep -af llama-server                                    # 떠 있으면: sudo systemctl stop llama-server 'llama-server@*'
sudo systemctl start llama-server@themaker-ko             # 템플릿 유닛이 있을 때
# 없으면 손으로:
# setsid nohup /home/pi/.llama.cpp/build/bin/llama-server -m /home/pi/.model/llm/themaker-0.8b-Q4_K_M.gguf \
#   --jinja --ctx-size 2048 -np 1 --cache-ram 0 -t 4 -ub 256 --host 0.0.0.0 --port 50020 \
#   > ~/themaker-server.log 2>&1 < /dev/null &
for i in $(seq 1 60); do curl -s http://127.0.0.1:50020/health | grep -q ok && { echo 준비됨; break; }; sleep 2; done
```

### C. 시험

```bash
cd ~/pibo-take
sudo ~/.pyenv/bin/python3 pibo_chat.py --url http://127.0.0.1:50020             # 한국어 모델
# 영어 모델(themaker-en)을 띄웠을 때:
# sudo ~/.pyenv/bin/python3 pibo_chat.py --url http://127.0.0.1:50020 --lang en
```

| 말 | 기대 |
| --- | --- |
| 빨간 불 켜 줘 / 이번엔 파란색으로 / 불 꺼 줘 | 눈 불빛이 빨강 → 파랑 → 꺼짐 |
| 앞에 뭐 있어? | 카메라로 보고 본 것을 말함 |
| 춤춰 줘 / 인사해 줘 / 앞으로 와 줘 | 그 동작 |
| 멈춰 | 모델에 묻지 않고 바로 멈춤 |
| 난 빨간색이 좋아 / 어제 춤 학원 갔어 | 도구 안 부르고 대화만 |
| 우리 반 친구가 인사를 잘 안 해 | 움직이면 안 됨 (화면에 `막음`이 나오면 앱이 막은 것) |
| 너 누구야? / 너 키가 몇이야? | 교육용 AI 더메이커 / 기기는 잘 모른다고 |
| (영어) Turn on the red light / Can you dance? / What do you see? / I like red the most | 영어로 말하고 같은 도구 실행, 마지막은 대화만 |

화면의 `[도구] …` 줄이 실행, `(읽기 N토큰 …)` 줄이 시간, 첫 줄 "(준비 …초)"가 도구 목록을 미리 읽는 시간입니다.

### D. 기존 음성 대화 앱에 붙이기 (설계안만 보고)

- 파이보의 음성 대화(음성 인식 → 대답 → 음성 합성)가 어느 파일·서비스에서 도는지 찾습니다 (확인 필요:
  `tools/run_tools.py`, `tools/lib.py` 등).
- 대답을 만드는 곳에 `pibo_chat.turn()`을 쓰거나 같은 순서로 옮기는 안을 냅니다: 멈춰 확인 → 시스템 글 + 최근 3번
  주고받기 + 도구 목록으로 묻기 → 이동 막기(`blocked`) → 도구 호출이 있으면 `act()`로 실행하고 말하기, 없으면 대답
  그대로. 앱 시작 때 `warm_up` 한 번.
- 영어 모델이면 시스템 글 `tools.SYSTEM_EN`, 앱이 정해 둔 말도 영어(`pibo_chat.SAY["en"]`).
- 블록 코딩의 `call_llm`(글자 대화)에 도구를 붙이는 것은 이번 범위 밖입니다.

---

## 4. 하지 말 것

- 모델 파일(`*.gguf`)을 받거나 옮기거나 지우지 않습니다 (사용자가 따로 옮김).
- `llm-model.gguf` 링크와 기존 서비스 파일을 바꾸지 않습니다. 서비스는 `stop`만 하고 `disable` 하지 않습니다.
- llama-server를 두 개 동시에 띄우지 않습니다 (램 2GB).
- `tools.SYSTEM`·`SYSTEM_EN`·`TOOLS`를 고치지 않습니다 (모델과 글자까지 같아야 함).
- 이동 막기(`blocked`)와 "멈춰" 처리를 빼지 않습니다.
- 파이보에서 빌드할 일이 생기면 `-j2`로 합니다 (`-j4` 금지).
- 모델이나 데이터를 기기에서 다시 만들지 않습니다. 학습·변환은 PC 세션(openpibo-chat)이 합니다.

---

## 5. 보고할 것

1. C의 화면 전체 (시간 줄 포함), 잘못 부른 도구나 안 부른 경우
2. `free -m` (서버만 / 앱까지 띄운 상태), 온도 (`vcgencmd measure_temp`)
3. D: 기존 음성 대화 앱의 위치와, 붙이려면 바꿀 파일과 줄
4. 확인 필요로 남은 것

## 6. 아직 확인 필요

- `tools.ACTIONS`의 동작 이름(greeting, dance1~5, wave1, clapping1, hifive, yes_h, no_h, forward1, backward1, left, right)이
  이 파이보의 motion_db에 모두 있는지
- 앞으로·뒤로 동작이 한 번에 얼마나 움직이는지 (책상 위에서 떨어질 위험)
- 영어판: 파이보의 음성 인식·합성이 영어를 지원하는지
