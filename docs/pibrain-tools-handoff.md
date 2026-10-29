# 파이브레인 도구 지시서 (파이브레인 세션용, 최종)

이 문서를 받은 세션이 할 일: 대화 모델 themaker(한국어 `themaker-ko`, 영어 `themaker-en`)가 학생 말을 보고 부르는
**도구 3개(불빛·보기·움직이기)**를 파이브레인의 openpibo 함수로 실행하게 붙입니다. 파이브레인은 몸(서보)이 없어서
**움직이기는 실행하지 않고 "움직일 수 없다"고 말합니다**.

- 먼저 참고용 앱 `pibrain_chat.py`로 파이브레인에서 시험합니다. 그다음 붙일 곳(글자 대화 웹 등)에 대한 **설계안을 보고**합니다.
- 서비스 설정 변경과 기존 파일 덮어쓰기는 **사용자 확인 뒤에만** 합니다.
- 모델 서버를 고르고 띄우는 것은 `docs/model-select-handoff.md`(모델 고르기 블록)의 일입니다. 이 문서는 도구만 다룹니다.

대답과 보고는 한국어 존댓말, 결론 먼저. 부품명·라이브러리·핀 이름은 영문 그대로. 전압·전류·핀맵 등 하드웨어 사실은
추측하지 말고, 확인 안 되면 "확인 필요"라고 적습니다.

---

## 1. 모델과 앱의 약속

모델은 파이보와 **같은 모델, 같은 시스템 글, 같은 도구 목록**입니다. 다른 것은 기기 함수뿐입니다.
앱은 매번 아래 두 가지를 학습 때와 **글자까지 똑같이** 보냅니다. 바꾸면 모델을 다시 학습해야 합니다.

| 보내는 것 | 한국어 모델 (`themaker-ko`) | 영어 모델 (`themaker-en`) |
| --- | --- | --- |
| 시스템 글 | `tools.SYSTEM` | `tools.SYSTEM_EN` |
| 도구 목록 | `tools.TOOLS` (움직이기도 그대로 보냄) | `tools.TOOLS` |
| 참고 앱 | `pibrain_chat.py` | `pibrain_chat.py --lang en` |

모델은 "서큘러스에서 만든 학생들을 위한 교육용 AI 더메이커(TheMaker)"로 이야기하고, 기기 이름이나 사양은 말하지
않습니다 (물으면 잘 모른다고 함).

| 도구 | 인자 | 파이브레인 함수 |
| --- | --- | --- |
| `light` | `color`: 빨강·주황·노랑·초록·파랑·하늘·보라·분홍·하양·끄기 | `DeviceByPiBrain().led_on(r, g, b)` / `led_off()` (`openpibo/device.py` 467행, NeoPixel). show()가 가끔 돌아오지 않아 **2초 넘으면 실패**로 봄. 색 → RGB는 `tools.COLORS` |
| `look` | 없음 | `Camera().read()` + `Detect().detect_object(img)` (파이보와 같음) |
| `move` | `action` | **실행하지 않음**. 앱이 "저는 몸이 없어서 움직일 수는 없어요. 대신 불빛을 바꿔 볼까요?" (영어: "I don't have a body, so I can't move. Want me to change the light instead?")라고 말함. 모델이 낸 "춤출게요!"는 버림 |

**앱이 지킬 것** (`pibrain_chat.py`·`pibo_chat.py`에 이미 그렇게 되어 있음)
- 모델이 도구를 부를 때는 호출 앞에 짧은 말("빨간색으로 바꿀게요!")을 함께 냅니다. 앱은 그 말을 하고 실행합니다.
  `look`은 본 것을 덧붙입니다. 대답 끝의 빈 줄은 `strip()`으로 지웁니다.
- 대화 기억(history)에는 학생 말과 실제로 한 말만 남깁니다 (도구 호출·결과 메시지는 넣지 않음). 최근 3번 주고받기.
- "멈춰"·"stop"은 모델에 묻지 않고 앱이 바로 대답합니다 (파이브레인은 멈출 동작이 없음).
- 이야기 속 말에 잘못 부른 이동(앞으로·뒤로·왼쪽·오른쪽)은 앱이 막고 대화로 대답합니다 (`blocked()`). 파이브레인에서는
  "못 움직여요"라는 엉뚱한 말 대신 대화로 대답하게 하는 장치입니다.
- 도구 이름·인자가 목록에 없으면 실행하지 않습니다 (`check_call()`).
- 시스템 글 + 도구 목록은 약 530토큰입니다. 앱을 켤 때 `warm_up`으로 한 번 읽혀 둡니다.

---

## 2. 지금 확인된 사실

| 항목 | 값 |
| --- | --- |
| 파이브레인 | 서보·배터리·마이크 없음 (그래서 움직이기 없음, 음성 인식 없음 → 글자 입력). NeoPixel LED 1개(GPIO12, root 필요). IP는 확인 필요 |
| 학습 PC | Ubuntu, IP `192.168.68.118`, 사용자 `circulus`, 저장소 `~/openpibo-chat` |
| llama.cpp | `/home/pi/.llama.cpp` (새 빌드). `--jinja`가 있어야 도구 호출을 읽음. 파이브레인 기기의 서비스·빌드 상태는 실기기로 아직 안 봄 (확인 필요) |
| 모델 파일 | 사용자가 `/home/pi/.model/llm/`에 따로 옮겨 둠. `themaker-0.8b-Q4_K_M.gguf` sha256 `2a8799ec40d2f9a3facc035d7ab7eabc0fcc61b5e5637d7ddbb9ea453b21970c`. 영어판은 PC 학습 뒤 |
| 파이썬 | `sudo ~/.pyenv/bin/python3` (openpibo가 여기 있다고 가정, 확인 필요) |
| 글자 대화 웹 | `chat/run_chat.py` (50030). 지금은 도구 없이 글자 대화만 |
| 채점 (PC) | 한국어 대화 4.61점(5점 만점), 도구를 맞게 부름 99%, 부르면 안 될 때 안 부름 94% |

---

## 3. 할 일

### A. 앱 파일 받기 (모델 파일은 받지 않음)

```bash
mkdir -p ~/pibo-take && cd ~/pibo-take
scp circulus@192.168.68.118:openpibo-chat/{pibo_chat.py,pibrain_chat.py,tools.py,pi_bench.sh} .
ls -la /home/pi/.model/llm/          # 모델 파일은 사용자가 옮겨 둠: 없으면 멈추고 알림
```

`pibrain_chat.py`는 `pibo_chat.py`와 `tools.py`가 같은 폴더에 있어야 합니다.

### B. 서버 띄우기 (시험용)

```bash
pgrep -af llama-server                                    # 떠 있으면: sudo systemctl stop llama-server 'llama-server@*'
sudo systemctl start llama-server@themaker-ko             # 모델 고르기 작업의 템플릿 유닛이 있을 때
# 없으면 손으로:
# setsid nohup /home/pi/.llama.cpp/build/bin/llama-server -m /home/pi/.model/llm/themaker-0.8b-Q4_K_M.gguf \
#   --jinja --ctx-size 2048 -np 1 --cache-ram 0 -t 4 -ub 256 --host 0.0.0.0 --port 50020 \
#   > ~/themaker-server.log 2>&1 < /dev/null &
for i in $(seq 1 60); do curl -s http://127.0.0.1:50020/health | grep -q ok && { echo 준비됨; break; }; sleep 2; done
```

시험 전에 LED를 함께 쓰는 Tools(50000)·분류기·H/W 검수가 떠 있으면 끕니다 (`sudo systemctl stop tools.service classify.service`).

### C. 시험 (키보드로 글자 입력)

```bash
cd ~/pibo-take
sudo ~/.pyenv/bin/python3 pibrain_chat.py --url http://127.0.0.1:50020           # 한국어 모델
# 영어 모델(themaker-en)을 띄웠을 때:
# sudo ~/.pyenv/bin/python3 pibrain_chat.py --url http://127.0.0.1:50020 --lang en
```

| 말 | 기대 |
| --- | --- |
| 빨간 불 켜 줘 / 이번엔 파란색으로 / 불 꺼 줘 | LED가 빨강 → 파랑 → 꺼짐 (2초 안에) |
| 앞에 뭐 있어? | 카메라로 보고 본 것을 말함 |
| 춤춰 줘 / 앞으로 와 줘 | 움직이지 않고 "저는 몸이 없어서 움직일 수는 없어요…" |
| 난 빨간색이 좋아 / 어제 춤 학원 갔어 | 도구 안 부르고 대화만 |
| 우리 반 친구가 인사를 잘 안 해 | "못 움직여요"가 아니라 대화로 대답 (화면에 `막음`이 나오면 앱이 막은 것) |
| 너 누구야? / 너 키가 몇이야? | 교육용 AI 더메이커 / 기기는 잘 모른다고 |
| (영어) Turn on the red light / Can you dance? / What do you see? | 영어로 말하고 불빛·보기 실행, 춤은 "I don't have a body…" |

### D. 붙일 곳 (설계안만 보고)

파이브레인은 마이크가 없어 음성 대화 앱이 없습니다. 도구를 쓸 곳으로 다음 중 하나를 골라 설계안을 냅니다.
1. 글자 대화 웹(`chat/run_chat.py`, 50030)에 도구를 붙이기: 대답을 만드는 곳에 `pibo_chat.turn()`과 `PiBrain` 기기
   클래스를 씀. LED·카메라는 root가 필요하니, 대화 웹이 root로 도는지 확인 필요
2. 블록 코딩용 예제(참고 앱을 IDE에서 실행)로만 두기

어느 쪽이든 IDE의 Tools(50000)·H/W 검수가 같은 LED를 쓰므로 동시에 켜지지 않게 하는 방법을 함께 적습니다.

---

## 4. 하지 말 것

- 모델 파일(`*.gguf`)을 받거나 옮기거나 지우지 않습니다 (사용자가 따로 옮김).
- `llm-model.gguf` 링크와 기존 서비스 파일을 바꾸지 않습니다. 서비스는 `stop`만 하고 `disable` 하지 않습니다.
- llama-server를 두 개 동시에 띄우지 않습니다 (램 2GB).
- `tools.SYSTEM`·`SYSTEM_EN`·`TOOLS`를 고치지 않습니다. 움직이기 도구도 목록에서 빼지 않습니다
  (모델이 그 목록으로 배움. 못 움직이는 것은 앱이 처리).
- 모델이나 데이터를 기기에서 다시 만들지 않습니다.

---

## 5. 보고할 것

1. C의 화면 전체 (시간 줄 포함), 잘못 부른 도구나 안 부른 경우, LED 2초 넘김이 있었는지
2. `free -m` (서버만 / 앱까지 띄운 상태), 온도
3. D: 고른 안과 바꿀 파일·줄
4. 확인 필요로 남은 것 (IP, 파이썬 경로, 서비스·빌드 상태, LED를 함께 쓰는 서비스와의 충돌)
