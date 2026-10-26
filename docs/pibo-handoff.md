# 대화 모델(themaker) 적용 지시서 (파이보·파이브레인 세션용)

이 문서를 받은 세션이 할 일: 기기에 새 대화 모델(themaker, Qwen3.5-0.8B 파인튜닝, Q4_K_M)을 띄우고,
기기 앱이 이 모델의 **도구 호출(불빛, 보기, 움직이기)**을 그 기기의 openpibo 함수로 실행하게 붙입니다.
**서비스 설정 변경과 기존 파일 덮어쓰기는 사용자 확인 뒤에만** 합니다.

대답과 보고는 한국어 존댓말, 결론 먼저. 부품명·라이브러리·핀 이름은 영문 그대로.
전압·전류·핀맵 등 하드웨어 사실은 추측하지 말고, 확인 안 되면 "확인 필요"라고 적습니다.

---

## 1. 모델과 앱의 약속

모델은 모든 기기 공통 하나입니다. 앱은 매번 아래 두 가지를 학습 때와 **글자까지 똑같이** 보냅니다.

| 보내는 것 | 값 |
| --- | --- |
| 시스템 글 | `tools.system_for(기기 이름)` (파이보: `tools.SYSTEM`, 파이브레인: `system_for("파이브레인")`) |
| 도구 목록 | `tools.TOOLS` (모든 기기 공통, 아래 3개). 바꾸면 모델을 다시 학습해야 함 |

| 도구 | 인자 | 파이보 (`pibo_chat.py`) | 파이브레인 (`pibrain_chat.py`) |
| --- | --- | --- | --- |
| `light` | `color`: 빨강·주황·노랑·초록·파랑·하늘·보라·분홍·하양·끄기 | `Device().eye_on(r,g,b)` / `eye_off()` | `DeviceByPiBrain().led_on(r,g,b)` / `led_off()` (2초 넘으면 실패) |
| `look` | 없음 | `Camera().read()` + `Detect().detect_object(img)` | 같음 |
| `move` | `action`: 인사·춤·손흔들기·박수·하이파이브·끄덕이기·도리도리·앞으로·뒤로·왼쪽·오른쪽 | `Motion().set_motion(tools.ACTIONS[action] 중 하나)` | 실행하지 않고 "저는 몸이 없어서 움직일 수는 없어요…"라고 말함 |

- 색 이름 → RGB는 `tools.COLORS`, 동작 이름 → motion_db 이름은 `tools.ACTIONS`.
- 모델이 도구를 부를 때는 호출 앞에 짧은 말("빨간색으로 바꿀게요!")을 함께 냅니다. 앱은 그 말을 하고
  실행합니다. `look`은 본 것을 덧붙입니다 ("한번 볼게요. 앞에 컵이 보여요.").
- 대화 기억(history)에는 학생 말과 실제로 한 말만 남깁니다 (도구 호출·결과 메시지는 넣지 않음).
  학습 데이터의 앞 대화가 그 모양입니다.
- "멈춰"는 안전을 위해 모델에 묻지 않고 앱이 바로 멈춥니다.
- 시스템 글 + 도구 목록은 약 530토큰(파이보에서 읽는 데 약 30초)입니다. 앱을 켤 때 `warm_up`으로
  한 번 읽혀 두면, 매번 같은 앞부분이라 서버가 기억해 대화할 때는 새로 한 말만 읽습니다.
- `pibo_chat.py`/`pibrain_chat.py`는 이 약속대로 만든 참고용 앱입니다 (키보드 입력, `--fake` 시험).
  기기 앱에 붙일 때 `turn()`, `act()`, 기기 클래스(`Pibo`, `PiBrain`)를 그대로 쓰거나 옮겨 씁니다.

---

## 2. 지금 확인된 사실

| 항목 | 값 |
| --- | --- |
| 파이보 | Raspberry Pi 4B, 램 2GB, IP `192.168.68.121`, 사용자 `pi` |
| 학습 PC | Ubuntu, IP `192.168.68.118`, 사용자 `circulus`, 저장소 `~/openpibo-chat` |
| 새 llama.cpp | `/home/pi/llama.cpp-new` (빌드 11889, 커밋 `d854d98`). 기존 `/home/pi/.llama.cpp`는 Qwen3.5를 못 읽음 |
| 기존 서버 | 서비스가 `/home/pi/.llama.cpp/build/bin/llama-server -m /home/pi/.model/llm/llm-model.gguf --ctx-size 2048 --log-disable --host 0.0.0.0 --port 50020`로 젬마3 1B를 띄움 |
| 새 모델 | `themaker-0.8b-Q4_K_M.gguf` (PC `out/`, 약 320MB) |
| 파이썬 | `sudo ~/.pyenv/bin/python3` (openpibo가 여기에 있고, Device·카메라·NeoPixel은 root 필요) |
| 기기 웹 대화 | `openpibo-os.{pibo,pibrain}/chat/run_chat.py`의 `SYSTEM`이 예전 문구. 새 모델에서는 `system_for(기기 이름)`과 글자까지 같게 바꿔야 함 (파이브레인은 "로봇 파이브레인이야.") |

파이보 실측 (예전 같은 크기 모델): 최대 515MB, 대화 한 번 11.5초 (젬마3 1B: 872MB, 21.5초).
읽기 초당 16.7토큰, 말하기 6.3토큰.

알려진 함정:
- llama-server를 셸에서 `&`로만 띄우면 그 창의 신호에 같이 꺼질 수 있어 `setsid nohup`으로 띄웁니다.
- llama-server 웹 화면은 도구 목록과 시스템 글을 보내지 않습니다. 새 GGUF 템플릿은 시스템 글이
  없으면 학습 때의 것(이름 더메이커)을 넣지만, 도구는 웹 화면에서 안 됩니다.

---

## 3. 하지 말 것

- `/home/pi/.model/llm/llm-model.gguf`(젬마)와 기존 서비스 파일을 백업 없이 덮어쓰지 않습니다.
  서비스는 `stop`만 하고 `disable` 하지 않습니다.
- llama-server를 두 개 동시에 띄우지 않습니다 (램 2GB).
- 파이보에서 `-j4`로 빌드하지 않습니다 (`-j2`).
- 모델이나 데이터를 기기에서 다시 만들지 않습니다. 학습·변환은 PC 세션(openpibo-chat)이 합니다.

---

## 4. 할 일

### A. 파일 받기 (기기에서)

```bash
mkdir -p ~/pibo-take && cd ~/pibo-take
scp circulus@192.168.68.118:openpibo-chat/{pibo_chat.py,pibrain_chat.py,tools.py,pi_bench.sh} .
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
```

`--jinja`가 있어야 도구 호출을 읽습니다. `-np 1 --cache-ram 0`은 램 2GB에 맞게.

### C. 시험

```bash
cd ~/pibo-take
sudo ~/.pyenv/bin/python3 pibo_chat.py --url http://127.0.0.1:50020      # 파이보
sudo ~/.pyenv/bin/python3 pibrain_chat.py --url http://127.0.0.1:50020   # 파이브레인
```

시험할 말: "빨간 불 켜 줘", "이번엔 파란색으로", "불 꺼 줘", "앞에 뭐 있어?", "춤춰 줘"(파이브레인은
못 움직인다고 해야 함), "멈춰", "난 빨간색이 좋아"(불이 켜지면 안 됨), "어제 춤 학원 갔어"(움직이면 안 됨).
화면의 `[도구] …` 줄이 실행, `(읽기 N토큰 …)` 줄이 시간입니다. 첫 줄 "(준비 …초)"가 도구 목록을
미리 읽는 시간입니다.

### D. 기기 앱에 붙이기

기존 기기 앱(음성 인식 → 대답 → 음성 합성)에서 대답을 만드는 곳에 `pibo_chat.turn()`을 쓰거나
같은 순서로 옮깁니다: 멈춰 확인 → 시스템 글 + 최근 3번 주고받기 + 도구 목록으로 묻기 →
도구 호출이 있으면 `act()`로 실행하고 말하기, 없으면 대답 그대로. 앱 시작 때 `warm_up` 한 번.

### E. 서비스 전환 (사용자가 하라고 할 때만)

서비스 파일과 젬마 모델을 백업한 뒤 `ExecStart`만 바꿉니다.

```
ExecStart=/home/pi/llama.cpp-new/build/bin/llama-server -m /home/pi/.model/llm/llm-model.gguf --jinja --ctx-size 2048 -np 1 --cache-ram 0 -t 4 -ub 256 --log-disable --host 0.0.0.0 --port 50020
```

---

## 5. 보고할 것

1. C의 화면 전체 (시간 줄 포함), 잘못 부른 도구나 안 부른 경우
2. `free -m` (서버만 / 앱까지 띄운 상태), 온도
3. D를 적용하려면 바꿀 파일과 줄
4. 확인 필요로 남은 것

## 6. 아직 확인 필요

- `tools.ACTIONS`의 동작 이름(greeting, dance1~5, wave1, clapping1, hifive, yes_h, no_h, forward1,
  backward1, left, right)이 이 파이보의 motion_db에 모두 있는지
- 파이브레인 NeoPixel을 앱에서 켤 때 Tools(50000)나 검수 서버와 겹치지 않는지
- 파이브레인은 마이크가 없어 음성 인식이 안 됨: 글자 입력 화면(`chat/run_chat.py`)에 도구를 붙일지
