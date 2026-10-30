# 대화 모델 고르기 블록 만들기 지시서 (파이보·파이브레인 세션용, 최종)

이 문서를 받은 세션이 할 일: 블록 코딩에서 대화 모델을 **gemma3 / themaker-ko / themaker-en** 중에서 골라
띄우는 **"대화 모델 고르기" 블록**을 만듭니다. 설계는 아래로 정해졌습니다. 그대로 만듭니다.

- **저장소 안의 작업**(블록, `speech.py`, IDE, 대화 웹, 문서)은 바로 합니다.
- **기기 이미지 작업**(서비스 파일, 모델 파일, 링크)은 **백업한 뒤 사용자에게 확인을 받고** 합니다.
- 파이보(`openpibo-os.pibo`)와 파이브레인(`openpibo-os.pibrain`) 둘 다 같은 방식으로 합니다.

대답과 보고는 한국어 존댓말, 결론 먼저. 부품명·라이브러리·핀 이름은 영문 그대로. 확인 안 된 사실은
"확인 필요"라고 적습니다.

---

## 1. 정해진 것

| 항목 | 정한 것 |
| --- | --- |
| 모델 | `gemma3`(젬마3 1B), `themaker-ko`(더메이커 한국어), `themaker-en`(더메이커 영어). 램 2GB라 **한 번에 하나만** |
| 예전 블록 | "대화 서버 시작하기(LLM)"(`speech_start_llm` → `dialog.start_llm()`)는 **바꾸지 않음**. 예전 학생 코드가 그대로 돌아야 함 |
| 새 블록 | "대화 모델 고르기 [젬마3 ▼ / 더메이커 한국어 / 더메이커 영어]" → `dialog.set_llm_model("themaker-ko")`. 이 블록 뒤의 예전 시작 블록이 고른 모델로 띄움 |
| 고른 모델의 기억 | 그 프로그램의 `Dialog` 객체 안에서만. 파일·링크·서비스 기본값은 바꾸지 않음 |
| 띄우는 방법 | systemd 템플릿 유닛 `llama-server@.service`. `systemctl start llama-server@themaker-ko`이면 `%i`가 `themaker-ko`가 되어 `/home/pi/.model/llm/themaker-ko.gguf`를 읽음 |
| 기본 모델 | `llama-server.service` + `llm-model.gguf` 링크는 그대로. 모델 고르기 블록이 없는 프로그램과 IDE [대화]는 지금 기본 모델로 뜸 |
| 링크 | 블록 실행 때 **링크를 바꾸지 않음**. 모델별 별칭 링크 3개는 이미지 만들 때 한 번만 만듦 |
| llama.cpp | `/home/pi/.llama.cpp` (그 자리에서 다시 빌드한 새 빌드, 사용자 확인). 예전 기록의 `/home/pi/llama.cpp-new`는 쓰지 않음 |

```
[대화 모델 고르기 (더메이커 한국어 ▼)]      ← 새 블록: 고르기만 함
[대화 서버 시작하기(LLM)]                   ← 예전 블록 그대로
[대화하기 "안녕" …]
```

| 모델 이름 | 파일 (`/home/pi/.model/llm/`) | 말 | 시스템 글 |
| --- | --- | --- | --- |
| `gemma3` | `gemma3.gguf` → `gemma-3-1b-it-Q4_K_M.gguf` (지금 있음, 약 806MB) | 한국어·영어 | 블록이 준 것 |
| `themaker-ko` | `themaker-ko.gguf` → `themaker-0.8b-Q4_K_M.gguf` (321MB) | 한국어 | 없으면 GGUF 템플릿이 학습 때의 글을 넣음 |
| `themaker-en` | `themaker-en.gguf` → `themaker-en-0.8b-Q4_K_M.gguf` (331MB, sha256 `39bf09302eb9…`) | 영어 | 없으면 GGUF 템플릿이 학습 때의 글을 넣음 |

모델 파일은 **사용자가 따로 옮겨 둡니다** (`/home/pi/.model/llm/`). 이 세션은 모델 파일을 받거나 옮기지 않고,
있는지만 확인합니다. `themaker-en`은 PC 학습이 끝나야 생깁니다. 파일이 아직 없으면 그 모델만 "준비 안 됨" 오류를 내게 하고 나머지는 먼저 만듭니다.

---

## 2. 먼저 기기에서 확인 (고치기 전에)

```bash
ls -la /home/pi/.model/llm/
systemctl cat llama-server.service
systemctl is-active llama-server; pgrep -af llama-server
ls /etc/systemd/system/ | grep -i llama
/home/pi/.llama.cpp/build/bin/llama-server --version     # 다시 빌드한 버전 기록 (11889 이상)
ls -d /home/pi/llama.cpp-new 2>/dev/null                 # 남아 있으면 보고만 (지우지 않음)
ps -o user= -p $(pgrep -f run_ide.py | head -1)          # IDE·학생 코드가 root로 도는지
free -m
```

- 블록으로 실행한 학생 코드가 root로 도는지 확인합니다 (`systemctl` 실행에 필요). 지금 `start_llm()`도
  `systemctl`을 쓰므로 같은 조건이지만, 확인 필요입니다.

---

## 3. 저장소 안의 작업 (바로 함)

### 3-1. `openpibo/speech.py` — `Dialog`에 더하기만

기존 `start_llm()`·`call_llm()`·`stop_llm()`의 **모델을 고르지 않았을 때 동작은 지금과 글자까지 같게** 둡니다.

| 함수 | 동작 |
| --- | --- |
| `set_llm_model(self, model)` (새로 만듦) | `model`은 `"gemma3"`, `"themaker-ko"`, `"themaker-en"` 중 하나. 그 밖이면 `ValueError`("모델 이름은 gemma3, themaker-ko, themaker-en 중 하나"). `/home/pi/.model/llm/{model}.gguf`가 없으면 `FileNotFoundError`("{model} 모델 파일이 아직 없어요"). 고른 이름을 `self`에 기억. 이 `Dialog`가 이미 서버를 띄운 뒤라면 고른 모델로 다시 띄움 |
| `start_llm(self, port=50020)` | 고른 모델이 없으면 **지금과 똑같이** `systemctl start llama-server`. 골랐으면 `systemctl stop llama-server 'llama-server@*'`로 다 끄고 `systemctl start llama-server@{model}`을 실행한 뒤 `http://127.0.0.1:50020/health`가 `ok`가 될 때까지 최대 120초 기다림 (로딩 중엔 503) |
| `stop_llm(self)` | `systemctl stop llama-server 'llama-server@*'` (어느 모델이든 끔) |
| `call_llm(...)` | 바꾸지 않음. `system_prompt`를 안 주면 시스템 글 없이 보내고, themaker는 GGUF 템플릿이 학습 때의 글을 넣음 |

- `__init__`에서 고른 모델은 `None`입니다.
- 파이보·파이브레인의 `speech.py`는 이 부분이 같습니다. 두 저장소에 같게 넣습니다.

### 3-2. 블록 (`ide/static/`)

| 파일 | 할 일 |
| --- | --- |
| `customblock.js` | 새 블록 `speech_set_llm_model`을 만듭니다. 모양은 `speech_start_llm`(1387행 근처)과 같고, `field_dropdown` 하나에 `gemma3`·`themaker-ko`·`themaker-en`을 둡니다. 색은 `color_type["speech"]` |
| `customblock_callback.js` | `Blockly.Python.forBlock['speech_set_llm_model']` → `dialog.set_llm_model('themaker-ko')\n`. `dialog` 준비는 `speech_start_llm`(427행 근처)과 같은 방식으로 합니다 |
| `customblock_toolbox.js` | 대화(speech) 상자에서 "대화 서버 시작하기(LLM)" **바로 앞**에 둡니다 |
| `ko.js` | `SPEECH_SET_LLM_MODEL` = "%1 대화 모델 고르기 %2", 툴팁 "대화 서버 시작하기 전에 쓸 모델을 고릅니다. 안 고르면 기본 모델로 시작합니다.", 드롭다운 "젬마3" / "더메이커 한국어" / "더메이커 영어" |
| `en.js` | "%1 Choose chat model %2", 툴팁 "Choose the model before starting the chat server. Without it, the default model starts.", "Gemma3" / "TheMaker Korean" / "TheMaker English" |

### 3-3. IDE가 끄는 곳 (`ide/run_ide.py`)

지금은 `llama-server.service`만 끕니다. 아래를 모두 `llama-server.service`와 `llama-server@*`를 함께 끄게 고칩니다.
빠뜨리면 램 2GB에 모델 두 개가 올라갈 수 있습니다.

- `/tools`, `/classifier`, `/hwtest` 켜기
- `/llm?enable=off`
- [정지], 코드 실행 시작
- `systemctl stop`을 부르는 다른 곳: `grep -n "llama-server" ide/run_ide.py`로 모두 찾습니다

`/llm?enable=on`(IDE [대화])을 할 때는 `llama-server@*`를 먼저 끄고 기본 `llama-server.service`를 켭니다.

### 3-4. 대화 웹 (`chat/run_chat.py`)

- IDE [대화]는 기본 모델(`llm-model.gguf`)로 뜹니다.
- `SYSTEM`(40행)을 링크가 가리키는 파일에 맞춰 고릅니다 (`os.path.realpath`로 파일 이름을 봄).
  - 이름에 `themaker-en`이 있으면 영어 모델의 시스템 글
  - `themaker`가 있으면 한국어 모델의 시스템 글
  - 그 밖(젬마 등)은 지금 글 그대로
- themaker의 시스템 글은 학습 저장소(openpibo-chat)의 `tools.SYSTEM` / `tools.SYSTEM_EN`과 **글자까지 같게** 넣습니다
  (PC `~/openpibo-chat/tools.py`에서 복사).

### 3-5. 문서

- `docs/source/blocks/guide.md`, `docs/source_en/blocks/guide.md`에 새 블록을 설명합니다.
  - 모델 셋의 차이: 젬마3는 한국어·영어, 더메이커 한국어는 한국어 전용·짧은 존댓말, 더메이커 영어는 영어 전용
  - 더메이커는 시스템 글 없이 쓰는 것이 가장 자연스럽다는 것
  - 고르기 블록은 시작 블록 앞에 놓는다는 것
- `CLAUDE.md`의 '대화 웹'·서비스 절에 템플릿 유닛과 끄는 곳 목록을 적습니다.

---

## 4. 기기 이미지 작업 (백업 → 사용자 확인 → 함)

```bash
# 백업
sudo cp /etc/systemd/system/llama-server.service ~/llama-server.service.bak
ls -la /home/pi/.model/llm/ > ~/model-llm-ls.bak

# 모델 파일은 사용자가 따로 옮겨 둠: 받거나 옮기지 말고 있는지만 확인
ls -la /home/pi/.model/llm/
sha256sum /home/pi/.model/llm/themaker-0.8b-Q4_K_M.gguf   # 2a8799ec40d2f9a3facc035d7ab7eabc0fcc61b5e5637d7ddbb9ea453b21970c 이어야 함
# 없으면 멈추고 사용자에게 알림 (themaker-en은 아직 없어도 됨)

# 모델별 별칭 링크 (한 번만)
cd /home/pi/.model/llm
sudo ln -s gemma-3-1b-it-Q4_K_M.gguf gemma3.gguf
sudo ln -s themaker-0.8b-Q4_K_M.gguf themaker-ko.gguf
sudo ln -s themaker-en-0.8b-Q4_K_M.gguf themaker-en.gguf      # 파일이 오기 전엔 깨진 링크 (set_llm_model이 파일 없음으로 알림)
sha256sum /home/pi/.model/llm/themaker-en-0.8b-Q4_K_M.gguf     # 39bf09302eb91777f7b4e9909feafc41ce13e1b183a5e7efff5afaceabca1570
```

`/etc/systemd/system/llama-server@.service`는 `systemctl cat llama-server`의 내용을 그대로 복사해 만듭니다.
`-m` 경로만 `/home/pi/.model/llm/%i.gguf`로 바꾸고, `Description`과 `Conflicts`를 더합니다.

```ini
[Unit]
Description=llama-server (%i)
After=network.target
Conflicts=llama-server.service

[Service]
# 아래 두 줄은 지금 llama-server.service와 같게 (User 등 다른 줄도 그대로)
WorkingDirectory=/home/pi/.llama.cpp/build/bin
ExecStart=/home/pi/.llama.cpp/build/bin/llama-server -m /home/pi/.model/llm/%i.gguf <지금 서비스의 나머지 옵션 그대로>
Restart=no
```

```bash
sudo systemctl daemon-reload
```

- **옵션**: themaker는 `--jinja`가 있어야 GGUF 템플릿(생각 끄기, 기본 시스템 글)을 씁니다. 지금 서비스에 `--jinja`가 없으면 템플릿
  유닛에는 넣고, 지금 서비스(`llama-server.service`)는 건드리지 않습니다. 램 2GB에 맞춘 옵션은
  `--ctx-size 2048 -np 1 --cache-ram 0 -t 4 -ub 256`입니다.
- **Conflicts**: `Conflicts=llama-server.service`로 기본 서버와는 서로 끕니다. 템플릿끼리(`@themaker-ko`와 `@themaker-en`)는
  서로 끄지 않으므로, `start_llm()`이 켜기 전에 반드시 다 끕니다 (3-1).
- **`enable` 안 함**: 템플릿 유닛은 `enable` 하지 않습니다 (부팅 때 안 뜸).

---

## 5. 시험

1. **예전 프로그램**: 고르기 블록 없이 [대화 서버 시작하기] → [대화하기]. 지금 기본 모델로 대답해야 합니다.
2. **모델 셋**: 각각 [대화 모델 고르기] → [대화 서버 시작하기] → [대화하기 "너는 누구야?"].
   - `systemctl list-units 'llama-server*'`에 하나만 active인지 봅니다.
   - `pgrep -c llama-server`가 1인지 봅니다.
   - 더메이커 한국어는 "교육용 AI 더메이커예요", 영어는 "I'm TheMaker"로 대답해야 합니다.
3. **바꾸기**: 한 프로그램에서 themaker-ko로 시작한 뒤 [대화 모델 고르기 (젬마3)]. 젬마로 다시 떠야 합니다 (서버 하나만).
4. **끄기**: [대화 서버 끄기], IDE [도구]·[분류기]·[H/W 검수]·[정지]를 누른 뒤 `pgrep -c llama-server`가 0이어야 합니다.
5. **IDE [대화]**: 모델을 고른 프로그램을 돌린 뒤 IDE [대화]를 열어도 기본 모델로 떠야 합니다.
6. **파일 없음**: themaker-en 파일이 없을 때 고르면 알아볼 수 있는 오류가 나야 합니다.
7. **젬마3 + `--jinja`**: 템플릿 유닛으로 띄운 젬마3가 정상으로 대답하는지 확인합니다.
8. 모델마다 `free -m`과 첫 대답까지 걸린 시간을 잽니다.

---

## 6. 하지 말 것

- 예전 블록(`speech_start_llm`)과 `start_llm()`·`call_llm()`의 **모델을 고르지 않았을 때 동작**을 바꾸지 않습니다.
- 블록 실행 때 `llm-model.gguf` 링크를 바꾸지 않습니다.
- llama-server를 두 개 동시에 띄우지 않습니다 (램 2GB).
- 서비스 파일과 젬마 모델 파일을 백업 없이 지우거나 덮어쓰지 않습니다. `llama-server.service`는 `disable` 하지 않습니다.
- 4장(이미지 작업)은 사용자 확인 전에 하지 않습니다.
- 모델 파일(`*.gguf`)을 받거나 옮기거나 지우지 않습니다. 사용자가 따로 옮깁니다.
- 파이보에서 빌드할 일이 생기면 `-j2`로 합니다 (`-j4` 금지).

---

## 7. 보고할 것

1. 2장 확인 결과 (링크, 서비스 파일, 버전, root 여부, `free -m`)
2. 바꾼 파일과 줄 (저장소 안 / 이미지 작업을 나눠서, 파이보·파이브레인 각각)
3. 5장 시험 결과 (모델마다 램·시간, 실패한 것)
4. 확인 필요로 남은 것
