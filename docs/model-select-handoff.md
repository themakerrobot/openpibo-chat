# 대화 모델 고르기 설계 지시서 (파이보·파이브레인 세션용)

이 문서를 받은 세션이 할 일: 기기의 대화 모델을 **gemma3 / themaker-ko / themaker-en** 셋 중에서 골라
띄우는 방법을 **조사하고 설계안을 내는 것**입니다. 지금 단계에서는 **기기와 저장소를 고치지 않습니다**.
설계안을 사용자가 고르면 그때 고칩니다.

대답과 보고는 한국어 존댓말, 결론 먼저. 부품명·라이브러리·핀 이름은 영문 그대로. 확인 안 된 사실은
"확인 필요"라고 적습니다.

---

## 1. 하려는 것

- 모델 셋을 기기에 같이 두고, 블록 코딩에서 **고른 하나만** 띄웁니다 (램 2GB라 동시에 하나).

  | 이름 | 파일 | 말 | 도구 | 시스템 글 |
  | --- | --- | --- | --- | --- |
  | gemma3 | `gemma-3-1b-it-Q4_K_M.gguf` (지금 기기에 있음, 약 806MB) | 한국어·영어 | 없음 | 블록이 주는 것 |
  | themaker-ko | `themaker-0.8b-Q4_K_M.gguf` (PC `out/`, 321MB) | 한국어 | 불빛·보기·움직이기 | `tools.SYSTEM` (GGUF 템플릿에도 들어 있음) |
  | themaker-en | `themaker-en-0.8b-Q4_K_M.gguf` (PC `out/`, 학습 중) | 영어 | 같음 | `tools.SYSTEM_EN` (GGUF 템플릿에도 들어 있음) |

- 지금 블록 **"대화 서버 시작하기(LLM)"**(`speech_start_llm` → `dialog.start_llm()`)는 **그대로 둡니다**.
  모델 고르는 칸이 없고, 예전에 만든 학생 코드가 그대로 돌아야 합니다.
- **모델 선택 블록을 하나 더 만듭니다** (사용자가 정함): "대화 모델 고르기 [젬마3 ▼ / 더메이커 한국어 / 더메이커 영어]"
  → `dialog.set_llm_model("themaker-ko")`. 그 뒤의 예전 블록 "대화 서버 시작하기(LLM)"가 고른 모델로 띄웁니다.

  ```
  [대화 모델 고르기 (더메이커 한국어 ▼)]      ← 새 블록: 고르기만 함
  [대화 서버 시작하기(LLM)]                   ← 예전 블록 그대로
  [말하기 (대화하기 "안녕" …)]
  ```

  - 고른 모델은 **그 프로그램(Dialog 객체) 안에서만** 기억합니다. 파일·링크·서비스 기본값을 바꾸지 않으므로,
    모델 선택 블록이 없는 예전 프로그램과 IDE [대화]는 지금 기본 모델 그대로 뜹니다.
  - 모델 선택 블록을 시작 블록 **뒤에** 놓았을 때: 이미 떠 있는 서버를 고른 모델로 다시 띄울지,
    아무것도 안 할지 정해서 설계안에 적어 주세요 (추천: 떠 있으면 고른 모델로 다시 띄움).

---

## 2. 지금 확인된 것 (저장소 기준, 기기에서 다시 확인)

| 항목 | 내용 | 근거 |
| --- | --- | --- |
| 지금 블록 | `speech_start_llm` → `dialog.start_llm()` | `ide/static/customblock.js` 1387행, `ide/static/customblock_callback.js` 427행 |
| `start_llm()` | `os.system("systemctl start llama-server")` 한 줄. 모델은 고르지 않음 | `openpibo/speech.py` 426행 |
| `stop_llm()` | `systemctl stop llama-server` | `openpibo/speech.py` 483행 |
| `call_llm()` | `localhost:50020/v1/chat/completions`, `system_prompt`가 있을 때만 시스템 글을 보냄, `"model": "llm-model.gguf"` | `openpibo/speech.py` 439행 |
| IDE [대화] | `/llm?enable=on` → tools·classify·검수를 끄고 `llama-server.service` 시작 + 대화 웹(`chat/run_chat.py`, 50030) | `ide/run_ide.py` 286행 |
| 대화 웹 시스템 글 | `chat/run_chat.py` 40행 `SYSTEM` (예전 문구 "교육용 로봇 파이보야") | |
| 모델 링크 | `/home/pi/.model/llm/llm-model.gguf` → (파이보 261026 전환 뒤) `pibo-qwen3.5-0.8b-school-small-Q4_K_M.gguf`, 젬마 파일은 남아 있음 | 파이보 `CLAUDE.md` '기기 서비스 전환' |
| 서비스 | `/etc/systemd/system/llama-server.service`, ExecStart는 새 llama.cpp(`/home/pi/llama.cpp-new`), `WorkingDirectory`도 새 빌드 폴더. 유닛 파일은 저장소 밖(이미지 작업) | 파이보 `CLAUDE.md` |
| 파이브레인 | `speech.py`의 `start_llm`/`stop_llm`/`call_llm`은 파이보와 같음. 서비스·모델 전환은 실기기에서 아직 안 봄 | 파이브레인 `CLAUDE.md` 214행 |

**기기에서 먼저 확인할 것 (확인 필요)**

```bash
ls -la /home/pi/.model/llm/
systemctl cat llama-server.service
systemctl is-active llama-server; pgrep -af llama-server
ls /etc/systemd/system/ | grep -i llama
ps -o user= -p $(pgrep -f run_ide.py | head -1)        # IDE·학생 코드가 root로 도는지
/home/pi/llama.cpp-new/build/bin/llama-server --version
free -m
```

- 새 llama.cpp(`llama.cpp-new`)가 젬마3 GGUF도 `--jinja`와 함께 문제없이 띄우는지
- 블록으로 실행한 학생 코드가 root로 도는지 (`systemctl`·링크 바꾸기에 필요)

---

## 3. 설계 후보

### A. 링크 바꾸기
새 블록이 `llm-model.gguf` 링크를 고른 파일로 바꾸고 `llama-server`를 다시 시작합니다.

- 좋은 점: 서비스 파일을 안 바꿔도 됩니다. 가장 간단합니다.
- 나쁜 점: 링크가 **계속 바뀐 채로 남습니다**. 그 뒤 예전 블록(`start_llm()`)이나 IDE [대화]도 마지막에 고른
  모델로 뜹니다. 기본 모델이 학생 코드에 따라 바뀌는 셈입니다. root 권한이 필요합니다.

### B. 모델마다 서비스 이름 (systemd 템플릿 유닛) — 추천
- `llama-server.service`와 `llm-model.gguf` 링크는 **그대로** 둡니다. 예전 블록과 IDE [대화]는 지금과 똑같습니다.
- 모델마다 링크를 하나씩 둡니다: `/home/pi/.model/llm/{gemma3,themaker-ko,themaker-en}.gguf`
- 템플릿 유닛 `llama-server@.service`를 하나 더 둡니다. ExecStart는 지금 서비스와 같고, `-m`만
  `/home/pi/.model/llm/%i.gguf`로 바꿉니다.
- 새 블록 → `dialog.start_llm(model="themaker-ko")` → 먼저 `llama-server`와 다른 `llama-server@*`를 끄고
  `systemctl start llama-server@themaker-ko`를 실행합니다.
- 좋은 점: 기본값이 바뀌지 않습니다. 어떤 모델이 떠 있는지 `systemctl list-units 'llama-server*'`로 바로 보입니다.
- 나쁜 점: 유닛 파일이 저장소 밖이라 **이미지 작업**이 됩니다. 끄는 곳을 모두 `llama-server@*`까지
  끄도록 고쳐야 합니다. 대상은 `stop_llm`, IDE `/llm?enable=off`, `/tools`, `/classifier`, `/hwtest`, [정지], 코드 실행입니다.
  빠뜨리면 램 2GB에 두 모델이 같이 올라갈 수 있습니다.

### C. 서비스 없이 파이썬이 직접 띄우기
`start_llm(model=...)`가 llama-server를 `subprocess`로 직접 띄웁니다.

- 좋은 점: 저장소 안에서 끝납니다 (이미지 작업 없음).
- 나쁜 점: 지금 IDE가 `systemctl`로 끄고 켜는 흐름과 따로 놉니다. 프로세스를 잃으면(IDE 재시작) 정리가 어렵습니다.
  검수 서버(`hwtest_proc`)처럼 `pkill -f`로 정리하는 방식을 따라야 합니다.

**어느 안이든 공통**
- 파이썬 API는 **더하기만** 합니다: `set_llm_model(self, model)`(고른 이름을 `self`에 기억)과
  `start_llm(self, port=50020)`의 안쪽 분기. 고른 모델이 없으면(`None`) 지금과 똑같이 `systemctl start llama-server`.
  A안(링크 바꾸기)은 고른 모델이 기본값에 남으므로 이 방식과 맞지 않습니다 → **B안이나 C안**.
- `stop_llm()`은 어느 모델이 떠 있든 끄게 합니다.
- 모델 선택 블록은 드롭다운으로 모델 셋을 고릅니다 (`speech_set_llm_model` 같은 이름). `ko.js`·`en.js`에 글을 넣고, 툴박스(`customblock_toolbox.js`)에 둡니다.
  가이드 문서(`docs/source*/blocks/guide.md`)도 고칩니다.
- 파이보와 파이브레인을 같은 방식으로 합니다.

---

## 4. 같이 정해야 할 것 (설계안에 의견을 적어 주세요)

1. **시스템 글**
   - themaker-ko/en은 학습한 시스템 글(`tools.SYSTEM`/`SYSTEM_EN`)로만 배웠습니다. `call_llm(system_prompt="너는 내 스마트한 비서야")`처럼
     다른 글을 주면 대답이 어떻게 될지 모릅니다 (확인 필요).
   - `system_prompt`를 안 주면 GGUF 템플릿이 학습 때의 글을 넣습니다.
   - themaker를 고른 경우 블록의 시스템 글을 무시할지, 학습한 글 뒤에 붙일지, 그대로 보낼지 정해야 합니다.
2. **대화 웹 (`chat/run_chat.py`)**: `SYSTEM`을 지금 떠 있는 모델에 맞춰야 합니다. 젬마는 지금 글, themaker-ko는
   `tools.SYSTEM`, themaker-en은 `tools.SYSTEM_EN`입니다. 대화 웹이 어떤 모델이 떠 있는지 아는 방법을 정해야 합니다
   (예: B안이면 `systemctl is-active llama-server@<이름>`).
3. **도구**: 블록의 `call_llm`은 도구를 보내지 않습니다 (글자 대화만). 도구(불빛·보기·움직이기)는
   `pibo_chat.py`/`pibrain_chat.py` 방식의 앱에서만 씁니다. 블록에서도 도구를 쓸지는 이번 범위 밖으로 둡니다.
4. **IDE [대화]의 기본 모델**: 지금처럼 `llm-model.gguf` 링크(기본 하나)로 둘지, IDE 화면에도 고르는 곳을 둘지 정합니다.
5. **llama-server 옵션**: 모델마다 같은 옵션(`--jinja --ctx-size 2048 -np 1 --cache-ram 0 -t 4 -ub 256`)으로
   되는지 확인합니다. 젬마는 `--jinja`로도 대답이 정상인지 확인 필요합니다.

---

## 5. 하지 말 것

- 설계안이 정해지기 전에 서비스 파일, 링크, 블록, `speech.py`를 바꾸지 않습니다.
- 예전 블록(`speech_start_llm`)과 `start_llm()`의 동작을 바꾸지 않습니다.
- llama-server를 두 개 동시에 띄우지 않습니다 (램 2GB).
- 젬마 모델 파일과 지금 서비스 파일을 백업 없이 지우거나 덮어쓰지 않습니다.

---

## 6. 보고할 것

1. 2장 "기기에서 먼저 확인할 것"의 결과
2. A·B·C 중 추천안과 까닭. 바꿀 파일과 줄 목록 (저장소 안 / 이미지 작업을 나눠서)
3. 4장 다섯 가지에 대한 의견
4. 확인 필요로 남은 것
