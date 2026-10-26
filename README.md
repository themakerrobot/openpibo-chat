# openpibo-chat

초·중·고 학생과 일상 대화를 하는 작은 한국어 대화 모델 **themaker(더메이커)**를 만듭니다.
결과물은 llama.cpp의 `llama-server`가 그대로 읽는 파일 하나입니다.

- `out/themaker-0.8b-Q4_K_M.gguf`: Qwen3.5-0.8B를 미세조정하고 어휘를 줄여 imatrix Q4_K_M으로 (약 320MB)
- 모든 기기(파이보, 파이브레인 …) 공통. 모델은 대화만 하고 도구를 모릅니다.
- 기기 이름과 소개는 앱이 시스템 글로 줍니다. 시스템 글이 없으면 "더메이커"로 대답합니다.
- 동작(춤, 눈 색, 멈춤 …)은 각 기기의 앱이 말을 보고 실행합니다 (파이보: `pibo_chat.py`).

## 데이터와 학습

모든 데이터는 5090 PC의 선생 모델 Qwen3-32B-AWQ(Apache-2.0, 안 뜨면 14B-AWQ, 8B)가 쓰고, 같은
선생이 생각을 켜고 사실·한국어·존댓말을 다시 봅니다. 학년 묶음(초저·초고·중·고)을 고르게,
대답은 초등 70자·중고 120자 안쪽입니다.

| 단계 | 파일 | 내용 |
| --- | --- | --- |
| 친구 대화 6만 | `make_data.py --chat`, `prompts.py` | 학생 주제 200여 개 × 장면 22 × 말 꺼내는 방식 14 × 흐름. 1~4번 주고받기 |
| 대화 이어 가기 1만 2천 | `make_data.py --lead` | 일상 주제로 4~8번 주고받기. 이어 묻는 말에 구체적으로, 같은 질문을 바꿔 물으면 다른 말로, 짧은 대답엔 쉬운 질문·이야깃거리·놀이로, 기분·고민·반대 의견·화제 전환·추천·계획 |
| 영어 9천 | `make_data.py --english --learn` | 학생이 영어로 말하면 쉬운 영어로 (틀린 영어는 맞는 표현으로 다시 말해 줌), 한국어로 묻는 영어 공부는 한국어로 설명 |
| 사실·존댓말 검사 | `filter_chat.py` | 틀린 사실, 정설을 얼버무린 말, 어색한 한국어, 되풀이, 반말을 고쳐 쓰거나 뺌 |
| 합치기 | `merge_data.py`, `devices.py` | 똑같은 파이보 대답은 3번까지만. 25%는 이름을 다른 이름(더메이커, 파이브레인 …)으로 바꾸고(조사도 받침에 맞게) 일부는 기기 소개를 시스템 글에. 시스템 글의 이름·소개를 보고 하는 자기소개 3천 |
| 직접 쓴 대화 | `data/extra.txt` | 꼭 이렇게 말했으면 하는 대화(이름, 만든 곳 등). 학습 때 세 번씩 더 봄 |
| 학습 | `train.py` | Qwen3.5-0.8B 전체 미세조정, 파이보의 대답에만 손실. 끊기면 체크포인트부터 이어서 |
| 어휘 줄이기 | `prune_vocab.py` | 24만 → 약 3만 토큰. 한글 토큰은 모두, 그 밖에는 학습 데이터에 나온 토큰과 흔한 영어 조각만 |
| GGUF | `export.py` | 학습 데이터로 imatrix를 재서 Q4_K_M. 템플릿은 늘 생각 없이, 시스템 글이 없으면 더메이커로 |

## 5090 PC에서 한 번에

```bash
git clone https://github.com/themakerrobot/openpibo-chat ~/openpibo-chat && cd ~/openpibo-chat
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
pip install ~/llama.cpp/gguf-py && cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_NATIVE=ON \
  && cmake --build ~/llama.cpp/build -j --target llama-quantize llama-server llama-imatrix
nohup flock ~/longrun/.lock .venv/bin/python run_all.py >> driver.log 2>&1 &   # 끊겨도 다시 돌리면 이어서
.venv/bin/python run_all.py --status   # 진행 상황 (REPORT.md)
.venv/bin/python show.py 20 lead data/lead-clean.jsonl   # 만든 데이터 훑어보기
```

로그는 `logs/<단계>.log`, 기기로 보낼 파일은 `out/themaker-0.8b-Q4_K_M.gguf`,
그 파일로 몇 가지 물어본 대답은 `out/themaker-0.8b-Q4_K_M-check.json`.

## 기기에서 (파이보, 라즈베리파이 4B 2GB)

새 llama.cpp가 필요합니다 (Qwen3.5 구조를 읽는 2026년 이후 빌드, 파이보에서는 `-j2`로 빌드).

```bash
llama-server -m themaker-0.8b-Q4_K_M.gguf --jinja -c 2048 -np 1 --cache-ram 0 -t 4 -ub 256 --port 8080 &
python3 pibo_chat.py --url http://127.0.0.1:8080          # tools.py 와 같은 폴더에서
bash pi_bench.sh 지금쓰는.gguf themaker-0.8b-Q4_K_M.gguf  # 메모리와 속도
```

- `--jinja`: GGUF의 대화 템플릿(생각 끄기, 기본 시스템 글)을 씁니다.
- `-c 2048 -np 1 --cache-ram 0`: 램 2GB에 맞게 (새 llama.cpp 기본은 자리 4개, 지난 질문을 램에 8GB까지).
- 파이보 실측(예전 같은 크기의 모델): 최대 515MB, 대답 약 11초 (젬마3 1B는 872MB, 21초).

`pibo_chat.py`가 하는 일:
- 동작: 멈춰(바로), 춤, 인사, 손 흔들기, 박수, 하이파이브, 앞으로·뒤로, 좌우 돌기, 눈 색·끄기,
  배터리, 앞에 뭐 있어(카메라), 날씨를 말 끝을 보고 바로 실행하고 정해진 말로 대답합니다
  ("어제 춤 학원 갔어"처럼 이야기 속 낱말은 동작이 아님).
- 그 밖에는 모델에 최근 3번의 주고받기와 함께 묻습니다. 되풀이를 줄이는 대답 설정을 쓰고,
  앞에서 한 말과 똑같으면 한 번 더 다르게 묻습니다.
- "대화만 하자" / "이제 움직여도 돼"로 동작을 끄고 켭니다.
- 음성 인식·합성은 `listen()`/`speak()`를 바꿔 끼우면 됩니다. TTS 목소리는 `lang_of(대답)`으로.

다른 기기는 이름·소개·있는 동작만 바꿔 같은 모델을 씁니다:

```bash
python3 pibo_chat.py --name 파이브레인 --about "코딩 로봇, 서큘러스에서 만듦" --tools eye_on,eye_off
```

자세한 적용 순서는 `docs/pibo-handoff.md`.

## 라이선스

Qwen3.5-0.8B, Qwen3 선생 모델, llama.cpp 모두 Apache-2.0 / MIT입니다. `LICENSE`를 보세요.
