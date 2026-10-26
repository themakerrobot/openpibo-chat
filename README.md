# openpibo-chat

초·중·고 학생과 일상 대화를 하는 작은 한국어 대화 모델 **themaker(더메이커)**를 만듭니다.
결과물은 llama.cpp의 `llama-server`가 그대로 읽는 파일 하나입니다.

- `out/themaker-0.8b-Q4_K_M.gguf`: Qwen3.5-0.8B를 미세조정하고 어휘를 줄여 imatrix Q4_K_M으로 (약 320MB)
- 모든 기기(파이보, 파이브레인 …) 공통. 모델은 대화만 하고 도구를 모릅니다.
- 기기 이름과 소개는 앱이 시스템 글로 줍니다. 시스템 글이 없으면 "더메이커"로 대답합니다.
- 동작(춤, 눈 색, 멈춤 …)은 각 기기의 앱이 말을 보고 실행합니다 (파이보: `pibo_chat.py`).

## 데이터와 학습

원본 대화(`data/source.jsonl`, 채점용 `data/source-heldout.jsonl`)는 PC에만 두고 git에 올리지 않습니다.
한 줄에 `{"id", "source", "topic", "turns": [{"user": 학생 말, "other": 상대 말}]}`입니다.
학생 말은 거의 그대로 두고, 상대의 말만 선생 모델(5090 PC의 Qwen3-32B-AWQ, Apache-2.0)이 파이보
말투로 바꿔 씁니다.

| 단계 | 파일 | 내용 |
| --- | --- | --- |
| 파이보 말로 바꿔 쓰기 | `rewrite.py`, `prompts.py` | 선생이 생각을 켜고 상대 말을 열네 살 친구 같은 짧은 존댓말로. 학생과 나눌 만하지 않은 대화는 선생이 뺌. 기호, 다른 나라 글자, 반말, 학생 말 되받아 읊기는 글자로 걸러 뺌 |
| 검사 | `rewrite.py` | 선생이 다시 읽고 자연스러움·알맞음·친구 말투 4점 미만, 저급한 말, 지어낸 사람 경험, 틀린 사실, 안전 문제가 있으면 뺌 (고치지 않음) |
| 합치기 | `merge_data.py`, `devices.py` | 똑같은 파이보 대답은 3번까지만. 25%는 이름을 다른 이름(더메이커, 파이브레인 …)으로, 일부는 기기 소개를 시스템 글에. 자기소개 3천 |
| 직접 쓴 대화 | `data/extra.txt` | 꼭 이렇게 말했으면 하는 대화(이름, 만든 곳 등). 학습 때 세 번씩 더 봄 |
| 학습 | `train.py` | Qwen3.5-0.8B 전체 미세조정, 파이보의 대답에만 손실. 끊기면 체크포인트부터 이어서 |
| 어휘 줄이기 | `prune_vocab.py` | 24만 → 약 3만 토큰 |
| GGUF | `export.py` | 학습 데이터로 imatrix를 재서 Q4_K_M. 템플릿은 늘 생각 없이, 시스템 글이 없으면 더메이커로 |
| 채점 | `score.py` | 학습에 안 쓴 대화로 최종 모델과 예전 모델을 같은 설정으로 묻고 선생이 매김 (`results/score.md`) |

## 5090 PC에서

```bash
git clone https://github.com/themakerrobot/openpibo-chat ~/openpibo-chat && cd ~/openpibo-chat
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
pip install ~/llama.cpp/gguf-py && cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_NATIVE=ON \
  && cmake --build ~/llama.cpp/build -j --target llama-quantize llama-server llama-imatrix
# data/source.jsonl, data/source-heldout.jsonl을 둔 뒤
.venv/bin/python rewrite.py --data data/source.jsonl --out data/try.jsonl --limit 200   # 먼저 200개
mkdir -p samples && .venv/bin/python show.py 200 all data/try.jsonl > samples/try.txt  # 사람이 읽어 보기
nohup flock ~/longrun/.lock .venv/bin/python run_all.py >> driver.log 2>&1 &             # 괜찮으면 전체
.venv/bin/python run_all.py --status
```

로그는 `logs/<단계>.log`, 기기로 보낼 파일은 `out/themaker-0.8b-Q4_K_M.gguf`, 채점은 `results/score.md`.

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
