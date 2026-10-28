# openpibo-chat

초·중·고 학생과 일상 대화를 하는 작은 한국어 대화 모델 **themaker(더메이커)**를 만듭니다.
결과물은 llama.cpp의 `llama-server`가 그대로 읽는 파일 하나입니다.

- `out/themaker-0.8b-Q4_K_M.gguf`: Qwen3.5-0.8B를 미세조정하고 어휘를 줄여 imatrix Q4_K_M으로 (약 320MB)
- 모든 기기(파이보, 파이브레인 …) 공통. 어느 기기에서든 "서큘러스에서 만든 학생들을 위한 교육용 AI
  더메이커"로 이야기하고, 기기 이름이나 사양(키, 색, 모터 …)은 말하지 않습니다 (물으면 잘 모른다고).
  시스템 글은 모든 기기에 같은 `tools.SYSTEM`이고, 앱이 보내지 않아도 GGUF 템플릿이 넣습니다.
- 도구 3개(불빛, 보기, 움직이기, `tools.TOOLS`)를 모델이 학생 말을 보고 부릅니다. 기기 앱은 그 호출을
  자기 함수로 실행합니다 (파이보 `pibo_chat.py`, 파이브레인 `pibrain_chat.py`: 움직이기는 못 한다고 말함).
- 영어 모델 `out/themaker-en-0.8b-Q4_K_M.gguf`: 같은 대화를 선생이 쉬운 영어로 옮겨 따로 학습한 것
  (`tools.SYSTEM_EN`, 도구 목록은 같음). 기기에 올릴 때 한국어·영어 중 하나를 고르고 앱은 `--lang en`.

## 데이터와 학습

원본 대화(`data/source.jsonl`, 채점용 `data/source-heldout.jsonl`)는 PC에만 두고 git에 올리지 않습니다.
한 줄에 `{"id", "source", "topic", "turns": [{"user": 학생 말, "other": 상대 말}]}`입니다.
학생 말은 거의 그대로 두고, 상대의 말만 선생 모델(5090 PC의 Qwen3-32B-AWQ, Apache-2.0)이 파이보
말투로 바꿔 씁니다.

| 단계 | 파일 | 내용 |
| --- | --- | --- |
| 파이보 말로 바꿔 쓰기 | `rewrite.py`, `prompts.py` | 선생이 생각을 켜고 상대 말을 열네 살 친구 같은 짧은 존댓말로. 학생과 나눌 만하지 않은 대화는 선생이 뺌. 기호, 다른 나라 글자, 반말, 학생 말 되받아 읊기는 글자로 걸러 뺌 |
| 검사 | `rewrite.py` | 선생이 다시 읽고 자연스러움·알맞음·친구 말투 4점 미만, 저급한 말, 지어낸 사람 경험, 틀린 사실, 안전 문제가 있으면 뺌 (고치지 않음) |
| 도구 대화 | `tool_data.py` | 불빛 켜기·끄기·이어서 바꾸기, 보기, 움직이기, 부르면 안 되는 이야기("난 빨간색이 좋아")를 정해 둔 말 모양으로 4천 개 (선생 없이) |
| 합치기 | `merge_data.py`, `persona.py` | 똑같은 대답은 3번까지만. 대화의 30%에 도구 목록을 함께 보여 줌. 시스템 글은 모두 `tools.SYSTEM`, 말 속 "파이보"는 "더메이커", "저는 로봇이라"는 "저는 AI라"로. 교육용 AI 자기소개 1천 |
| 직접 쓴 대화 | `data/extra.txt` | 꼭 이렇게 말했으면 하는 대화(이름, 만든 곳, 누구를 위한 AI인지, 기기 사양은 모름 등). 학습 때 세 번씩 더 봄 |
| 학습 | `train.py` | Qwen3.5-0.8B 전체 미세조정, 파이보의 대답에만 손실. 끊기면 체크포인트부터 이어서 |
| 어휘 줄이기 | `prune_vocab.py` | 24만 → 약 3만 토큰 |
| GGUF | `export.py` | 학습 데이터로 imatrix를 재서 Q4_K_M. 템플릿은 늘 생각 없이, 시스템 글이 없으면 `tools.SYSTEM`을 넣음 |
| 채점 | `score.py` | 학습에 안 쓴 대화로 최종 모델과 예전 모델을 같은 설정으로 묻고 선생이 매김. 도구는 맞게 불렀는지 정답률 (`results/score.md`) |
| 영어 모델 | `translate.py`, `prompts_en.py`, `data/extra-en.txt` | 거른 한국어 대화(`data/pibo.jsonl`)를 선생이 생각 없이 쉬운 영어로 옮기고, 생각을 켜고 다시 검사(4점 미만은 뺌). 도구 대화·자기소개도 영어로, 학습·어휘 줄이기·GGUF·채점은 같은 방법 (`results-en/score.md`). `run_all.py`가 한국어 다음에 이어서 함 |

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

## 기기에서 (파이보, 파이브레인)

새 llama.cpp가 필요합니다 (Qwen3.5 구조를 읽는 2026년 이후 빌드, 파이보에서는 `-j2`로 빌드).

```bash
llama-server -m themaker-0.8b-Q4_K_M.gguf --jinja -c 2048 -np 1 --cache-ram 0 -t 4 -ub 256 --port 50020 &
sudo ~/.pyenv/bin/python3 pibo_chat.py --url http://127.0.0.1:50020      # 파이보 (tools.py와 같은 폴더)
sudo ~/.pyenv/bin/python3 pibrain_chat.py --url http://127.0.0.1:50020   # 파이브레인
bash pi_bench.sh 지금쓰는.gguf themaker-0.8b-Q4_K_M.gguf               # 메모리와 속도
```

- `--jinja`: GGUF의 대화 템플릿(생각 끄기, 기본 시스템 글, 도구 호출)을 씁니다.
- `-c 2048 -np 1 --cache-ram 0`: 램 2GB에 맞게.
- 앱은 켤 때 시스템 글과 도구 목록(약 530토큰)을 미리 읽혀 둡니다. 그 뒤에는 새로 한 말만 읽습니다.
- openpibo의 Device·카메라·NeoPixel은 root가 필요해 `sudo`로 실행합니다.
- 기기 앱에 붙이는 방법과 확인할 것은 `docs/pibo-handoff.md`.

## 라이선스

Qwen3.5-0.8B, Qwen3 선생 모델, llama.cpp 모두 Apache-2.0 / MIT입니다. `LICENSE`를 보세요.
