# openpibo-chat

파이보(라즈베리파이 4B, 램 2GB) 안에서 직접 도는 작은 대화 모델을 만듭니다.
아이와 짧게 대화하고, "춤춰 줘", "눈 파랗게 해 줘" 같은 말을 파이보 동작으로
바꿉니다(도구 호출). 결과물은 llama.cpp의 `llama-server`가 그대로 읽는 GGUF(Q4)
파일입니다.

## 무엇을 하나요

| 단계 | 파일 | 내용 |
| --- | --- | --- |
| 데이터 만들기 | `make_data.py`, `prompts.py` | 5090에서 선생 모델(Qwen3-32B-AWQ, 안 뜨면 14B-AWQ, 8B; 모두 Apache-2.0)이 주제 200여 개 × 말하는 사람 12종을 섞어 대화 2만 4천 개, 도구 호출 8천 개를 씁니다. 말투 규칙(존댓말 한두 문장, 기호·이모지 없음, 모르면 모른다고)에 어긋나거나 없는 도구를 부른 답은 버립니다. 학습에 안 쓴 700개는 채점용으로 따로 둡니다. |
| 초·중·고 데이터 | `make_data.py --audience school` | 파이보의 주 사용자인 초·중·고 학생과의 대화 2만 4천 개. 학년 묶음(초저·초고·중·고)을 고르게, 대답은 초등 70자·중고 120자 안쪽, 친구처럼 하는 대화가 대부분이고 호기심·공부 질문은 15%. 도구 예시는 거른 어린이 데이터의 것을 다시 씁니다. 채점용 질문은 어린이 채점용의 대화 질문 400개를 같은 자리에서 초·중·고 대화로 바꾼 것(`--set school`, 결과는 `results-school/`)이라 도구 문제는 그대로입니다. |
| 사실 검사 | `filter_chat.py` | 생각을 끈 선생이 쓴 대화에 틀린 사실("공룡은 수천 년 전에 살았어요"), 정설을 얼버무린 말, 어색한 한국어가 섞여 있었습니다. 초·중·고 대화 데이터는 선생이 생각을 켜고 하나씩 다시 보고, 문제 있는 파이보 말은 고쳐 쓰고 못 고친 예시는 뺍니다(`data/train-school.jsonl`, 판단은 `data/chat-check.jsonl`). 이 데이터로 `qwen3-0.6b-school`, `qwen3.5-0.8b-school`을 학습해 Q4_K_M·Q6_K·Q8_0으로 채점합니다. |
| 직접 쓴 대화 | `data/extra.txt` | 파이보가 꼭 이렇게 말했으면 하는 대화(이름, 만든 곳 등). 학습 때 세 번씩 봅니다. |
| 미세조정 | `train.py` | Qwen3-0.6B(Q4로 약 0.4GB)를 전체 미세조정. 파이보의 대답과 도구 호출에만 손실을 줍니다. 1.7B는 KV 캐시까지 합치면 파이보에서 젬마3 1B보다 램을 더 많이 써서 쓰지 않습니다. |
| 증류 (켤 때만) | `distill.py`, `train.py --kd` | `state/kd.on`이 있으면 선생 모델이 학습 데이터의 파이보 대답을 토큰마다 어떻게 보는지(상위 20개 확률)를 구해 두고, 0.6B를 정답 문장과 그 확률을 함께 따라 배우게 해 `qwen3-0.6b-kd`로 나란히 채점합니다. 선생과 학생은 토크나이저가 같아야 합니다(Qwen3는 모두 같음). |
| 채점 | `evaluate.py` | 지금 쓰는 젬마3 1B 원본, Qwen3 원본, 미세조정한 모델이 같은 질문에 답하고, 선생 모델이 대화를 네 가지로 채점합니다. 따로 선생이 생각을 켜고 틀린 사실과 얼버무린 대답의 비율을 셉니다. 도구는 정답률과 잘못 부른 비율을 셉니다. |
| GGUF | `export.py` | Q4_K_M, Q5_K_M, Q6_K, Q8_0으로 바꾸고, PC의 llama-server로 대화와 도구 호출이 되는지 확인합니다. 양자화마다 같은 채점용 질문에 답하게 해(`evaluate.py gguf`) 선생 모델이 채점합니다. 0.6B는 작아서 Q4에서 많이 나빠집니다. |
| 파이보 | `pibo_chat.py`, `pi_bench.sh` | 파이보에서 모델과 대화하며 도구를 openpibo로 실행합니다. 모델마다 메모리와 속도를 잽니다. |

도구 목록은 `tools.py` 하나에 있습니다: 동작 28가지(`set_motion`), 눈 색(`eye_on`,
`eye_off`), 카메라로 물체 찾기, 배터리, 날씨.

## 5090 PC에서 한 번에

```bash
git clone https://github.com/themakerrobot/openpibo-chat ~/openpibo-chat && cd ~/openpibo-chat
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
pip install ~/llama.cpp/gguf-py && cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_NATIVE=ON \
  && cmake --build ~/llama.cpp/build -j --target llama-quantize llama-server
python run_all.py            # 끊겨도 다시 돌리면 이어서
python run_all.py --status   # 진행 상황과 REPORT.md
python show.py 30            # 만든 데이터 훑어보기
```

로그는 `logs/<단계>.log`, 결과는 `REPORT.md`, 파이보로 보낼 파일은 `out/*.gguf`.

## PC에서 먼저 대화해 보기

`out/*.gguf`가 생기면 PC의 CPU로 바로 대화해 볼 수 있습니다(GPU는 쓰지 않습니다).
openpibo가 없으니 `--fake`로 도구는 부른 척만 합니다.

```bash
~/llama.cpp/build/bin/llama-server -m out/pibo-qwen3-0.6b-Q4_K_M.gguf --jinja -c 2048 -np 1 --port 8080 > /tmp/pibo-llama.log 2>&1 &
.venv/bin/python pibo_chat.py --fake         # 그만 하려면 "그만", 서버는 kill %1
```

## 파이보에서

```bash
llama-server -m pibo-qwen3-0.6b-Q4_K_M.gguf --jinja -c 2048 -np 1 --cache-ram 0 --port 8080 &
python3 pibo_chat.py                         # tools.py 와 같은 폴더에서
bash pi_bench.sh 지금쓰는-gemma.gguf pibo-qwen3-0.6b-Q4_K_M.gguf
```

- `--jinja`가 있어야 도구 호출을 알아듣습니다. 도구 목록(약 800토큰)은 처음 한 번만
  읽고 그다음부터는 llama-server가 기억해 둔 것을 씁니다.
- `-c 2048`을 꼭 주세요. 새 llama.cpp는 안 주면 모델이 배운 최대 길이(Qwen3 4만 토큰)로
  KV 캐시를 잡으려 합니다. Qwen3는 KV 캐시가 젬마3 1B보다 훨씬 커서(2048토큰에 224MB,
  젬마는 30MB) 램 2GB에서는 모자랍니다. 더 줄이려면 `-ctk q8_0 -ctv q8_0 -fa on`(절반).
- `-np 1`: 새 llama.cpp는 기본으로 4명 몫의 자리를 만듭니다. 파이보는 한 명이면 됩니다.
- `--cache-ram 0`: 새 llama.cpp는 지난 질문을 램에 8GB까지 쌓아 둡니다. 옛 버전에는
  이 설정이 없으니(없다고 나오면) 빼세요.
- 생각(thinking)은 GGUF의 템플릿에서 꺼 두었습니다(`export.py`). 앱이 따로 보낼 것은 없습니다.

## 라이선스

코드는 Apache-2.0. 학습 시작점 Qwen3-0.6B와 선생 모델 Qwen3-32B도
Apache-2.0이라, 만든 모델도 같은 조건으로 쓸 수 있습니다. 비교용으로 채점하는
젬마3 1B는 젬마 이용약관을 따릅니다(학습에는 쓰지 않습니다).
