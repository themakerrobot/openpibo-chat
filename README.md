# openpibo-chat

파이보(라즈베리파이 4B, 램 2GB) 안에서 직접 도는 작은 대화 모델을 만듭니다.
아이와 짧게 대화하고, "춤춰 줘", "눈 파랗게 해 줘" 같은 말을 파이보 동작으로
바꿉니다(도구 호출). 결과물은 llama.cpp의 `llama-server`가 그대로 읽는 GGUF(Q4)
파일입니다.

## 무엇을 하나요

| 단계 | 파일 | 내용 |
| --- | --- | --- |
| 데이터 만들기 | `make_data.py`, `prompts.py` | 5090에서 선생 모델(Qwen3-32B-AWQ, 안 뜨면 14B-AWQ, 8B; 모두 Apache-2.0)이 주제 200여 개 × 말하는 사람 12종을 섞어 대화 2만 4천 개, 도구 호출 8천 개를 씁니다. 말투 규칙(존댓말 한두 문장, 기호·이모지 없음, 모르면 모른다고)에 어긋나거나 없는 도구를 부른 답은 버립니다. 학습에 안 쓴 700개는 채점용으로 따로 둡니다. |
| 직접 쓴 대화 | `data/extra.txt` | 파이보가 꼭 이렇게 말했으면 하는 대화(이름, 만든 곳 등). 학습 때 세 번씩 봅니다. |
| 미세조정 | `train.py` | Qwen3-0.6B(Q4로 약 0.4GB)와 Qwen3-1.7B(약 1.1GB)를 전체 미세조정. 파이보의 대답과 도구 호출에만 손실을 줍니다. |
| 채점 | `evaluate.py` | 지금 쓰는 젬마3 1B 원본, Qwen3 원본, 미세조정한 모델이 같은 질문에 답하고, 선생 모델이 대화를 네 가지로 채점합니다. 도구는 정답률과 잘못 부른 비율을 셉니다. |
| GGUF | `export.py` | Q4_K_M으로 바꾸고, PC의 llama-server로 대화와 도구 호출이 되는지 확인합니다. |
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

## 파이보에서

```bash
llama-server -m pibo-qwen3-0.6b-Q4_K_M.gguf --jinja -c 2048 --port 8080 &
python3 pibo_chat.py                         # tools.py 와 같은 폴더에서
bash pi_bench.sh 지금쓰는-gemma.gguf pibo-qwen3-0.6b-Q4_K_M.gguf
```

`--jinja`가 있어야 도구 호출을 알아듣습니다. 도구 목록(약 800토큰)은 처음 한 번만
읽고 그다음부터는 llama-server가 기억해 둔 것을 씁니다.

## 라이선스

코드는 Apache-2.0. 학습 시작점 Qwen3-0.6B/1.7B와 선생 모델 Qwen3-32B도
Apache-2.0이라, 만든 모델도 같은 조건으로 쓸 수 있습니다. 비교용으로 채점하는
젬마3 1B는 젬마 이용약관을 따릅니다(학습에는 쓰지 않습니다).
