# openpibo-chat

초·중·고 학생과 일상 대화를 하는 작은 한국어·영어 대화 모델 **themaker(더메이커)**를 만듭니다.
결과물은 llama.cpp의 `llama-server`가 그대로 읽는 GGUF 파일 하나입니다.

## 지금 기기에 들어간 판 (v2)

| 기기 파일 (`/home/pi/.model/llm/`) | 판 | 학습 PC 파일 | 크기 |
| --- | --- | --- | --- |
| `themaker-ko.gguf` | themaker-ko-v2-0.8b-Q4_K_M | `out/themaker-ko-v2-0.8b-Q4_K_M.gguf` | 약 321MB |
| `themaker-en.gguf` | themaker-en-v2-0.8b-Q4_K_M | `out/themaker-en-v2-0.8b-Q4_K_M.gguf` | 약 330MB |
| `general.gguf` | gemma-3-1b-it-Q4_K_M (처음 쓰던 모델, 그대로) | - | 약 806MB |

- 배포: Hugging Face [`leeyunjai/themaker`](https://huggingface.co/leeyunjai/themaker)의 `.model/`이 기기의 `/home/pi/.model`에
  그대로 들어갑니다. 파일마다 sha256과 판 이름은 `.model/VERSION`에 있습니다 (v1은 그 저장소의 이전 커밋에 남아 있음).
- Qwen3.5-0.8B를 미세조정하고 어휘를 줄여(24만 → 약 3만 토큰) imatrix Q4_K_M으로 만든 것입니다.
- 모든 기기(파이보, 파이브레인 …) 공통. 어느 기기에서든 "서큘러스에서 만든 학생들을 위한 교육용 AI
  더메이커"로 이야기하고, 기기 이름이나 사양(키, 색, 모터 …)은 말하지 않습니다 (물으면 잘 모른다고).
  시스템 글은 모든 기기에 같은 `tools.SYSTEM`(영어는 `tools.SYSTEM_EN`)이고, 앱이 보내지 않아도 GGUF 템플릿이 넣습니다.
- 도구 3개(불빛, 보기, 움직이기, `tools.TOOLS`)를 모델이 학생 말을 보고 부릅니다. 기기 앱은 그 호출을
  자기 함수로 실행합니다 (파이보 `pibo_chat.py`, 파이브레인 `pibrain_chat.py`: 움직이기는 못 한다고 말함).
  v1과 v2는 시스템 글·도구·템플릿이 같아서 기기 앱은 그대로 쓰고 모델 파일만 바꿉니다.

### v1 → v2에서 바뀐 것

- 학습 데이터를 아이 말투 대화로 새로 꾸림 (`data/train-themaker-kid.jsonl`, `data/train-themaker-kid-en.jsonl`).
- 대답이 조금 길어짐 (한국어 평균 28자 → 42자, 영어 60자 → 88자). 파이보에서 대답이 끝나기까지 그만큼 더 걸릴 수 있음.
- 질문에 맞는 내용을 담는 비율과, 도구를 부르면 안 될 때 안 부르는 비율이 올라감 (아래 점수).

## 점수 (새 시험 `mytest/`)

학습 데이터와 따로 손으로 쓴 시험입니다 (대화 30개 × 2번 묻기 = 60개, 도구 60개 = 부를 것 40 + 안 부를 것 20).
`data/` 안의 학습 파일과 학생 말이 똑같은 것 0개, 비슷한 것 1개 (380개 중). 선생 모델 없이 규칙으로만 셉니다.
세 모델 모두 같은 시스템 글, 기기와 같은 대답 설정(`pibo_chat.SAMPLING`)으로 묻습니다.

| 한국어 | 종합 | 대화 종합 | 도구 종합 | 확인 통과 | 존댓말 | 짧음 (70자) | 기호 없음 | 도구 정답 | 안 부름 | 평균 글자 수 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma3 1B | 46 | 42 | 50 | 56% | 35% | 47% | 28% | 0% | (100%) | 99 |
| 더메이커 v1 | 81 | 82 | 80 | 30% | 100% | 100% | 100% | 80% | 80% | 28 |
| 더메이커 v2 | **88** | **91** | **85** | 65% | 100% | 100% | 100% | 80% | 90% | 42 |

| 영어 | 종합 | 대화 종합 | 도구 종합 | 확인 통과 | 영어만 | 짧음 (160자) | 기호 없음 | 도구 정답 | 안 부름 | 평균 글자 수 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemma3 1B | 59 | 68 | 50 | 67% | 100% | 52% | 52% | 0% | (100%) | 182 |
| 더메이커 v1 | 84 | 85 | 82 | 40% | 100% | 100% | 100% | 75% | 90% | 60 |
| 더메이커 v2 | **90** | **93** | **88** | 70% | 100% | 100% | 100% | 80% | 95% | 88 |

- 확인 통과: 대답에 꼭 있어야 할 낱말이 하나라도 있는지 (늦은 밤이면 잠·쉬기, 모르는 것이면 모른다, 정체면 AI·더메이커 …).
  대답이 길수록 걸리기 쉬워서 v2에 조금 유리합니다.
- gemma3는 이 도구 형식을 배우지 않아 도구를 한 번도 부르지 않았습니다. 그래서 "안 부름 100%"는 잘한 것이 아닙니다.
- v2가 아직 틀리는 것: 도구를 쓴 뒤 "고마워", "웃기다"에 또 부름 / 방향을 반대로 움직임("back up" → 앞으로) /
  돌려 말한 부탁("불빛 꺼도 돼", "인사드려")을 못 알아들음 / 물건으로 말한 색(귤 → 주황)을 못 맞힘.
  다음 데이터에서 보강할 것.

다시 돌리기: `bash mytest/run_mytest.sh` (gemma3가 없으면 받고, 겹침 확인 → 세 모델 대답 → 점수까지).
결과는 `results-mytest/mytest-score.md`, `results-mytest-en/mytest-score.md`, 질문별 대답을 나란히 놓은 `mytest-answers.md`.

## 데이터와 학습

원본 대화는 PC에만 두고 git에 올리지 않습니다.

| 단계 | 파일 | 내용 |
| --- | --- | --- |
| 도구 대화 | `tool_data.py` | 불빛 켜기·끄기·이어서 바꾸기, 보기, 움직이기, 부르면 안 되는 이야기("난 빨간색이 좋아")를 정해 둔 말 모양으로 (선생 없이) |
| 직접 쓴 대화 | `data/extra.txt`, `data/extra-en.txt` | 꼭 이렇게 말했으면 하는 대화(이름, 만든 곳, 누구를 위한 AI인지, 기기 사양은 모름 등). 학습 때 세 번씩 더 봄 |
| 학습 | `train.py` | Qwen3.5-0.8B 전체 미세조정, 더메이커의 대답에만 손실. 끊기면 체크포인트부터 이어서 |
| 어휘 줄이기 | `prune_vocab.py` | 24만 → 약 3만 토큰 |
| GGUF | `export.py` | 학습 데이터로 imatrix를 재서 Q4_K_M. 템플릿은 늘 생각 없이, 시스템 글이 없으면 `tools.SYSTEM`을 넣음. 만든 뒤 확인 질문 10개를 바로 물어 봄 |
| 채점 | `score.py`, `mytest/` | `score.py answer`로 학습에 안 쓴 대화의 대답과 도구 호출을 받고, `mytest/score_mytest.py`가 규칙으로 점수 |
| v1 때 쓰던 것 | `rewrite.py`, `merge_data.py`, `persona.py`, `translate.py`, `run_all.py` | v1 데이터를 선생 모델로 바꿔 쓰고 합치던 단계. v2에는 쓰지 않음 |

v2 학습 (5090 PC, 언어마다 약 1시간 20분, 2바퀴):

```bash
.venv/bin/python train.py --base Qwen/Qwen3.5-0.8B --data data/train-themaker-kid.jsonl --out runs/themaker-ko-v2 --lang ko --lr 2e-5 --batch 2 --accum 16
.venv/bin/python prune_vocab.py --model runs/themaker-ko-v2/final --out runs/themaker-ko-v2/small --data data/train-themaker-kid.jsonl
.venv/bin/python export.py --model runs/themaker-ko-v2/small --name themaker-ko-v2-0.8b --quant Q4_K_M --imatrix data/train-themaker-kid.jsonl --lang ko
# 영어: --data data/train-themaker-kid-en.jsonl --lang en --extra data/extra-en.txt, 이름 themaker-en-v2
```

- 영어는 `--extra data/extra-en.txt`를 꼭 줍니다 (주지 않으면 `--data` 옆의 한국어 `extra.txt`를 씀).
- `flash-linear-attention`과 `causal-conv1d`를 깔면 한 걸음이 약 3.1초에서 1.6초로 빨라집니다.

## 5090 PC 준비

```bash
git clone https://github.com/themakerrobot/openpibo-chat ~/openpibo-chat && cd ~/openpibo-chat
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
pip install ~/llama.cpp/gguf-py && cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_NATIVE=ON \
  && cmake --build ~/llama.cpp/build -j --target llama-quantize llama-server llama-imatrix
```

## 기기에서 (파이보, 파이브레인)

모델은 Hugging Face에서 받아 `/home/pi/.model`에 넣고 sha256을 대조합니다.

```bash
huggingface-cli download leeyunjai/themaker --include ".model/*" --local-dir /home/pi
cd /home/pi/.model && sed -n '/^## sha256/,$p' VERSION | tail -n +2 | sha256sum -c --quiet && echo 같음
sudo systemctl restart llama-server@themaker-ko     # 떠 있던 서버는 다시 띄워야 새 파일을 읽음 (영어는 @themaker-en)
```

- 실행은 `llama-server@<이름>` (llama.cpp b11889, `--ctx-size 2048`, 자세한 것은 `.model/VERSION`).
  모델 고르기 블록과 서비스 설정은 `docs/model-select-handoff.md`.
- 앱 실행 (openpibo의 Device·카메라·NeoPixel은 root가 필요해 `sudo`):
  ```bash
  sudo ~/.pyenv/bin/python3 pibo_chat.py --url http://127.0.0.1:50020      # 파이보 (tools.py와 같은 폴더)
  sudo ~/.pyenv/bin/python3 pibrain_chat.py --url http://127.0.0.1:50020   # 파이브레인
  bash pi_bench.sh 예전.gguf 새.gguf                                        # 메모리와 속도 견주기
  ```
  `--url`은 llama-server가 듣는 주소입니다. 영어 모델이면 앱에 `--lang en`.
- 앱은 켤 때 시스템 글과 도구 목록(약 530토큰)을 미리 읽혀 둡니다. 그 뒤에는 새로 한 말만 읽습니다.
- 기기 앱에 붙이는 방법과 확인할 것은 `docs/pibo-tools-handoff.md`(파이보), `docs/pibrain-tools-handoff.md`(파이브레인).

## 기기 없이 시험

- 브라우저 실행기 [`webrunner/`](https://themakerrobot.github.io/openpibo-chat/) (GitHub Pages): GGUF 파일을 골라 브라우저 안(wllama, WASM)에서
  대화·도구 호출. 모델 파일은 올리지 않고 그때그때 고릅니다. 도구 묶음은 "더메이커 한국어"/"더메이커 영어",
  기기와 비슷하게 보려면 temperature 0.7. 자세한 것은 `webrunner/README.md`.
- 학습 PC에서 `python web_chat.py` → `http://<학습 PC IP>:8500`: 더메이커 한국어·영어·젬마3를 골라 대화.
- 파이썬이 없는 윈도우 PC: `python make_web.py --package out/themaker-win` → `out/themaker-win.zip`
  (llama.cpp 윈도우판 + 모델 + 웹 페이지 + run-ko.bat/run-en.bat).

## 라이선스

Qwen3.5-0.8B, llama.cpp 모두 Apache-2.0 / MIT입니다. `LICENSE`를 보세요.
Gemma 3는 Gemma Terms of Use를 따릅니다.
