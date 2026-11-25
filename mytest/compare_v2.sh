#!/usr/bin/env bash
# v2를 학습 없이/조금만 더 해서 나아지는 판이 있는지 새 시험(mytest)으로 견줍니다 (학습 PC에서).
#
#   cd ~/openpibo-chat && nohup bash mytest/compare_v2.sh > logs/compare-v2.log 2>&1 &
#   QUANT_ONLY=1 bash mytest/compare_v2.sh     # 양자화 견주기만 (학습 없이 10분쯤)
#   LANGS=ko bash mytest/compare_v2.sh         # 한국어만
#
# 견주는 것 (언어마다):
#   themaker-<언어>-v2      지금 배포한 Q4_K_M (기준)
#   themaker-<언어>-v2-q5   같은 모델을 Q5_K_M으로 (imatrix 같음): 양자화로 잃는 것이 큰지
#   themaker-<언어>-v2-f16  같은 모델을 양자화 없이 F16으로: 잃을 수 있는 최대치
#   themaker-<언어>-v2e1    같은 데이터·설정으로 1바퀴만 학습 (2바퀴째에 외우기 시작했는지)
#       예전 체크포인트(1바퀴 끝)는 save_total_limit=2로 지워져서 다시 학습합니다 (언어마다 40~60분)
# 결과: results-compare/mytest-score.md, results-compare-en/mytest-score.md
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
T=mytest
LANGS=${LANGS:-"ko en"}

# GPU에 다른 일이 돌고 있으면 한 번에 1개씩 (같은 32개 묶음, 메모리는 줄고 조금 느려짐)
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${free:-0}" -lt 30000 ]; then BATCH="--batch 1 --accum 32"; else BATCH="--batch 2 --accum 16"; fi
echo "$(date '+%m-%d %H:%M') GPU 빈 메모리 ${free}MiB → $BATCH"

for lang in $LANGS; do
    if [ "$lang" = ko ]; then sfx=""; data=data/train-themaker-kid.jsonl; extra=""
    else sfx="-en"; data=data/train-themaker-kid-en.jsonl; extra="--extra data/extra-en.txt"; fi
    small=runs/themaker-$lang-v2/small
    if [ ! -f "$small/config.json" ]; then echo "$small 이 없어요 (v2 어휘 줄인 모델)"; exit 1; fi

    echo "$(date '+%m-%d %H:%M') [$lang] 양자화 판 만들기 (Q5_K_M, F16)"
    $PY export.py --model "$small" --name "themaker-$lang-v2-0.8b" --quant Q5_K_M --imatrix "$data" --lang "$lang" > "logs/compare-export-$lang-q5.log" 2>&1 \
        || { echo "Q5_K_M 만들기 실패: logs/compare-export-$lang-q5.log"; exit 1; }
    $PY export.py --model "$small" --name "themaker-$lang-v2-0.8b" --quant F16 --lang "$lang" > "logs/compare-export-$lang-f16.log" 2>&1 \
        || { echo "F16 만들기 실패: logs/compare-export-$lang-f16.log"; exit 1; }

    models=("themaker-$lang-v2 out/themaker-$lang-v2-0.8b-Q4_K_M.gguf"
            "themaker-$lang-v2-q5 out/themaker-$lang-v2-0.8b-Q5_K_M.gguf"
            "themaker-$lang-v2-f16 out/themaker-$lang-v2-0.8b-F16.gguf")

    if [ -z "${QUANT_ONLY:-}" ]; then
        run=runs/themaker-$lang-v2e1
        echo "$(date '+%m-%d %H:%M') [$lang] 1바퀴 학습 → $run"
        $PY train.py --base Qwen/Qwen3.5-0.8B --data "$data" --out "$run" --lang "$lang" $extra \
            --lr 2e-5 $BATCH --epochs 1 > "logs/compare-train-$lang.log" 2>&1 \
            || { echo "학습 실패: logs/compare-train-$lang.log"; exit 1; }
        $PY prune_vocab.py --model "$run/final" --out "$run/small" --data "$data" > "logs/compare-prune-$lang.log" 2>&1 \
            || { echo "어휘 줄이기 실패: logs/compare-prune-$lang.log"; exit 1; }
        $PY export.py --model "$run/small" --name "themaker-$lang-v2e1-0.8b" --quant Q4_K_M --imatrix "$data" --lang "$lang" > "logs/compare-export-$lang-e1.log" 2>&1 \
            || { echo "GGUF 만들기 실패: logs/compare-export-$lang-e1.log"; exit 1; }
        models+=("themaker-$lang-v2e1 out/themaker-$lang-v2e1-0.8b-Q4_K_M.gguf")
    fi

    res="results-compare$sfx"; chat="$T/mytest-chat$sfx.jsonl"; tools="$T/mytest-tools$sfx.jsonl"
    names=()
    for pair in "${models[@]}"; do
        set -- $pair
        if [ ! -f "$2" ]; then echo "$2 이 없어 $1은 건너뜀"; continue; fi
        echo "$(date '+%m-%d %H:%M') [$lang] 대답 받기: $1 ($(du -m "$2" | cut -f1)MB)"
        $PY score.py answer --gguf "$2" --name "$1" --lang "$lang" --heldout "$chat" --tools-heldout "$tools" --results "$res"
        names+=("$1")
    done
    $PY $T/score_mytest.py --lang "$lang" --results "$res" --chat "$chat" --tools "$tools" --models "${names[@]}"
done
echo "$(date '+%m-%d %H:%M') 끝: results-compare/mytest-score.md, results-compare-en/mytest-score.md"
