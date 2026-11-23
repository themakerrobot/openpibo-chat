#!/usr/bin/env bash
# gemma3 1B (처음 쓰던 모델), 더메이커 v1, v2를 새 시험(mytest)으로 견줍니다 (학습 PC에서).
#
#   cd ~/openpibo-chat && bash mytest/run_mytest.sh
#
# 1. 시험 파일이 학습 데이터와 겹치는지 먼저 봄 (dedup_check.py, data/ 안의 학습 파일 전부)
# 2. 언어마다 세 모델의 대답을 받음 (score.py answer, llama-server GPU, 기기와 같은 SAMPLING)
#    gemma3는 학습한 도구 형식이 없어 도구 질문에서 서버가 오류를 내면 대화만 받음
# 3. 규칙으로 점수: score_mytest.py → results-mytest/mytest-score.md, results-mytest-en/mytest-score.md
#    대답 나란히 보기: results-mytest*/mytest-answers.md
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
T=mytest
GEMMA=out/gemma-3-1b-it-Q4_K_M.gguf

if [ ! -f "$GEMMA" ]; then
    echo "gemma3 1B GGUF 받기 → $GEMMA"
    .venv/bin/hf download ggml-org/gemma-3-1b-it-GGUF gemma-3-1b-it-Q4_K_M.gguf --local-dir out \
        || .venv/bin/huggingface-cli download ggml-org/gemma-3-1b-it-GGUF gemma-3-1b-it-Q4_K_M.gguf --local-dir out \
        || echo "gemma3를 못 받았어요: 기기의 /home/pi/.model/llm/gemma-3-1b-it-Q4_K_M.gguf를 $GEMMA 로 복사해 두세요"
fi

echo "== 학습 데이터와 겹치는지"
$PY $T/dedup_check.py --test $T/mytest-*.jsonl --train data/*.jsonl data/*.txt | grep -v "^학습"

for lang in ko en; do
    if [ "$lang" = ko ]; then sfx=""; v1=out/themaker-0.8b-Q4_K_M.gguf; else sfx="-en"; v1=out/themaker-en-0.8b-Q4_K_M.gguf; fi
    res="results-mytest$sfx"; chat="$T/mytest-chat$sfx.jsonl"; tools="$T/mytest-tools$sfx.jsonl"
    for pair in "gemma3-1b $GEMMA" "themaker-$lang-v1 $v1" "themaker-$lang-v2 out/themaker-$lang-v2-0.8b-Q4_K_M.gguf"; do
        set -- $pair
        if [ ! -f "$2" ]; then echo "$2 이 없어 $1은 건너뜀"; continue; fi
        echo "$(date '+%m-%d %H:%M') 대답 받기 ($lang): $1"
        if ! $PY score.py answer --gguf "$2" --name "$1" --lang "$lang" --heldout "$chat" --tools-heldout "$tools" --results "$res"; then
            echo "$1: 도구 질문에서 실패해 대화만 다시 받음 (도구 점수는 비어 나옴)"
            $PY score.py answer --gguf "$2" --name "$1" --lang "$lang" --heldout "$chat" --tools-heldout /dev/null --results "$res"
        fi
    done
    $PY $T/score_mytest.py --lang "$lang" --results "$res" --chat "$chat" --tools "$tools" \
        --models gemma3-1b "themaker-$lang-v1" "themaker-$lang-v2"
done
echo "$(date '+%m-%d %H:%M') 끝: results-mytest/mytest-score.md, results-mytest-en/mytest-score.md"
