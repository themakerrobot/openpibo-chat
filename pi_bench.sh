#!/bin/bash
# 파이보에서: GGUF 모델마다 메모리와 속도를 같은 조건으로 잽니다.
#   bash pi_bench.sh ~/models/gemma-3-1b-it-Q4_K_M.gguf ~/pibo-qwen3-0.6b-Q4_K_M.gguf
# llama.cpp가 다른 곳에 있으면: LLAMA_BIN=/경로/build/bin bash pi_bench.sh ...
BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin}"
PROMPT="공룡은 왜 사라졌는지 아이에게 설명해 줘."
printf "%-45s %10s %12s %12s\n" "모델" "최대메모리" "읽기(토큰/초)" "말하기(토큰/초)"
for m in "$@"; do
  # 최대 메모리: 모델을 올리고 64토큰을 말하는 동안 가장 많이 쓴 양
  mem=$( { /usr/bin/time -v "$BIN/llama-cli" -m "$m" -p "$PROMPT" -n 64 -t 4 -c 2048 \
           -no-cnv --no-warmup >/dev/null 2>/dev/null; } 2>&1 | awk '/Maximum resident/ {print $6}')
  # 속도: 프롬프트 128토큰 읽기, 64토큰 만들기
  speed=$("$BIN/llama-bench" -m "$m" -p 128 -n 64 -t 4 -o csv 2>/dev/null | tail -n +2)
  pp=$(echo "$speed" | awk -F, 'NR==1 {gsub(/"/,"",$0); print $(NF-1)}')
  tg=$(echo "$speed" | awk -F, 'NR==2 {gsub(/"/,"",$0); print $(NF-1)}')
  printf "%-45s %8s MB %12s %12s\n" "$(basename "$m")" "$((mem / 1024))" "$pp" "$tg"
done
