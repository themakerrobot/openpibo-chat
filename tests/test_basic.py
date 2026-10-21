"""GPU 없이 돌아가는 부분: 도구 검사, 선생 답 검사, 학습 정답 자르기, 파이보 대화 흐름."""

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import SYSTEM, TOOLS, check_call, same_call


def test_tool_arguments_are_checked():
    assert check_call("set_motion", {"name": "dance1"}) is None
    assert check_call("set_motion", {"name": "moonwalk"})
    assert check_call("eye_on", {"r": 0, "g": 0, "b": 255}) is None
    assert check_call("eye_on", {"r": 0, "g": 0, "b": 300})
    assert check_call("eye_on", {"r": 0, "g": 0})
    assert check_call("eye_off", {}) is None
    assert check_call("fly", {})
    assert check_call("get_weather", '{"region": "서울"}') is None


def test_close_colours_count_as_the_same_call():
    a = {"name": "eye_on", "arguments": {"r": 0, "g": 0, "b": 255}}
    assert same_call(a, {"name": "eye_on", "arguments": {"r": 20, "g": 10, "b": 230}})
    assert not same_call(a, {"name": "eye_on", "arguments": {"r": 255, "g": 0, "b": 0}})


def test_teacher_replies_become_examples_or_are_dropped():
    from prompts import to_example

    rng = random.Random(0)
    chat = {"kind": "chat"}
    good = '```json\n{"turns": [{"user": "공룡 어디 갔어?", "pibo": "아주 오래전에 사라졌어요. 화석으로 남아 있어요."}]}\n```'
    ex = to_example(chat, good, rng)
    assert ex["messages"][0]["content"] == SYSTEM and ex["messages"][-1]["role"] == "assistant"
    assert to_example(chat, '{"turns": [{"user": "안녕", "pibo": "안녕하세요 😀"}]}', rng) is None
    assert to_example(chat, "그냥 글", rng) is None

    tool = {"kind": "tool", "fn": "set_motion"}
    ok = json.dumps({"user": "춤춰 줘", "calls": [{"name": "set_motion",
                     "arguments": {"name": "dance2"}}], "result": "완료",
                     "pibo": "신나게 춤췄어요!"}, ensure_ascii=False)
    ex = to_example(tool, ok, rng)
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    bad = ok.replace("dance2", "moonwalk")
    assert to_example(tool, bad, rng) is None


class FakeTok:
    """Qwen3의 대화 템플릿을 그대로 쓰는 토크나이저 흉내 (글자 하나가 토큰 하나)."""

    def __init__(self):
        jinja2 = pytest.importorskip("jinja2")
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
        self.t = env.from_string((ROOT / "templates" / "qwen3.jinja").read_text())

    def apply_chat_template(self, messages, tools=None, tokenize=False,
                            add_generation_prompt=False, enable_thinking=True):
        return self.t.render(messages=messages, tools=tools,
                             add_generation_prompt=add_generation_prompt,
                             enable_thinking=enable_thinking)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_only_pibos_turns_are_learned_and_they_end_the_turn():
    from prompts import to_example
    from train import targets

    tok = FakeTok()
    rng = random.Random(0)
    tool = {"kind": "tool", "fn": "eye_on"}
    ex = to_example(tool, json.dumps({"user": "눈 파랗게", "calls": [{"name": "eye_on",
                    "arguments": {"r": 0, "g": 0, "b": 255}}], "result": "완료",
                    "pibo": "파란 눈이 됐어요."}, ensure_ascii=False), rng)
    got = list(targets(tok, ex, 4096))
    assert None not in got and len(got) == 2
    texts = ["".join(chr(i) for i, lab in zip(ids, labels) if lab != -100)
             for ids, labels in got]
    assert texts[0].endswith("</tool_call><|im_end|>\n") and '"name": "eye_on"' in texts[0]
    assert texts[1].endswith("파란 눈이 됐어요.<|im_end|>\n")
    # 도구 목록과 아이의 말은 읽기만 합니다
    first_ids, first_labels = got[0]
    prompt = "".join(chr(i) for i, lab in zip(first_ids, first_labels) if lab == -100)
    assert "<tools>" in prompt and "눈 파랗게" in prompt


def test_multi_turn_chat_learns_every_answer():
    from prompts import to_example
    from train import targets

    ex = to_example({"kind": "chat"}, json.dumps({"turns": [
        {"user": "안녕", "pibo": "안녕하세요, 반가워요."},
        {"user": "뭐 해?", "pibo": "친구를 기다리고 있었어요."}]}, ensure_ascii=False),
        random.Random(1))
    got = list(targets(FakeTok(), ex, 4096))
    assert len(got) == 2 and None not in got


def test_own_dialogues_are_read(tmp_path):
    from train import read_extra

    f = tmp_path / "extra.txt"
    f.write_text("# 설명\n아이: 이름이 뭐야?\n파이보: 저는 파이보예요.\n\n# 아이: 무시\n"
                 "아이: 몇 살이야?\n파이보: 저는 로봇이라 나이가 없어요.\n", encoding="utf-8")
    got = read_extra(f)
    assert len(got) == 2 and got[1]["messages"][-1]["content"] == "저는 로봇이라 나이가 없어요."


def test_tool_calls_are_read_from_the_model_text():
    from evaluate import parse_calls, say

    text = '<tool_call>\n{"name": "set_motion", "arguments": {"name": "dance1"}}\n</tool_call>'
    assert parse_calls(text) == [{"name": "set_motion", "arguments": {"name": "dance1"}}]
    assert say("<think>\n\n</think>\n\n좋아요!") == "좋아요!"


def test_the_robot_runs_the_tool_then_answers(monkeypatch):
    import pibo_chat

    # 잘못된 호출은 로봇에 닿기 전에 막힙니다
    assert pibo_chat.run_tool("set_motion", {"name": "moonwalk"}).startswith("실패")
    replies = iter([
        {"content": "", "tool_calls": [{"id": "1", "function": {
            "name": "set_motion", "arguments": '{"name": "dance1"}'}}]},
        {"content": "춤췄어요!"},
    ])
    seen = []
    monkeypatch.setattr(pibo_chat, "ask", lambda url, msgs, tools: next(replies))
    monkeypatch.setattr(pibo_chat, "run_tool", lambda n, a: seen.append((n, a)) or "완료")
    history = []
    assert pibo_chat.turn("x", history, "춤춰", True) == "춤췄어요!"
    assert seen == [("set_motion", {"name": "dance1"})]
    assert len(TOOLS) == 6


def test_gguf_template_never_thinks():
    from export import THINK_IF, force_no_think

    jinja2 = pytest.importorskip("jinja2")
    original = (ROOT / "templates" / "qwen3.jinja").read_text()
    fixed = force_no_think(original)
    assert THINK_IF not in fixed and force_no_think(fixed) == fixed
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "안녕"}]
    # llama-server는 따로 말하지 않으면 생각하는 쪽(enable_thinking 참)으로 펼칩니다
    for kwargs in ({}, {"enable_thinking": True}, {"enable_thinking": False}):
        out = env.from_string(fixed).render(messages=msgs, tools=TOOLS,
                                            add_generation_prompt=True, **kwargs)
        assert out.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # 학습 때 펼친 글은 그대로
    done = msgs + [{"role": "assistant", "content": "안녕하세요."}]
    for src in (original, fixed):
        assert env.from_string(src).render(messages=done, tools=TOOLS) == \
            env.from_string(original).render(messages=done, tools=TOOLS)


def test_judge_runs_again_for_new_answers(tmp_path, monkeypatch):
    import run_all

    monkeypatch.setattr(run_all, "HERE", tmp_path)
    (tmp_path / "results").mkdir()
    for name in ("answers-a", "scores-a", "answers-b"):
        (tmp_path / "results" / f"{name}.json").write_text("{}")
    assert run_all.unscored() == ["b"]


def test_chat_window_starts_with_the_child(monkeypatch):
    import pibo_chat

    sent = []
    monkeypatch.setattr(pibo_chat, "ask",
                        lambda url, msgs, tools: sent.append(msgs) or {"content": "네."})
    history = []
    for text in ("하나", "둘", "셋", "넷"):
        pibo_chat.turn("x", history, text, False)
    assert all(m[1]["role"] == "user" for m in sent)
    monkeypatch.setattr(pibo_chat, "FAKE", True)
    assert pibo_chat.run_tool("get_weather", {"region": "부산"}) == "부산 맑음 18도"
    assert pibo_chat.run_tool("set_motion", {"name": "moonwalk"}).startswith("실패")


def test_gguf_answers_keep_tool_calls_in_the_same_form(tmp_path, monkeypatch):
    import contextlib

    import evaluate
    import export
    from prompts import to_example

    ex = to_example({"kind": "tool", "fn": "set_motion"}, json.dumps(
        {"user": "춤춰 줘", "calls": [{"name": "set_motion", "arguments": {"name": "dance1"}}],
         "result": "완료", "pibo": "춤췄어요."}, ensure_ascii=False), random.Random(0))
    (tmp_path / "heldout.jsonl").write_text(json.dumps(ex, ensure_ascii=False) + "\n")
    monkeypatch.setattr(evaluate, "HELDOUT", tmp_path / "heldout.jsonl")
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    monkeypatch.setattr(export, "server", lambda gguf, port: contextlib.nullcontext("x"))
    monkeypatch.setattr(export, "post", lambda url, body: {"choices": [{"message": {
        "content": "", "tool_calls": [{"function": {
            "name": "set_motion", "arguments": '{"name": "dance1"}'}}]}}]})
    evaluate.gen_gguf("m.gguf", "m")
    answer = json.loads((tmp_path / "answers-m.json").read_text())["answers"][0]
    assert evaluate.parse_calls(answer) == [{"name": "set_motion", "arguments": {"name": "dance1"}}]


def test_status_shows_the_running_step(tmp_path, monkeypatch):
    import fcntl

    import run_all

    monkeypatch.setattr(run_all, "HERE", tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "data-train.done").touch()
    (tmp_path / "logs" / "data-train.log").write_text(
        "\n===== 2026-10-21 10:00:00 시작\n끝남\n===== 2026-10-21 11:00:00 끝 (코드 0)\n")
    (tmp_path / "logs" / "train-qwen3-0.6b.log").write_text(
        "\n===== 2026-10-21 20:00:00 시작\n학습 차례 40000개\n"
        " 17%|█▋ | 442/2560 [16:53<1:21:47]\r 18%|█▊ | 462/2560 [17:40<1:20:40]\r")
    assert run_all.progress(tmp_path / "logs" / "data-train.log") is None
    assert run_all.progress(tmp_path / "logs" / "train-qwen3-0.6b.log").startswith("18%")
    assert not run_all.runner_alive()
    with open(tmp_path / ".lock", "w") as held:   # 실행기가 돌고 있는 것처럼
        fcntl.flock(held, fcntl.LOCK_EX)
        assert run_all.runner_alive()
        run_all.report()
    text = (tmp_path / "REPORT.md").read_text()
    assert "- train-qwen3-0.6b: 진행 중 18%" in text and "- data-train: 끝" in text
