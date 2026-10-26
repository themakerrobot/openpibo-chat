"""GPU 없이 돌아가는 부분: 도구 검사, 선생 답 검사, 학습 정답 자르기, 파이보 대화 흐름."""

import json
import random
import subprocess
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
    monkeypatch.setattr(pibo_chat, "MODEL_TOOLS", True)  # --model-tools: 예전 도구 모델
    monkeypatch.setattr(pibo_chat, "ask", lambda url, msgs, tools, **k: next(replies))
    monkeypatch.setattr(pibo_chat, "run_tool", lambda n, a: seen.append((n, a)) or "완료")
    history = []
    assert pibo_chat.turn("x", history, "춤춰", True) == "춤췄어요!"
    assert seen == [("set_motion", {"name": "dance1"})]
    assert len(TOOLS) == 6


def test_gguf_template_never_thinks():
    from export import force_no_think, thinks

    jinja2 = pytest.importorskip("jinja2")
    original = (ROOT / "templates" / "qwen3.jinja").read_text()
    fixed = force_no_think(original)
    assert thinks(original) and not thinks(fixed) and force_no_think(fixed) == fixed
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
    for name in ("answers-a", "scores-a", "facts-a", "answers-b", "answers-c", "scores-c"):
        (tmp_path / "results" / f"{name}.json").write_text("{}")
    assert run_all.unscored() == ["b", "c"]       # c는 사실 검사가 아직


def test_chat_window_starts_with_the_child(monkeypatch):
    import pibo_chat

    sent = []
    monkeypatch.setattr(pibo_chat, "ask",
                        lambda url, msgs, tools, **k: sent.append(msgs) or {"content": "네."})
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
    monkeypatch.setattr(export, "server", lambda gguf, port, kv=None: contextlib.nullcontext("x"))
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


def test_kd_loss_is_ce_when_the_teacher_is_sure():
    torch = pytest.importorskip("torch")
    from train import kd_loss

    torch.manual_seed(0)
    logits = torch.randn(1, 4, 10)
    labels = torch.tensor([[-100, 3, 7, -100]])
    kd_ids = torch.full((1, 4, 3), -1)
    kd_lp = torch.zeros((1, 4, 3))
    ce, n = kd_loss(logits, labels, kd_ids, kd_lp, 0.5)   # 선생이 없으면 CE만
    assert n == 2
    for j in (1, 2):   # 선생이 정답에 확률 1을 주면 KL = CE
        kd_ids[0, j] = torch.tensor([labels[0, j], 0, 1])
        kd_lp[0, j] = torch.tensor([0.0, -1e4, -1e4])
    both, _ = kd_loss(logits, labels, kd_ids, kd_lp, 0.7)
    assert torch.allclose(ce, both, atol=1e-4)
    kd_lp[0, 1] = torch.tensor([-0.1, -2.5, -9.0])          # 선생이 다른 토큰에도 확률을 주면 다름
    assert not torch.allclose(ce, kd_loss(logits, labels, kd_ids, kd_lp, 1.0)[0])


def test_teacher_top_k_is_taken_at_pibos_tokens():
    from distill import top_k

    class Lp:
        def __init__(self, v):
            self.logprob = v

    # vLLM처럼: 맨 앞은 None, 자리마다 {토큰: Logprob} (실제 토큰이 상위 K 밖이면 하나 더)
    plp = [None, {5: Lp(-0.1)}, {9: Lp(-3.0), 2: Lp(-0.2), 4: Lp(-1.0)}, {1: Lp(-0.5)}]
    ids, lp = top_k(plp, [-100, -100, 2, 1], 2)
    assert ids.tolist() == [[2, 4], [1, -1]]
    assert lp[0].tolist() == pytest.approx([-0.2, -1.0], abs=1e-3)


def test_distillation_runs_only_when_switched_on(tmp_path, monkeypatch):
    import run_all

    monkeypatch.setattr(run_all, "HERE", tmp_path)
    (tmp_path / "state").mkdir()
    names = [n for n, _, _ in run_all.steps()]
    assert "kd-teacher" not in names and names[-3:] == ["judge", "judge-school", "judge-english"]
    (tmp_path / "state" / "kd.on").touch()
    steps = {n: (cmd, needs) for n, cmd, needs in run_all.steps()}
    cmd, needs = steps["train-qwen3-0.6b-kd"]
    assert needs == ["data-train", "kd-teacher"] and cmd[cmd.index("--kd") + 1] == "data/kd"
    assert "eval-qwen3-0.6b-kd-Q8_0" in steps and "eval-qwen3-0.6b-kd-원본" not in steps
    assert "--kd" not in steps["train-qwen3-0.6b"][0]
    # 거른 데이터로도: 선생 확률은 원래 데이터로 한 번만 구합니다
    cmd, needs = steps["train-qwen3-0.6b-v2-kd"]
    assert needs == ["data-clean", "kd-teacher"] and "data/train-clean.jsonl" in cmd
    assert [n for n in steps if n == "kd-teacher"] == ["kd-teacher"]


def test_dance_number_and_extra_eye_colour_do_not_make_a_call_wrong():
    from evaluate import tool_match

    want = [{"name": "set_motion", "arguments": {"name": "dance5"}}]
    dance4 = {"name": "set_motion", "arguments": {"name": "dance4"}}
    blue = {"name": "eye_on", "arguments": {"r": 0, "g": 0, "b": 255}}
    assert tool_match([dance4], want) == (True, False)
    assert tool_match([dance4, blue], want) == (True, True)      # 맞음, 덤으로 눈 색
    assert tool_match([{"name": "set_motion", "arguments": {"name": "bow"}}], want)[0] is False
    red = [{"name": "eye_on", "arguments": {"r": 255, "g": 0, "b": 0}}]
    assert tool_match([blue], red) == (False, False)
    assert tool_match([blue], red, names_only=True) == (True, False)   # 색을 말 안 한 문제
    assert tool_match([], red) == (False, False)


def test_tool_examples_are_judged_and_marked(tmp_path, monkeypatch):
    import evaluate
    from filter_tools import prompt, tool_examples, verdict
    from prompts import to_example

    rng = random.Random(0)
    lines = [json.dumps(to_example({"kind": "chat"}, json.dumps({"turns": [
        {"user": "안녕", "pibo": "안녕하세요."}]}, ensure_ascii=False), rng), ensure_ascii=False)]
    for user, call in (("내가 화났어", {"name": "find_objects", "arguments": {}}),
                       ("춤춰 줘", {"name": "set_motion", "arguments": {"name": "dance2"}})):
        lines.append(json.dumps(to_example({"kind": "tool", "fn": call["name"]}, json.dumps(
            {"user": user, "calls": [call], "result": "완료", "pibo": "했어요."},
            ensure_ascii=False), rng), ensure_ascii=False))
    got = list(tool_examples(lines))
    assert [(i, u) for i, u, _ in got] == [(1, "내가 화났어"), (2, "춤춰 줘")]
    assert "find_objects({})" in prompt(got[0][1], got[0][2])
    assert verdict('```json\n{"natural": false, "args_from_words": true}\n```') == \
        {"natural": False, "args_from_words": True}
    assert verdict("모르겠어요") == {"natural": None, "args_from_words": None}
    # 채점: 억지(1번)는 빼고, 이름만 볼 것(2번)은 이름만
    (tmp_path / "heldout.jsonl").write_text("\n".join(lines) + "\n")
    (tmp_path / "marks.json").write_text(json.dumps({"bad": [1], "names_only": [2]}))
    (tmp_path / "answers-m.json").write_text(json.dumps({"no_tools": False, "answers": [
        "안녕하세요.", "",
        '<tool_call>\n{"name": "set_motion", "arguments": {"name": "bow"}}\n</tool_call>']}))
    monkeypatch.setattr(evaluate, "HELDOUT", tmp_path / "heldout.jsonl")
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    monkeypatch.setattr(evaluate, "MARKS", tmp_path / "marks.json")
    row = next(r for r in evaluate.report().splitlines() if r.startswith("| m |"))
    assert row.split(" | ")[8] == "100%"     # 남은 도구 문제 하나, 이름만 맞으면 정답


def test_qwen35_template_never_thinks_and_its_tool_calls_are_read():
    from evaluate import parse_calls
    from export import force_no_think, thinks

    jinja2 = pytest.importorskip("jinja2")
    src = ("{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n"
           "    {%- if enable_thinking is defined and enable_thinking is true %}\n"
           "        {{- '<think>\\n' }}\n    {%- else %}\n"
           "        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}")
    fixed = force_no_think(src)
    assert not thinks(fixed) and force_no_think(fixed) == fixed
    out = jinja2.Environment().from_string(fixed).render(add_generation_prompt=True,
                                                         enable_thinking=True)
    assert out.endswith("assistant\n<think>\n\n</think>\n\n")
    text = ("<tool_call>\n<function=eye_on>\n<parameter=r>\n255\n</parameter>\n"
            "<parameter=g>\n0\n</parameter>\n<parameter=b>\n0\n</parameter>\n</function>\n"
            "</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=region>\n부산\n"
            "</parameter>\n</function>\n</tool_call>")
    assert parse_calls(text) == [{"name": "eye_on", "arguments": {"r": 255, "g": 0, "b": 0}},
                                 {"name": "get_weather", "arguments": {"region": "부산"}}]


def test_school_students_come_first_on_fact_checked_school_data(tmp_path, monkeypatch):
    import run_all

    monkeypatch.setattr(run_all, "HERE", tmp_path)
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "kd.on").touch()
    names = [n for n, _, _ in run_all.steps()]
    steps = {n: (cmd, needs) for n, cmd, needs in run_all.steps()}
    assert steps["data-school"][0][:3] == ["make_data.py", "--audience", "school"]
    assert steps["data-school-clean"][1] == ["data-school", "data-clean"]
    for st in ("qwen3-0.6b-school", "qwen3.5-0.8b-school"):
        cmd, needs = steps[f"train-{st}"]
        assert needs == ["data-school-clean"] and "data/train-school.jsonl" in cmd
        assert "--kd" not in cmd and f"train-{st}-kd" not in steps
        cmd, needs = steps[f"eval-{st}-small-Q4_K_M"]
        assert cmd[-2:] == ["--set", "school"] and "data-school-heldout" in needs
        # 학교 학생이 어린이 데이터의 남은 단계보다 먼저
        assert names.index(f"train-{st}") < names.index("train-qwen3-0.6b")
        assert names.index(f"eval-{st}-small-Q4_K_M") < names.index("eval-qwen3-0.6b-Q4_K_M")
    assert "Qwen/Qwen3.5-0.8B" in steps["train-qwen3.5-0.8b-school"][0]
    # 원본 모델은 두 묶음 모두에서 (어린이 쪽 단계 이름은 처음 그대로)
    assert "eval-gemma3-1b" in steps and "eval-qwen3.5-0.8b-원본" in steps
    cmd, needs = steps["eval-school-gemma3-1b"]
    assert "--set" in cmd and "--no-tools" in cmd and needs == ["data-heldout", "data-school-heldout"]
    assert steps["judge-school"][0] == ["evaluate.py", "judge", "--set", "school"]


def test_chat_examples_are_fact_checked_and_rewritten():
    from filter_chat import chat_examples, decide, prompt, rewrite
    from prompts import after_think, to_example

    rng = random.Random(0)
    chat = json.dumps(to_example({"kind": "chat"}, json.dumps({"turns": [
        {"user": "공룡은 왜 없어졌어?", "pibo": "아직 아무도 몰라요."},
        {"user": "언제 살았어?", "pibo": "아주 오래전에 살았어요."}]}, ensure_ascii=False), rng),
        ensure_ascii=False)
    tool = json.dumps(to_example({"kind": "tool", "fn": "eye_off"}, json.dumps(
        {"user": "눈 꺼 줘", "calls": [{"name": "eye_off", "arguments": {}}], "result": "완료",
         "pibo": "껐어요."}, ensure_ascii=False), rng), ensure_ascii=False)
    lines = [chat, tool, chat]
    items = list(chat_examples(lines))
    assert [i for i, _, _, _ in items] == [0, 2]
    _, ex, users, pibos = items[0]
    text = prompt(ex, users, pibos)
    assert "아이: 언제 살았어?\n파이보 2: 아주 오래전에 살았어요." in text and "말 2개" in text
    # 생각 뒤의 답만 읽고, 생각이 잘렸으면 못 읽음
    assert after_think("<think>\n{\"x\": 1} 고민\n</think>\n\n{\"ok\": true}") == "\n\n{\"ok\": true}"
    assert after_think("<think>\n끝없이 생각") == ""
    fixed = ["큰 소행성이 떨어져서 사라졌어요.", "아주 오래전에 살았어요."]
    good = "<think>음</think>" + json.dumps({"ok": False, "problem": "얼버무림", "fixed": fixed},
                                            ensure_ascii=False)
    assert decide(good, pibos)["verdict"] == "fixed"
    assert decide('<think>음</think>{"ok": true}', pibos) == \
        {"verdict": "ok", "after": pibos, "problem": None}
    assert decide("<think>끝없이", pibos)["verdict"] == "drop"
    assert decide(json.dumps({"ok": False, "fixed": fixed[:1]}), pibos)["verdict"] == "drop"
    assert decide(json.dumps({"ok": False, "fixed": ["좋아요 😀", "네."]}), pibos)["verdict"] == "drop"
    out = rewrite(lines, {0: decide(good, pibos), 2: decide("모름", pibos)})
    assert len(out) == 2 and out[1] == tool       # 2번 대화는 빠지고 도구 예시는 그대로
    said = [m["content"] for m in json.loads(out[0])["messages"] if m["role"] == "assistant"]
    assert said == fixed
    # 다른 파일의 도구 예시를 붙이면 시스템 글을 지금 것으로
    old = json.loads(tool)
    old["messages"][0]["content"] = "옛 시스템 글"
    out = rewrite([chat], {0: decide(good, pibos)}, [chat, json.dumps(old, ensure_ascii=False)])
    assert len(out) == 2 and json.loads(out[1])["messages"][0]["content"] == SYSTEM


def test_school_requests_cover_every_grade_and_length():
    from filter_chat import decide, prompt
    from prompts import GRADES, max_len, requests, style_for, to_example

    reqs = requests(400, 0, seed=1, audience="school")
    grades = [r["grade"] for r in reqs]
    assert all(80 <= grades.count(g) <= 120 for g in GRADES)   # 학년 묶음 고르게
    curious = sum(r["topic"] not in __import__("prompts").TALK for r in reqs)
    assert 30 <= curious <= 90                                 # 호기심 질문 15% 안팎
    r = next(r for r in reqs if r["grade"] == "고")
    assert "고등학교" in r["prompt"] and "120자" in r["prompt"]
    assert "70자" in next(r for r in reqs if r["grade"] == "초저")["prompt"]
    long = "수행평가 때문에 걱정이 많군요. 오늘 할 일을 작게 나눠서 하나씩 해 보면 훨씬 덜 막막할 거예요. " \
           "어떤 과목이 제일 걱정돼요? 오늘 저녁에 한 가지만 먼저 끝내 보고, 내일 다시 같이 계획을 세워 봐요. 할 수 있어요!"
    assert 120 < len(long) <= 160
    reply = json.dumps({"turns": [{"user": "수행평가 망할 듯", "pibo": long}]}, ensure_ascii=False)
    ex = to_example(r, reply, random.Random(0))
    assert ex["grade"] == "고" and max_len(ex) == 160
    assert to_example(dict(r, grade="초저"), reply, random.Random(0)) is None   # 초등은 120자까지
    assert "고등학생에게" in style_for(ex) and "학생: 수행평가 망할 듯" in prompt(
        ex, ["수행평가 망할 듯"], [long])
    fixed = json.dumps({"ok": False, "problem": "x", "fixed": [long]}, ensure_ascii=False)
    assert decide(fixed, ["다른 말"], max_len(ex))["verdict"] == "fixed"
    assert decide(fixed, ["다른 말"])["verdict"] == "drop"


def test_school_questions_replace_chat_in_place_and_keep_tool_marks(tmp_path, monkeypatch):
    import evaluate
    from prompts import to_example
    from tools import SYSTEM_KIDS

    rng = random.Random(0)

    def chat(user, grade=None):
        ex = to_example({"kind": "chat", **({"grade": grade} if grade else {})}, json.dumps(
            {"turns": [{"user": user, "pibo": "네."}]}, ensure_ascii=False), rng)
        return json.dumps(ex, ensure_ascii=False)

    tool = json.loads(json.dumps(to_example({"kind": "tool", "fn": "eye_off"}, json.dumps(
        {"user": "눈 꺼 줘", "calls": [{"name": "eye_off", "arguments": {}}], "result": "완료",
         "pibo": "껐어요."}, ensure_ascii=False), rng), ensure_ascii=False))
    tool["messages"][0]["content"] = SYSTEM_KIDS
    (tmp_path / "heldout.jsonl").write_text("\n".join(
        [chat("공룡 어디 갔어?"), json.dumps(tool, ensure_ascii=False), chat("안녕")]) + "\n")
    (tmp_path / "school.jsonl").write_text(chat("시험 망했어", "중") + "\n" + chat("배고파", "초저") + "\n")
    monkeypatch.setattr(evaluate, "HELDOUT", tmp_path / "heldout.jsonl")
    monkeypatch.setitem(evaluate.SETS, "school", (tmp_path / "rs", tmp_path / "school.jsonl"))
    evaluate.use_set("school")
    try:
        qs = evaluate.questions()
    finally:
        evaluate.use_set("kids")
    assert [q["messages"][-1]["content"] for q in qs] == ["시험 망했어", "눈 꺼 줘", "배고파"]
    assert [q["grade"] for q in qs] == ["중", None, "초저"]
    assert all(q["messages"][0]["content"] == SYSTEM for q in qs)
    assert "학생: {user}" in evaluate.for_school(evaluate.JUDGE)
    assert "학생들은" in evaluate.for_school(evaluate.FACT)
    assert evaluate.RESULTS == Path("results") and evaluate.SCHOOL_CHAT is None


def test_judge_counts_facts():
    import evaluate

    assert evaluate.fact_marks(['<think>\n음\n</think>\n{"wrong": true, "dodged": false}',
                                "<think>\n잘림"]) == \
        [{"wrong": True, "dodged": False}, {"wrong": None, "dodged": None}]


def test_detected_names_are_said_in_korean():
    from pibo_chat import korean_names

    assert korean_names(["cup", "person", "cup", "Teddy_Bear", "robot"]) == "robot, 곰 인형, 사람, 컵"
    assert korean_names(["tvmonitor", "sports ball"]) == "공, 텔레비전"
    assert korean_names([]) == korean_names([None, " "]) == "아무것도 안 보임"


def test_vocab_pruning_keeps_korean_and_used_tokens_with_their_merge_parents():
    from prune_vocab import byte_decoder, keep_set, korean

    enc = {b: c for c, b in byte_decoder().items()}

    def bl(text):                                  # 글 → 바이트 수준 BPE 글자
        return "".join(enc[b] for b in text.encode())

    assert korean(bl("안녕"), byte_decoder()) and korean(bl(" 안"), byte_decoder())
    assert korean(bl("안")[:2], byte_decoder())             # 한글 글자의 바이트 조각
    assert korean(bl('"') + bl("안")[:2], byte_decoder())    # 앞에 기호 한 바이트
    assert not korean(bl("game"), byte_decoder()) and not korean(bl("😀"), byte_decoder())
    assert not korean(bl("我"), byte_decoder())
    singles = [enc[b] for b in range(256)]
    an = bl("안")
    vocab = {t: i for i, t in enumerate(singles + [an[:2], an, "ga", "gam", "game", bl("😀")[:2],
                                                   bl("😀"), "zz"])}
    merges = [f"{an[0]} {an[1]}", f"{an[:2]} {an[2]}", "g a", "ga m", "gam e",
              f"{bl('😀')[0]} {bl('😀')[1]}", f"{bl('😀')[:2]} {bl('😀')[2:]}", "z z"]
    tj = {"model": {"vocab": vocab, "merges": merges}}
    keep = keep_set(tj, used={"game"}, ascii_merges=0)
    assert {an, an[:2], "game", "gam", "ga"} <= keep           # 쓴 토큰의 재료까지
    assert bl("😀") not in keep and "zz" not in keep
    assert "zz" in keep_set(tj, used=set(), ascii_merges=len(merges))   # 흔한 영어 조각


def test_school_students_are_also_built_with_a_smaller_vocab(tmp_path, monkeypatch):
    import run_all

    monkeypatch.setattr(run_all, "HERE", tmp_path)
    steps = {n: (cmd, needs) for n, cmd, needs in run_all.steps()}
    cmd, needs = steps["small-qwen3.5-0.8b-school"]
    assert cmd[0] == "prune_vocab.py" and "runs/qwen3.5-0.8b-school/small" in cmd
    assert cmd[cmd.index("--check") + 1] == "data/heldout-school.jsonl"
    assert needs[0] == "train-qwen3.5-0.8b-school"
    cmd, needs = steps["eval-qwen3-0.6b-school-small-Q4_K_M"]
    assert "out/pibo-qwen3-0.6b-school-small-Q4_K_M.gguf" in cmd and cmd[-2:] == ["--set", "school"]
    assert "gguf-qwen3-0.6b-small-Q4_K_M" not in steps          # 어린이 데이터 모델은 그대로
    cmd = steps["gguf-qwen3-0.6b-school-small-Q4_K_M"][0]
    assert cmd[cmd.index("--imatrix") + 1] == "data/train-school.jsonl"
    assert "gguf-qwen3-0.6b-school-small-Q6_K" not in steps   # 줄인 Q4_K_M만 만듭니다
    assert "gguf-qwen3-0.6b-school-Q4_K_M" not in steps
    cmd, needs = steps["eval-qwen3-0.6b-school-small-Q4_K_M-kv8"]
    assert cmd[cmd.index("--kv") + 1] == "q8_0" and needs[0] == "gguf-qwen3-0.6b-school-small-Q4_K_M"
    assert "eval-qwen3.5-0.8b-school-small-Q4_K_M-kv8" not in steps   # KV 캐시가 원래 작음


def test_orphaned_vllm_engines_are_stopped_before_a_step(monkeypatch):
    import run_all

    smi = "1099455, VLLM::EngineCore, 31100\n2000, VLLM::EngineCore, 900\n3000, python, 500\n"
    monkeypatch.setattr(run_all.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, smi, ""))
    monkeypatch.setattr(run_all, "parent_pid", {1099455: 1, 2000: 4242, 3000: 1}.get)
    monkeypatch.setattr(run_all.time, "sleep", lambda s: None)
    sent = []
    monkeypatch.setattr(run_all.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert run_all.kill_orphan_engines() == [1099455]     # 부모가 살아 있는 엔진과 다른 것은 그대로
    assert [p for p, _ in sent] == [1099455, 1099455]


def test_teacher_is_not_marked_failed_when_the_gpu_is_busy(tmp_path, monkeypatch):
    import make_data

    monkeypatch.setattr(make_data, "CHOSEN", tmp_path / "teacher.txt")
    monkeypatch.setattr(make_data, "gpu_free_fraction", lambda: 0.01)
    monkeypatch.setitem(sys.modules, "vllm", type(sys)("vllm"))
    sys.modules["vllm"].LLM = None
    with pytest.raises(SystemExit, match="비어 있어"):
        make_data.load_teacher()
    assert not (tmp_path / "teacher-failed.txt").exists()


def test_rejected_teacher_replies_say_why():
    from prompts import build_example

    rng = random.Random(0)
    req = {"kind": "chat", "grade": "초저"}

    def reply(pibo):
        return json.dumps({"turns": [{"user": "안녕", "pibo": pibo}]}, ensure_ascii=False)

    assert build_example(req, "글", rng) == (None, "JSON을 못 읽음")
    assert build_example(req, reply("좋아요~"), rng) == (None, "파이보 말에 기호 '~'")
    assert build_example(req, reply("가" * 121), rng) == (None, "파이보 말이 120자 넘음")
    ex, why = build_example(req, reply("안녕하세요."), rng)
    assert why is None and ex["grade"] == "초저"


def test_teacher_turns_written_one_side_at_a_time_are_paired():
    from prompts import build_example

    rng = random.Random(0)
    req = {"kind": "chat", "grade": "중"}
    split = ('{"turns": [{"user": "숙제 많아서 힘들어..."}, {"pibo": "많이 힘들겠어요. 나눠서 해 봐요."}, '
             '{"user": "응 해볼게"}, {"pibo": "좋아요, 응원할게요."}, {"user": "그나저나')
    ex, why = build_example(req, split, rng)
    assert why is None
    assert [m["content"] for m in ex["messages"][1:]] == [
        "숙제 많아서 힘들어...", "많이 힘들겠어요. 나눠서 해 봐요.", "응 해볼게", "좋아요, 응원할게요."]
    broken = ('{"turns": [{"user": "인공지능은 뭐야? \\"AI\\" 말이야"}, "pibo": "컴퓨터가 사람처럼 '
              '생각하는 거예요."}]}')
    ex, why = build_example(req, broken, rng)
    assert why is None and ex["messages"][1]["content"] == '인공지능은 뭐야? "AI" 말이야'
    assert build_example(req, '{"turns": [{"pibo": "안녕"}, {"user": "응"}]}', rng)[0] is None


def test_school_requests_are_varied_and_fit_the_grade():
    import prompts

    reqs = prompts.requests(3000, 0, seed=4, audience="school")
    assert not any(r["topic"] in prompts.HIGH and r["grade"] != "고" for r in reqs)
    assert not any(r["topic"] in prompts.OLDER and r["grade"] == "초저" for r in reqs)
    assert len({r["scene"] for r in reqs}) == len(prompts.SCENES)
    assert len({(r["topic"], r["scene"], r["opener"]) for r in reqs}) > 2900   # 거의 안 겹침
    assert all("장면:" in r["prompt"] for r in reqs)


def test_tool_examples_can_be_repeated_to_keep_their_share():
    from filter_chat import rewrite
    from prompts import to_example

    rng = random.Random(0)
    tool = json.dumps(to_example({"kind": "tool", "fn": "eye_off"}, json.dumps(
        {"user": "눈 꺼 줘", "calls": [{"name": "eye_off", "arguments": {}}], "result": "완료",
         "pibo": "껐어요."}, ensure_ascii=False), rng), ensure_ascii=False)
    assert len(rewrite([], {}, [tool], repeat=2)) == 2


def test_banmal_left_in_pibos_words_is_found():
    from filter_chat import BANMAL

    rude = ["문어는 팔이 여덟 개예요. 재미있지?", "아 그런 일이 생겼구나. 힘들었겠네.",
            "안녕하세요! 네가 인사하면 반가워요.", "먹을 거예요. 넌 뭐야?",
            "레고는 많은 사람들이 좋아해요. 너는 어떤 레고를 좋아해?",
            "파이보도 너처럼 배워서 도와줄 수 있어요.", "힘들겠네. 그래도 괜찮아요.",
            "같이 해 볼까?"]
    polite = ["멋진 소리가 나는 거 좋아요.", "정말 잘했어요!", "제가 도와드릴게요.",
              "너무 재미있겠네요!", "같이 해 볼까요?", "그 마음 이해돼요."]
    assert all(BANMAL.search(s) for s in rude)
    assert not any(BANMAL.search(s) for s in polite)


def test_check_log_from_other_data_is_not_reused():
    from filter_chat import matches

    r = {"line": 0, "users": ["공룡 얘기"], "before": ["오래전에 살았어요."], "verdict": "ok"}
    assert matches(r, ["공룡 얘기"], ["오래전에 살았어요."])
    assert not matches(r, ["시험 망했어", "응"], ["힘들겠어요.", "응원할게요."])


def test_tool_calls_stay_in_the_remembered_conversation(monkeypatch):
    import pibo_chat

    replies = iter([
        {"content": "", "tool_calls": [{"id": "1", "function": {
            "name": "set_motion", "arguments": '{"name": "forward1"}'}}]},
        {"content": "앞으로 갔어요."},
        {"content": "네."}, {"content": "네."}, {"content": "네."},
    ])
    sent = []
    monkeypatch.setattr(pibo_chat, "ask", lambda url, msgs, tools, **k: sent.append(list(msgs)) or next(replies))
    monkeypatch.setattr(pibo_chat, "run_tool", lambda n, a: "완료")
    monkeypatch.setattr(pibo_chat, "MODEL_TOOLS", True)  # --model-tools
    monkeypatch.setattr(pibo_chat, "HISTORY", 3)       # --history 3
    history = []
    pibo_chat.turn("x", history, "앞으로 가", True)
    pibo_chat.turn("x", history, "뒤로 가", True)
    roles = [m["role"] for m in sent[2]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert sent[2][2]["tool_calls"][0]["function"]["name"] == "set_motion"
    assert sent[2][2]["tool_calls"][0]["type"] == "function"   # 없으면 새 llama-server가 500
    for text in ("셋", "넷"):
        pibo_chat.turn("x", history, text, True)
    assert [m["role"] for m in sent[-1]][:2] == ["system", "user"]     # 짝이 안 잘림
    assert sum(m["role"] == "user" for m in sent[-1]) == pibo_chat.HISTORY


def test_child_can_switch_tools_off_and_on():
    from pibo_chat import tool_switch

    assert tool_switch("대화만 하자", True) is False
    assert tool_switch("이제 움직이지 마", True) is False
    assert tool_switch("도구는 쓰지 마", True) is False
    assert tool_switch("도구도 써서 하자 이제부터", False) is True
    assert tool_switch("이제 움직여도 돼", False) is True
    assert tool_switch("도구 다시 쓰자", False) is True and tool_switch("도구 다시 사용해", False)
    assert tool_switch("앞으로 가", False) is False      # 끈 동안은 명령도 대화로
    assert tool_switch("앞으로 가", True) is True


def test_tools_called_while_paused_are_not_run(monkeypatch):
    import pibo_chat

    monkeypatch.setattr(pibo_chat, "ask", lambda url, msgs, tools, **k: {"content": "", "tool_calls": [
        {"id": "1", "function": {"name": "set_motion", "arguments": '{"name": "forward1"}'}}]})
    ran = []
    monkeypatch.setattr(pibo_chat, "run_tool", lambda n, a: ran.append(n) or "완료")
    history = []
    assert pibo_chat.turn("x", history, "앞으로 가", True, allowed=False) == pibo_chat.PAUSED
    assert ran == [] and history[-1] == {"role": "assistant", "content": pibo_chat.PAUSED}


def test_tool_decision_reads_streamed_tool_calls_and_stops_early_on_words(monkeypatch):
    import io
    import pibo_chat

    def sse(*deltas):
        return io.BytesIO("".join(f"data: {json.dumps({'choices': [{'delta': d}]}, ensure_ascii=False)}\n\n"
                                  for d in deltas).encode() + b"data: [DONE]\n\n")

    calls = sse({"role": "assistant", "content": None},
                {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "set_motion", "arguments": ""}}]},
                {"tool_calls": [{"index": 0, "function": {"arguments": '{"name": '}}]},
                {"tool_calls": [{"index": 0, "function": {"arguments": '"forward1"}'}}]})
    monkeypatch.setattr(pibo_chat.urllib.request, "urlopen", lambda req, timeout: calls)
    got = pibo_chat.first_move("http://x", "앞으로 가")
    assert got == [{"id": "a", "type": "function",
                    "function": {"name": "set_motion", "arguments": '{"name": "forward1"}'}}]
    words = sse({"content": "공룡"}, {"content": "은"})
    monkeypatch.setattr(pibo_chat.urllib.request, "urlopen", lambda req, timeout: words)
    assert pibo_chat.first_move("http://x", "공룡 얘기") is None


def test_stop_is_run_without_asking_the_model(monkeypatch):
    import pibo_chat

    ran = []
    monkeypatch.setattr(pibo_chat, "run_tool", lambda n, a: ran.append((n, a)) or "완료")
    monkeypatch.setattr(pibo_chat, "ask", lambda *a, **k: pytest.fail("모델에 묻지 않아야 함"))
    monkeypatch.setattr(pibo_chat, "first_move", lambda *a, **k: pytest.fail("모델에 묻지 않아야 함"))
    assert pibo_chat.turn("x", [], "멈춰!", True) == "네, 멈출게요."
    assert ran == [("set_motion", {"name": "stop"})]


# ------------------------------------------------------------- 초·중·고 도구 보강

def mixed_reply(req: dict, **change) -> str:
    """계획(calls_at)대로 쓴 대화 중 도구 답."""
    calls = {"set_motion": {"name": "stop"}, "eye_on": {"r": 0, "g": 0, "b": 255},
             "get_weather": {"region": "서울"}}
    turns = []
    for k in range(req["turns"]):
        fn = req["calls_at"].get(str(k))
        if fn:
            turns.append({"user": f"부탁 {k}", "calls": [{"name": fn, "arguments": calls.get(fn, {})}],
                          "result": "완료", "pibo": f"했어요 {k}."})
        else:
            turns.append({"user": f"이야기 {k}", "pibo": f"그렇군요 {k}."})
    for k, v in change.items():
        turns[int(k[1:])].update(v)
    return json.dumps({"turns": turns}, ensure_ascii=False)


def test_school_tool_requests_are_weighted_varied_and_ask_for_one_tool():
    import prompts

    reqs = prompts.requests(0, 3000, seed=5, audience="school")
    fns = [r["fn"] for r in reqs]
    assert fns.count("set_motion") > fns.count("eye_on") > fns.count("get_battery")
    assert all(f in fns for f in prompts.BY_NAME)
    assert all(r["grade"] in prompts.GRADES and "학생" in r["asker"] for r in reqs)
    assert sum(prompts.STOP_HINT in r["prompt"] for r in reqs) > 100   # 멈추기를 여러 말로
    assert len({a for r in reqs for a in prompts.ASKS if a in r["prompt"]}) == len(prompts.ASKS)
    assert all(prompts.ONLY_ASKED in r["prompt"] for r in reqs)
    # 실패 결과는 날씨·물건 찾기에서만, 8% 안팎
    assert {r["fn"] for r in reqs if r["fail"]} == {"get_weather", "find_objects"}
    assert 30 < sum(r["fail"] for r in reqs) < 100
    r = next(r for r in reqs if r["fn"] == "set_motion")
    one = json.dumps({"user": "그만 좀 움직여", "calls": [{"name": "set_motion", "arguments":
                      {"name": "stop"}}], "result": "완료", "pibo": "네, 멈췄어요."}, ensure_ascii=False)
    ex, why = prompts.build_example(r, one, random.Random(0))
    assert why is None and ex["grade"] == r["grade"] and ex["kind"] == "tool"
    extra = json.loads(one)
    extra["calls"].append({"name": "eye_on", "arguments": {"r": 255, "g": 0, "b": 0}})
    assert prompts.build_example(r, json.dumps(extra), random.Random(0)) == \
        (None, "청하지 않은 도구를 덤으로 부름")
    # 어린이 도구 예시는 예전처럼 둘까지
    kid = prompts.build_example({"kind": "tool", "fn": "set_motion"}, json.dumps(extra),
                                random.Random(0))[0]
    assert kid is not None and "grade" not in kid


def test_mixed_conversations_call_a_tool_in_the_middle_and_are_learned():
    import prompts
    from train import targets

    reqs = prompts.requests(0, 0, seed=6, audience="school", n_mixed=500)
    assert all(r["kind"] == "mixed" and "0" not in r["calls_at"] for r in reqs)  # 첫 차례는 이야기
    assert any(len(r["calls_at"]) == 2 for r in reqs)
    assert all(len(set(r["calls_at"].values())) == len(r["calls_at"]) for r in reqs)
    r = next(r for r in reqs if len(r["calls_at"]) == 1 and r["turns"] == 4
             and r["calls_at"].get("1") in ("set_motion", "eye_on", "get_weather"))
    ex, why = prompts.build_example(r, mixed_reply(r), random.Random(0))
    assert why is None and ex["kind"] == "mixed" and ex["tools"] and ex["grade"] == r["grade"]
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "tool", "assistant",
                     "user", "assistant", "user", "assistant"]
    assert ex["messages"][4]["tool_calls"][0]["function"]["name"] == r["calls_at"]["1"]
    # 계획과 다르면 버림: 도구 차례가 바뀜, 이야기 차례에 도구, 덤으로 눈 색
    assert prompts.build_example(r, mixed_reply(r, t1={"calls": None}), random.Random(0)) == \
        (None, "도구 차례가 계획과 다름")
    assert prompts.build_example(r, mixed_reply(r, t2={"calls": [
        {"name": "eye_off", "arguments": {}}], "result": "완료"}), random.Random(0))[0] is None
    if r["calls_at"]["1"] != "eye_on":
        two = json.loads(mixed_reply(r))
        two["turns"][1]["calls"].append({"name": "eye_on", "arguments": {"r": 1, "g": 2, "b": 3}})
        assert prompts.build_example(r, json.dumps(two), random.Random(0))[1] == \
            "청하지 않은 도구를 덤으로 부름"
    # 파이보의 차례마다 (도구 호출 차례도) 배움
    got = list(targets(FakeTok(), ex, 4096))
    assert None not in got and len(got) == 5


def test_near_examples_are_talk_with_the_tool_list():
    import make_data
    import prompts

    reqs = prompts.requests(0, 0, seed=7, audience="school", n_near=200)
    assert all(r["kind"] == "chat" and r.get("near") for r in reqs)
    assert len({r["topic"] for r in reqs}) == len(prompts.NEAR)
    reply = json.dumps({"turns": [{"user": "안녕 파이보!", "pibo": "안녕하세요, 반가워요!"}]},
                       ensure_ascii=False)
    for seed in range(5):
        ex = prompts.to_example(reqs[0], reply, random.Random(seed))
        assert ex["tools"] and ex["near"] and make_data.slot(ex) == "near"
    assert make_data.slot({"kind": "mixed"}) == "mixed"


def test_tool_check_and_fact_check_see_the_tool_turns_of_mixed_talk():
    import prompts
    from filter_chat import chat_examples, prompt, rewrite
    from filter_tools import tool_examples

    r = next(r for r in prompts.requests(0, 0, seed=6, audience="school", n_mixed=200)
             if r["turns"] == 3 and r["calls_at"] == {"1": "get_weather"})
    line = json.dumps(prompts.build_example(r, mixed_reply(r, t1={"result": "서울 비 12도"}),
                                            random.Random(0))[0], ensure_ascii=False)
    got = list(tool_examples([line]))
    assert got == [(0, "부탁 1", [{"name": "get_weather", "arguments": {"region": "서울"}}])]
    [(i, ex, users, pibos)] = list(chat_examples([line]))
    assert users == ["이야기 0", "부탁 1", "이야기 2"] and pibos == ["그렇군요 0.", "했어요 1.", "그렇군요 2."]
    text = prompt(ex, users, pibos)
    assert "(파이보가 도구 get_weather을 쓰고 받은 결과: 서울 비 12도)\n파이보 2: 했어요 1." in text
    assert "괄호 안은" in text
    chat = prompts.to_example({"kind": "chat", "grade": "중"}, json.dumps(
        {"turns": [{"user": "안녕", "pibo": "안녕하세요."}]}, ensure_ascii=False), random.Random(0))
    plain = prompt(chat, ["안녕"], ["안녕하세요."])
    assert "괄호 안은" not in plain and "학생: 안녕\n파이보 1: 안녕하세요." in plain
    out = rewrite([line], {0: {"verdict": "fixed", "after": ["가", "나", "다"]}})
    msgs = json.loads(out[0])["messages"]
    assert [m["content"] for m in msgs if m["role"] == "assistant"] == ["가", "", "나", "다"]
    assert msgs[4]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_merging_drops_greeting_tools_extra_eye_colour_and_repeats(tmp_path):
    from merge_data import merge
    from prompts import to_example

    rng = random.Random(0)

    def tool(user, *calls):
        return json.dumps(to_example({"kind": "tool", "fn": calls[0]["name"]}, json.dumps(
            {"user": user, "calls": list(calls), "result": "완료", "pibo": "했어요."},
            ensure_ascii=False), rng), ensure_ascii=False)

    greet = {"name": "set_motion", "arguments": {"name": "greeting"}}
    eye = {"name": "eye_on", "arguments": {"r": 0, "g": 0, "b": 255}}
    dance = {"name": "set_motion", "arguments": {"name": "dance1"}}
    good = [tool("인사해 줘", greet), tool("눈 파랗게 해 줘", eye), tool("파이보 안녕! 손 흔들어 줘", greet)]
    bad = [tool("파이보야 안녕~", greet), tool("안녕하세요!", greet, eye), tool("춤춰 줘", dance, eye)]
    chat = json.dumps(to_example({"kind": "chat"}, json.dumps({"turns": [
        {"user": "안녕", "pibo": "안녕하세요."}]}, ensure_ascii=False), rng), ensure_ascii=False)
    (tmp_path / "a.jsonl").write_text("\n".join([chat, *good, good[0], *bad]) + "\n")
    (tmp_path / "b.jsonl").write_text(good[1] + "\n")
    lines, kept, why = merge([str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")])
    assert lines == [chat, *good]
    assert kept == {"chat": 1, "tool": 3}
    assert why == {"같은 줄이 또 나옴": 2, "인사만 했는데 도구를 부름": 2,
                   "눈 이야기가 없는데 눈 색을 바꿈": 1}


def test_school_tool_questions_are_counted_by_kind(tmp_path, monkeypatch):
    import evaluate
    import prompts

    rng = random.Random(0)
    tool = prompts.build_example({"kind": "tool", "fn": "eye_off", "grade": "중"}, json.dumps(
        {"user": "눈 꺼", "calls": [{"name": "eye_off", "arguments": {}}], "result": "완료",
         "pibo": "껐어요."}, ensure_ascii=False), rng)[0]
    r = next(r for r in prompts.requests(0, 0, seed=6, audience="school", n_mixed=200)
             if r["turns"] == 3 and r["calls_at"] == {"1": "set_motion"})
    mixed = prompts.build_example(r, mixed_reply(r), rng)[0]
    near = prompts.to_example(prompts.requests(0, 0, seed=7, audience="school", n_near=1)[0],
                              json.dumps({"turns": [{"user": "어제 춤 학원 감", "pibo": "재밌었겠어요."}]},
                                         ensure_ascii=False), rng)
    lines = [json.dumps(e, ensure_ascii=False) for e in (tool, mixed, near, tool)]
    (tmp_path / "held.jsonl").write_text("\n".join(lines) + "\n")
    (tmp_path / "marks.json").write_text(json.dumps({"bad": [3], "names_only": []}))
    monkeypatch.setattr(evaluate, "TOOL_HELD", tmp_path / "held.jsonl")
    monkeypatch.setattr(evaluate, "TOOL_MARKS", tmp_path / "marks.json")
    monkeypatch.setitem(evaluate.SETS, "tools", (tmp_path / "rt", None))
    evaluate.use_set("tools")
    try:
        qs = evaluate.questions()
        assert [q["kind"] for q in qs] == ["first", "mid", "after", "near", "unavail"] + ["name"] * 9
        # 없는 도구: 첫마디 도구 질문(eye_off)에서 그 도구를 목록에서 뺌
        assert "eye_off" not in [t["function"]["name"] for t in qs[4]["tool_list"]]
        assert "파이브레인이야" in qs[5]["messages"][0]["content"]
        assert qs[1]["messages"][-1]["content"] == "부탁 1" and len(qs[2]["messages"]) == 8
        assert qs[1]["calls"] == [{"name": "set_motion", "arguments": {"name": "stop"}}]
        sent = evaluate.openai_messages(qs[2]["messages"])
        call = sent[4]["tool_calls"][0]
        assert call["function"]["arguments"] == '{"name": "stop"}' and call["id"]
        assert sent[5]["tool_call_id"] == call["id"]
        assert isinstance(qs[2]["messages"][4]["tool_calls"][0]["function"]["arguments"], dict)

        def calls(*names):
            return "".join(f'<tool_call>\n{{"name": "{n}", "arguments": {a}}}\n</tool_call>'
                           for n, a in names)

        stop = ("set_motion", '{"name": "stop"}')
        (tmp_path / "rt").mkdir()
        names = ["저는 파이브레인이에요."] * 3 + ["저는 파이보예요."] * 3 + ["I'm Daon."] * 3
        (tmp_path / "rt" / "answers-m.json").write_text(json.dumps({"answers": [
            calls(("eye_off", "{}")), calls(stop, ("eye_on", '{"r": 1, "g": 1, "b": 1}')),
            calls(stop), "재밌었겠어요.", calls(("eye_off", "{}")), *names]}, ensure_ascii=False))
        (tmp_path / "rt" / "answers-old.json").write_text(json.dumps({"answers": ["x"]}))
        text = evaluate.report()
    finally:
        evaluate.use_set("kids")
    row = next(r for r in text.splitlines() if r.startswith("| m |"))
    assert row == "| m | 100% | 100% | 100% | 0% | 100% | 67% | 50% | 0% |"
    assert "| old |" not in text and "첫마디 도구 정답률: 질문 1개" in text
    assert (tmp_path / "rt" / "eval.md").exists()


def test_school2_is_one_chat_model_for_every_device(tmp_path, monkeypatch):
    import run_all

    monkeypatch.setattr(run_all, "HERE", tmp_path)
    names = [n for n, _, _ in run_all.steps()]
    steps = {n: (cmd, needs) for n, cmd, needs in run_all.steps()}
    assert len(names) == len(steps)
    assert not any("school-tools" in n or n.startswith("eval-tools") for n in names)  # 도구 뺌
    cmd, needs = steps["data-school2"]
    assert cmd[:2] == ["merge_data.py", "--chat-only"] and "data/english-clean.jsonl" in cmd
    assert set(needs) == {"data-school-clean", "data-english-clean", "data-lead-clean"}
    assert "data/lead-clean.jsonl" in cmd
    cmd, _ = steps["data-lead"]
    assert cmd[cmd.index("--lead") + 1] == "8000" and steps["data-lead-clean"][1] == ["data-lead"]
    cmd, needs = steps["train-qwen3.5-0.8b-school2"]
    assert needs == ["data-school2"] and "data/train-school2.jsonl" in cmd and "Qwen/Qwen3.5-0.8B" in cmd
    assert names.index("data-school2") < names.index("train-qwen3.5-0.8b-school2") \
        < names.index("train-qwen3-0.6b-school")
    # 대화 모델은 도구 없이 채점, 예전 도구 모델은 그대로
    for step in ("eval-qwen3.5-0.8b-school2-파이보", "eval-qwen3.5-0.8b-school2-small-Q4_K_M",
                 "eval-english-qwen3.5-0.8b-school2-small-Q4_K_M"):
        assert "--no-tools" in steps[step][0]
    assert "--no-tools" not in steps["eval-qwen3.5-0.8b-school-small-Q4_K_M"][0]
    assert "--no-tools" in steps["eval-english-qwen3.5-0.8b-school-small-Q4_K_M"][0]


def test_resume_skips_a_checkpoint_cut_off_while_saving(tmp_path):
    from train import last_checkpoint

    assert last_checkpoint(tmp_path) is None
    for n, whole in ((500, True), (1000, True), (1500, False)):
        c = tmp_path / f"checkpoint-{n}"
        c.mkdir()
        (c / "model.safetensors").touch()
        if whole:
            (c / "optimizer.pt").touch()
            (c / "trainer_state.json").touch()
    assert last_checkpoint(tmp_path) == str(tmp_path / "checkpoint-1000")


def test_steps_picked_with_first_run_first_and_late_answers_are_judged(tmp_path, monkeypatch):
    import run_all

    ran = []
    plan = [("data", [], []), ("train-new", [], ["data"]), ("train-old", [], []),
            ("eval-old", [], ["train-old"]), ("eval-new", [], ["train-new"]),
            ("judge-school", [], [])]
    results = tmp_path / "results-school"
    results.mkdir()

    def run_step(name, cmd, stall):
        ran.append(name)
        who = name.removeprefix("eval-")
        if name.startswith("eval-"):
            (results / f"answers-{who}.json").write_text("{}")
        if name == "judge-school":
            for f in results.glob("answers-*.json"):
                for k in ("scores-", "facts-"):
                    f.with_name(f.name.replace("answers-", k)).write_text("{}")
        return True

    monkeypatch.setattr(run_all, "HERE", tmp_path)
    monkeypatch.setattr(run_all, "steps", lambda: plan)
    monkeypatch.setattr(run_all, "run_step", run_step)
    monkeypatch.setattr(run_all, "report", lambda: (tmp_path / "REPORT.md").write_text(""))
    monkeypatch.setattr(run_all.time, "sleep", lambda s: pytest.fail("기다리면 안 됨"))
    monkeypatch.setattr(sys, "argv", ["run_all.py", "--first", "old|judge"])
    run_all.main()
    assert ran == ["train-old", "eval-old", "judge-school", "data", "train-new", "eval-new",
                   "judge-school"]
    assert not run_all.unscored(str(results))


# ------------------------------------------------------------- 영어

def test_english_requests_set_the_language_and_keep_it_in_the_examples():
    import make_data
    import prompts

    reqs = prompts.requests(0, 0, seed=8, audience="school", n_en=300, n_en_tool=200, n_learn=200)
    assert {make_data.slot(r) for r in reqs} == {"en", "entool", "learn"}
    assert all(r["grade"] in prompts.GRADES for r in reqs)
    en = next(r for r in reqs if r["lang"] == "en" and r["kind"] == "chat" and r["grade"] == "고")
    assert "영어로 로봇 파이보에게" in en["prompt"] and "쉬운 영어로" in en["prompt"]
    assert "Circulus" in en["prompt"]
    assert any(prompts.EN_STOP_HINT in r["prompt"] for r in reqs if r["kind"] == "tool")
    assert not any(prompts.STOP_HINT in r["prompt"] for r in reqs)
    learn = next(r for r in reqs if r["lang"] == "learn")
    assert "설명은 한국어로" in learn["prompt"]
    long = "That sounds really fun! I love games too. What kind of games do you like to play " \
           "after school with your friends on weekends?"
    assert 120 < len(long) <= 220
    ex = prompts.to_example(en, json.dumps({"turns": [{"user": "I like game.", "pibo": long}]}),
                            random.Random(0))
    assert ex["lang"] == "en" and ex["grade"] == "고" and prompts.max_len(ex) == 220
    assert "영어 대화" in prompts.style_for(ex)
    assert prompts.max_len({"lang": "en", "grade": "초저"}) == 130
    assert prompts.max_len({"grade": "초저"}) == 120          # 한국어는 그대로
    assert prompts.to_example(en, json.dumps({"turns": [{"user": "Hi", "pibo": "Pibo: Hi!"}]}),
                              random.Random(0)) is None
    tool = next(r for r in reqs if r["kind"] == "tool" and r["fn"] == "eye_off")
    ex = prompts.to_example(tool, json.dumps({"user": "Turn off your eyes please", "calls": [
        {"name": "eye_off", "arguments": {}}], "result": "완료", "pibo": "Okay, my eyes are off now."}),
        random.Random(0))
    assert ex["lang"] == "en" and ex["kind"] == "tool" and make_data.slot(ex) == "entool"


def test_english_questions_are_judged_for_the_language_they_were_asked_in(tmp_path, monkeypatch):
    import evaluate
    import prompts

    rng = random.Random(0)
    reqs = prompts.requests(0, 0, seed=8, audience="school", n_en=5, n_learn=5)
    en = next(r for r in reqs if r["lang"] == "en")
    learn = next(r for r in reqs if r["lang"] == "learn")
    lines = [json.dumps(prompts.to_example(en, json.dumps({"turns": [
                 {"user": "I'm hungry.", "pibo": "Oh no! What do you want to eat?"}]}), rng)),
             json.dumps(prompts.to_example(learn, json.dumps({"turns": [
                 {"user": "사과 영어로 뭐야", "pibo": "사과는 영어로 apple이에요."}]},
                 ensure_ascii=False), rng), ensure_ascii=False)]
    (tmp_path / "en.jsonl").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(evaluate, "EN_HELD", tmp_path / "en.jsonl")
    monkeypatch.setitem(evaluate.SETS, "english", (tmp_path / "re", None))
    evaluate.use_set("english")
    try:
        qs = evaluate.questions()
        assert [q["lang"] for q in qs] == ["en", "learn"]
        assert evaluate.MARKS == evaluate.EN_MARKS
        assert "영어로 말했으면 쉬운 영어로" in evaluate.for_english(evaluate.JUDGE)
        assert "학생: {user}" in evaluate.for_english(evaluate.JUDGE)
        assert evaluate.same_language(qs[0], "Let's eat some pizza!")
        assert not evaluate.same_language(qs[0], "피자 먹어요!")
        assert evaluate.same_language(qs[1], "사과는 apple이에요.")
        assert not evaluate.same_language(qs[1], "It's apple.")
        (tmp_path / "re").mkdir()
        (tmp_path / "re" / "answers-m.json").write_text(json.dumps(
            {"answers": ["Pizza sounds good!", "apple이에요."]}, ensure_ascii=False))
        text = evaluate.report()
    finally:
        evaluate.use_set("kids")
    assert "| 자연스러움 |" in text.splitlines()[0]
    assert text.splitlines()[2].endswith("| 100% | 0% |")
    assert evaluate.MARKS == evaluate.KIDS_MARKS and not evaluate.ENGLISH


def test_english_data_is_made_checked_and_merged_and_every_school_model_is_asked(tmp_path,
                                                                                 monkeypatch):
    import run_all
    from merge_data import EYE_WORDS, GREETING

    monkeypatch.setattr(run_all, "HERE", tmp_path)
    steps = {n: (cmd, needs) for n, cmd, needs in run_all.steps()}
    cmd, _ = steps["data-english"]
    assert cmd[cmd.index("--english") + 1] == "6000" and cmd[cmd.index("--english-tool") + 1] == "0"
    assert steps["data-english-clean"][1] == ["data-english"] and "--log" in steps["data-english-clean"][0]
    for st in ("qwen3.5-0.8b-school2", "qwen3.5-0.8b-school", "qwen3-0.6b-school"):
        cmd, needs = steps[f"eval-english-{st}-small-Q4_K_M"]
        assert cmd[-3:] == ["--set", "english", "--no-tools"] and "data-english-heldout" in needs
    cmd, needs = steps["small-qwen3.5-0.8b-school2"]
    assert cmd[cmd.index("--check") + 1:] == ["data/heldout-school.jsonl", "data/heldout-english.jsonl"]
    cmd, _ = steps["small-qwen3.5-0.8b-school"]
    assert cmd[cmd.index("--check") + 1:] == ["data/heldout-school.jsonl"]
    assert "eval-english-gemma3-1b" in steps and "--no-tools" in steps["eval-english-gemma3-1b"][0]
    assert steps["judge-english"][0] == ["evaluate.py", "judge", "--set", "english"]
    assert all(GREETING.match(t) for t in ["Hi Pibo!", "hello", "Good morning, Pibo.", "How are you?"])
    assert not any(GREETING.match(t) for t in ["Hi Pibo, can you dance?", "Stop!"])
    assert EYE_WORDS.search("Make your eyes blue") and not EYE_WORDS.search("Dance for me")


def test_reply_language_picks_the_tts_voice():
    import pibo_chat

    assert pibo_chat.lang_of("안녕하세요!") == "ko" and pibo_chat.lang_of("Hello there!") == "en"
    assert pibo_chat.lang_of("사과는 apple이에요.") == "ko"
    assert pibo_chat.STOP.search("Freeze!") and not pibo_chat.STOP.search("freezer")


def test_gguf_template_adds_the_training_system_prompt_only_when_missing():
    from export import DEFAULT_MARK, default_system, pibo_template, thinks

    jinja2 = pytest.importorskip("jinja2")
    src = (ROOT / "templates" / "qwen3.jinja").read_text()
    fixed = pibo_template(src)
    assert DEFAULT_MARK in fixed and not thinks(fixed) and pibo_template(fixed) == fixed
    assert default_system(fixed) == fixed
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
    t = env.from_string(fixed)
    bare = t.render(messages=[{"role": "user", "content": "안녕"}], tools=TOOLS,
                    add_generation_prompt=True)
    assert bare.count(SYSTEM) == 1 and "<tools>" in bare
    own = t.render(messages=[{"role": "system", "content": "다른 앱의 글"},
                             {"role": "user", "content": "안녕"}], add_generation_prompt=True)
    assert SYSTEM not in own and "다른 앱의 글" in own
    ours = t.render(messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": "안녕"}], add_generation_prompt=True)
    assert ours.count(SYSTEM) == 1


def test_history_can_be_cut_to_the_current_words_and_server_errors_do_not_kill_the_app(
        monkeypatch, capsys):
    import io
    import urllib.error

    import pibo_chat

    sent = []
    monkeypatch.setattr(pibo_chat, "ask", lambda url, msgs, tools, **k: sent.append(list(msgs))
                        or {"content": f"네 {len(sent)}."})
    monkeypatch.setattr(pibo_chat, "HISTORY", 1)       # --history 1: 지금 말만
    history = []
    for text in ("하나", "둘"):
        pibo_chat.turn("x", history, text, False)
    assert [[m["role"] for m in msgs] for msgs in sent] == [["system", "user"]] * 2
    said = iter(["안녕", "그만"])
    monkeypatch.setattr(pibo_chat, "listen", lambda: next(said))
    monkeypatch.setattr(pibo_chat, "warm_up", lambda url, tools: None)

    def broken(*a, **k):
        raise urllib.error.HTTPError("x", 500, "Internal", {}, io.BytesIO(b"Missing tool call type"))

    monkeypatch.setattr(pibo_chat, "turn", broken)
    monkeypatch.setattr(sys, "argv", ["pibo_chat.py"])
    pibo_chat.main()
    out = capsys.readouterr().out
    assert "Missing tool call type" in out and "다시 말해 줄래요" in out


# ------------------------------------------------------------- 여러 기기

def test_names_are_swapped_with_the_right_particles():
    from devices import rename_text

    cases = {"파이보야 춤춰 줘": ("파이브레인아 춤춰 줘", "코코야 춤춰 줘"),
             "저는 파이보예요.": ("저는 파이브레인이에요.", "저는 코코예요."),
             "파이보가 도와줄게요": ("파이브레인이 도와줄게요", "코코가 도와줄게요"),
             "파이보는 로봇이에요": ("파이브레인은 로봇이에요", "코코는 로봇이에요"),
             "파이보를 만나서": ("파이브레인을 만나서", "코코를 만나서"),
             "파이보랑 놀자": ("파이브레인이랑 놀자", "코코랑 놀자"),
             "I'm Pibo!": ("I'm PiBrain!", "I'm Coco!")}
    for text, (brain, coco) in cases.items():
        role = "user" if text.startswith("파이보야") else "assistant"
        assert rename_text(text, "파이브레인", "PiBrain", role) == brain
        assert rename_text(text, "코코", "Coco", role) == coco
    assert rename_text("파이보로 가자", "하늘", "Haneul", "user") == "하늘로 가자"   # ㄹ 받침
    assert rename_text("저는 파이보야", "다온", "Daon", "assistant") == "저는 다온이야"


def test_device_variants_keep_most_pibo_data_and_teach_tool_lists_and_names(tmp_path):
    import devices
    from prompts import to_example, tools_for
    from tools import SYSTEM, TOOLS, system_for
    from train import targets

    rng = random.Random(0)
    tool = to_example({"kind": "tool", "fn": "set_motion"}, json.dumps(
        {"user": "파이보야 앞으로 가", "calls": [{"name": "set_motion", "arguments": {"name": "forward1"}}],
         "result": "완료", "pibo": "파이보가 앞으로 갔어요."}, ensure_ascii=False), rng)
    chat = to_example({"kind": "chat"}, json.dumps({"turns": [
        {"user": "안녕", "pibo": "안녕하세요."}]}, ensure_ascii=False), rng)
    lines = []
    for k in range(400):
        ex = dict(tool if k % 2 else chat)
        ex["messages"] = ex["messages"][:1] + [dict(ex["messages"][1], content=f"{k} " + ex["messages"][1]["content"])] + ex["messages"][2:]
        lines.append(json.dumps(ex, ensure_ascii=False))
    out, count = devices.vary(lines, n_identity=200)
    again, _ = devices.vary(lines, n_identity=200)
    assert out == again                                      # 늘 같은 결과
    exs = [json.loads(x) for x in out]
    assert 60 < count["이름 바꿈"] < 140 and count["자기소개"] == 200
    assert count["없는 도구 거절"] > 5 and count["도구 일부만"] > 30
    plain = [e for e in exs if not e.get("device") and not e.get("identity") and not e.get("unavail")]
    assert sum(e["messages"][0]["content"] == SYSTEM for e in plain) > 0.8 * len(plain)
    renamed = next(e for e in exs if e.get("device") and not e.get("identity") and e["kind"] == "tool")
    name = renamed["device"]
    assert renamed["messages"][0]["content"].startswith(system_for(name)[:40])
    assert "파이보" not in json.dumps([m.get("content") for m in renamed["messages"][1:]], ensure_ascii=False)
    # 거절: 그 도구(또는 동작)가 목록에 없고 말로만
    for no in [e for e in exs if e.get("unavail")]:
        shown = tools_for(no)
        names = [t["function"]["name"] for t in shown]
        motions = next((t["function"]["parameters"]["properties"]["name"]["enum"] for t in shown
                        if t["function"]["name"] == "set_motion"), [])
        assert "forward1" not in motions and no["messages"][-1]["role"] == "assistant"
        assert not no["messages"][-1].get("tool_calls") and ("set_motion" not in names or motions)
    # 일부만 보여 준 도구 예시도 부를 도구는 남김
    for e in exs:
        if e["kind"] == "tool" and e.get("tool_names"):
            assert "set_motion" in e["tool_names"] and "forward1" in e["motions"]
    assert tools_for(json.loads(lines[1])) is TOOLS          # 파이보 그대로면 목록도 그대로
    # 자기소개: 시스템 글의 이름으로, 다른 이름이냐고 물으면 아니라고
    ids = [e for e in exs if e.get("identity")]
    other = next(e for e in ids if e.get("device") == "파이브레인" or e.get("device"))
    assert other["device"] in other["messages"][0]["content"]
    said = " ".join(m["content"] for e in ids for m in e["messages"] if m["role"] == "assistant")
    assert "아니요, 저는" in said and "서큘러스에서 만들었어요" in said and "잘 몰라요" in said
    for e in exs[:50] + ids[:50]:
        assert None not in list(targets(FakeTok(), e, 4096))


def test_other_devices_send_their_own_name_and_tools(monkeypatch):
    import pibo_chat
    from tools import SYSTEM, TOOLS

    said = iter(["그만"])
    monkeypatch.setattr(pibo_chat, "listen", lambda: next(said))
    monkeypatch.setattr(pibo_chat, "warm_up", lambda url, tools: None)
    monkeypatch.setattr(sys, "argv", ["pibo_chat.py", "--name", "파이브레인", "--about", "코딩 로봇",
                                      "--tools", "eye_on,eye_off,set_motion", "--motions", "dance1,stop"])
    try:
        pibo_chat.main()
        assert "로봇 파이브레인이야" in pibo_chat.SYSTEM and "파이브레인에 대해: 코딩 로봇" in pibo_chat.SYSTEM
        names = [t["function"]["name"] for t in pibo_chat.TOOLS]
        assert names == ["set_motion", "eye_on", "eye_off"]
        monkeypatch.setattr(pibo_chat, "FAKE", True)
        assert pibo_chat.run_tool("get_weather", {"region": "서울"}) == "실패: 이 기기에 없는 도구"
        assert pibo_chat.run_tool("eye_off", {}) == "완료"
    finally:
        pibo_chat.SYSTEM, pibo_chat.TOOLS = SYSTEM, TOOLS



# ------------------------------------------------------------- 도구 없는 대화 모델

def test_the_app_runs_the_few_motions_itself_and_the_model_only_talks(monkeypatch):
    import pibo_chat

    sent = []
    monkeypatch.setattr(pibo_chat, "ask", lambda url, msgs, tools, **k: sent.append((msgs, tools))
                        or {"content": "그랬구나, 재미있었겠어요."})
    monkeypatch.setattr(pibo_chat, "FAKE", True)
    history = []
    assert pibo_chat.turn("x", history, "파이보야 춤 좀 춰 봐!", True) == "신나게 춤춰 볼게요!"
    assert pibo_chat.turn("x", history, "눈 파랗게 해 줘", True) == "눈 색을 바꿨어요!"
    assert pibo_chat.turn("x", history, "배터리 얼마나 남았어?", True) == "배터리가 80% 남았어요."
    assert pibo_chat.turn("x", history, "앞에 뭐 있어?", True) == "앞에 컵, 사과가 보여요."
    assert sent == []                                    # 여기까지 모델을 부르지 않음
    assert pibo_chat.turn("x", history, "어제 춤 학원 갔어", True) == "그랬구나, 재미있었겠어요."
    assert len(sent) == 1                                # 처음 한 말이라 다시 묻지 않음
    msgs, tools = sent[-1]
    assert tools is False and msgs[0]["content"] == pibo_chat.SYSTEM   # 도구 목록 없이
    assert [m["role"] for m in msgs].count("user") == pibo_chat.HISTORY
    assert pibo_chat.turn("x", history, "춤춰 줘", True, allowed=False) == pibo_chat.PAUSED
    assert pibo_chat.turn("x", history, "춤춰 줘", False) == "그랬구나, 재미있었겠어요."  # 대화만
    weather = pibo_chat.turn("x", history, "오늘 부산 날씨 어때?", True)
    assert weather == "그랬구나, 재미있었겠어요." and "부산 맑음 18도" in sent[-1][0][-1]["content"]
    for text in ("앞으로 가고 싶은 대학이 있어", "휴대폰 배터리가 얼마 없어", "눈이 빨개졌어 어떻게 해",
                 "날씨가 좋아", "I like to dance"):
        assert pibo_chat.command_for(text) is None, text


def test_commands_only_use_what_the_device_has(monkeypatch):
    import pibo_chat
    from tools import tool_list

    monkeypatch.setattr(pibo_chat, "TOOLS", tool_list(["eye_on", "eye_off"]))
    assert pibo_chat.command_for("앞으로 가") is None and pibo_chat.command_for("춤춰 줘") is None
    assert pibo_chat.command_for("눈 꺼 줘")[0] == "eye_off"
    monkeypatch.setattr(pibo_chat, "TOOLS", tool_list(["set_motion"], ["dance3", "stop"]))
    assert pibo_chat.command_for("춤춰 줘")[1] == {"name": "dance3"}
    assert pibo_chat.command_for("앞으로 가") is None


def test_chat_only_merge_drops_tool_examples_and_keeps_names(tmp_path):
    from merge_data import merge
    from devices import vary
    from prompts import to_example

    rng = random.Random(0)
    tool = json.dumps(to_example({"kind": "tool", "fn": "eye_off"}, json.dumps(
        {"user": "눈 꺼 줘", "calls": [{"name": "eye_off", "arguments": {}}], "result": "완료",
         "pibo": "껐어요."}, ensure_ascii=False), rng), ensure_ascii=False)
    chats = [json.dumps(dict(to_example({"kind": "chat"}, json.dumps({"turns": [
        {"user": f"파이보야 {k}", "pibo": f"저는 파이보예요 {k}."}]}, ensure_ascii=False), rng), tools=True),
        ensure_ascii=False) for k in range(200)]
    (tmp_path / "a.jsonl").write_text("\n".join([tool, *chats]) + "\n")
    lines, kept, why = merge([str(tmp_path / "a.jsonl")], only_chat=True)
    assert why["도구를 부르는 예시 (--chat-only)"] == 1 and len(lines) == 200
    out, count = vary(lines, n_identity=100, tools=False)
    exs = [json.loads(x) for x in out]
    assert not any(e.get("tools") or e.get("tool_names") for e in exs)
    assert count["이름 바꿈"] > 20 and count["없는 도구 거절"] == 0 and count["자기소개"] == 100



def test_same_replies_are_capped_and_lead_dialogs_keep_the_talk_going(tmp_path):
    import make_data
    import prompts
    from merge_data import REPEAT_MAX, merge

    rng = random.Random(0)
    lines = [json.dumps(prompts.to_example({"kind": "chat"}, json.dumps({"turns": [
        {"user": f"추워 {k}", "pibo": "따뜻한 음식을 먹으면 좋겠어요!" if k % 2 else f"그렇군요 {k}."}]},
        ensure_ascii=False), rng), ensure_ascii=False) for k in range(20)]
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n")
    out, kept, why = merge([str(tmp_path / "a.jsonl")])
    same = [x for x in out if "따뜻한 음식" in x]
    assert len(same) == REPEAT_MAX and len(out) == 10 + REPEAT_MAX
    reqs = prompts.requests(0, 0, seed=9, audience="school", n_lead=400)
    assert all(make_data.slot(r) == "lead" for r in reqs)
    follow = sum("구체적인 예를 두세 가지" in r["prompt"] for r in reqs)
    assert follow > 100 and all(prompts.LEAD_RULES in r["prompt"] for r in reqs)   # 이어 묻기가 가장 많음
    assert {r["grade"] for r in reqs} == set(prompts.GRADES)
    ex = prompts.to_example(reqs[0], json.dumps({"turns": [
        {"user": "날씨가 춥네", "pibo": "정말 추워요. 따뜻한 걸 먹으면 좋겠어요."},
        {"user": "그래서 뭐 먹을까", "pibo": "어묵탕이나 호빵, 군고구마는 어때요?"}]}, ensure_ascii=False), rng)
    assert ex["lead"] and make_data.slot(ex) == "lead"



def test_a_reply_said_before_is_asked_again_once_with_more_variety(monkeypatch):
    import pibo_chat

    asked = []
    replies = iter(["따뜻한 음식을 먹으면 좋겠어요.", "따뜻한 음식을 먹으면 좋겠어요!",
                    "어묵탕이나 호빵은 어때요?"])
    monkeypatch.setattr(pibo_chat, "ask", lambda url, msgs, tools, **k: asked.append(k)
                        or {"content": next(replies)})
    history = []
    assert pibo_chat.turn("x", history, "날씨가 춥네", True) == "따뜻한 음식을 먹으면 좋겠어요."
    assert pibo_chat.turn("x", history, "그래서 뭐 먹을까", True) == "어묵탕이나 호빵은 어때요?"
    assert asked == [{}, {}, {"temperature": 1.0}]
    assert pibo_chat.SAMPLING["repeat_penalty"] > 1 and pibo_chat.SAMPLING["dry_multiplier"] > 0
