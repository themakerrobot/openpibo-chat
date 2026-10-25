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
    monkeypatch.setattr(pibo_chat, "ask", lambda url, msgs, tools: next(replies))
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
    assert "kd-teacher" not in names and names[-2:] == ["judge", "judge-school"]
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
    monkeypatch.setattr(pibo_chat, "ask", lambda url, msgs, tools: sent.append(list(msgs)) or next(replies))
    monkeypatch.setattr(pibo_chat, "run_tool", lambda n, a: "완료")
    history = []
    pibo_chat.turn("x", history, "앞으로 가", True)
    pibo_chat.turn("x", history, "뒤로 가", True)
    roles = [m["role"] for m in sent[2]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert sent[2][2]["tool_calls"][0]["function"]["name"] == "set_motion"
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

    monkeypatch.setattr(pibo_chat, "ask", lambda url, msgs, tools: {"content": "", "tool_calls": [
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
    assert got == [{"id": "a", "function": {"name": "set_motion", "arguments": '{"name": "forward1"}'}}]
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
