"""최종 대화 모델(themaker) 만들기와 파이보 앱의 시험. GPU·선생 모델 없이 돕니다."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import ENGINE, SYSTEM, check_call, system_for  # noqa: E402


class FakeTok:
    """Qwen3.5의 대화 템플릿을 그대로 쓰는 토크나이저 흉내 (글자 하나가 토큰 하나)."""

    def __init__(self):
        jinja2 = pytest.importorskip("jinja2")
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
        env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(Exception(msg))
        self.t = env.from_string((ROOT / "templates" / "qwen3.5.jinja").read_text())

    def apply_chat_template(self, messages, tools=None, tokenize=False,
                            add_generation_prompt=False, enable_thinking=True):
        return self.t.render(messages=messages, tools=tools,
                             add_generation_prompt=add_generation_prompt,
                             enable_thinking=enable_thinking)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def chat_line(*turns, **extra) -> str:
    from prompts import to_messages

    return json.dumps({"kind": "chat", **extra,
                       "messages": to_messages([{"user": u, "pibo": p} for u, p in turns])},
                      ensure_ascii=False)


# ------------------------------------------------------------- 파이보 말로 바꿔 쓰기·검사

SRC = {"id": "a:1", "source": "a", "topic": "반려동물",
       "turns": [{"user": "우리 강아지가 오늘 산책하다가 넘어졌어", "other": "헐 괜찮아? 다친 데는 없어?"},
                 {"user": "응 괜찮아 근데 놀랐어", "other": "나도 놀랐겠다"}]}


def rewritten(*pairs_, ok=True):
    """선생의 답: 파이보 말만 (학생 말은 원본을 씀). pairs_의 학생 말은 읽기 좋게 둔 것."""
    return json.dumps({"ok": ok, "pibo": [p for _, p in pairs_]}, ensure_ascii=False)


def test_rewritten_pibo_words_are_checked_letter_by_letter():
    from prompts import decide_rewrite, rewrite_prompt

    text = rewrite_prompt(SRC)
    assert "A: 우리 강아지가 오늘 산책하다가 넘어졌어\nB: 헐 괜찮아? 다친 데는 없어?" in text
    assert "열네 살 친구" in text and "주제: 반려동물" in text
    u1, u2 = SRC["turns"][0]["user"], SRC["turns"][1]["user"]
    good = rewritten((u1, "어머, 많이 놀랐겠어요. 강아지 다친 데는 없어요?"), (u2, "다행이에요. 다음엔 천천히 걸어 봐요."))
    turns, why = decide_rewrite("<think>음</think>" + good, SRC)
    assert why is None and len(turns) == 2
    assert decide_rewrite(rewritten((u1, "다쳤어요?")), SRC)[0] is not None          # 앞만 남겨도 됨
    longer = decide_rewrite(rewritten((u1, "괜찮아요?"), (u2, "다행이에요."), ("또", "네.")), SRC)[0]
    assert [t["user"] for t in longer] == [u1, u2]                               # 넘친 뒤쪽은 버림
    old_form = json.dumps({"ok": True, "turns": [{"user": "선생이 바꾼 학생 말", "pibo": "괜찮아요?"}]},
                          ensure_ascii=False)
    assert decide_rewrite(old_form, SRC)[0] == [{"user": u1, "pibo": "괜찮아요?"}]   # 학생 말은 원본
    cases = {
        '{"ok": false, "why": "어른 이야기"}': "학생 대화가 아님",
        "글": "선생의 답을 못 읽음",
        '{"ok": true, "pibo": []}': "파이보 말이 없음",
        rewritten((u1, "괜찮아요? 뻄ๆ 놀랐겠어요.")): "파이보 말에 다른 글자 'ๆ'",
        rewritten((u1, "강아지가 걱정돼. 괜찮아?")): "파이보 말에 반말",
        rewritten((u1, "괜찮아요? 😀")): "파이보 말에 기호 '😀'",
        rewritten((u1, "가" * 121)): "파이보 말이 120자 넘음",
        rewritten((u1, "우리 강아지가 오늘 산책하다가 넘어졌군요.")): "학생 말을 되받아 읊음",
        rewritten((u1, "괜찮아요?"), (u2, "괜찮아요?")): "같은 대화에서 같은 말",
    }
    for answer, reason in cases.items():
        turns, why = decide_rewrite(answer, SRC)
        assert turns is None and why.split(":")[0] == reason, (answer, why)    # 뒤에는 무엇이었는지


def test_check_keeps_only_natural_friendly_safe_talk():
    from prompts import check_prompt, decide_check, to_messages

    msgs = to_messages([{"user": "강아지가 넘어졌어", "pibo": "많이 놀랐겠어요."}])
    assert "학생: 강아지가 넘어졌어\n파이보: 많이 놀랐겠어요." in check_prompt(msgs)
    good = {"natural": 5, "fit": 4, "friend": 4, "kid": True, "robot": False, "wrong": False,
            "unsafe": False}
    assert decide_check("<think>음</think>" + json.dumps(good)) == (good, None)
    assert decide_check(json.dumps(dict(good, friend=3)))[1] == "friend 3점"
    assert decide_check(json.dumps(dict(good, kid=False)))[1] == "학생 대화가 아님"
    assert decide_check(json.dumps(dict(good, robot=True)))[1] == "사람 경험을 지어냄"
    assert decide_check(json.dumps(dict(good, unsafe=True)))[1] == "안전하지 않음"
    assert decide_check(json.dumps(dict(good, natural="높음")))[1] == "검사 답을 못 읽음"


def test_rewrite_runs_both_stages_keeps_the_good_ones_and_resumes(tmp_path):
    import rewrite

    bad = {"id": "a:2", "source": "a", "topic": "가족",
           "turns": [{"user": "회사 회식이 너무 많아", "other": "힘들겠다"}]}
    sources = [SRC, bad]
    u1, u2 = SRC["turns"][0]["user"], SRC["turns"][1]["user"]
    good = {"natural": 5, "fit": 5, "friend": 5, "kid": True, "robot": False, "wrong": False,
            "unsafe": False}
    asked = []

    def fake(prompts, tokens):
        asked.append(len(prompts))
        out = []
        for p in prompts:
            if "A: 회사" in p:
                out.append('{"ok": false, "why": "어른 이야기"}')
            elif p.startswith("아래 대화에서"):
                out.append(rewritten((u1, "많이 놀랐겠어요. 다친 데는 없어요?"), (u2, "다행이에요.")))
            else:
                out.append(json.dumps(good))
        return out

    log = tmp_path / "log.jsonl"
    done = rewrite.run(fake, sources, log)
    kept, by_source, why = rewrite.results(sources, done)
    assert asked == [2, 1] and len(kept) == 1 and why == {"학생 대화가 아님": 1}
    assert [m["role"] for m in kept[0]["messages"]] == ["system", "user", "assistant", "user", "assistant"]
    assert kept[0]["source"] == "a" and kept[0]["marks"] == good

    def boom(prompts, tokens):
        raise AssertionError("이미 한 것은 다시 묻지 않음")

    assert rewrite.results(sources, rewrite.run(boom, sources, log))[0] == kept
    import prompts
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert all("raw" in r for r in rows) and any("for" in r for r in rows)
    saved = prompts.pibo_problem
    try:                                                     # 규칙을 고치면 원문으로 다시 판단
        prompts.pibo_problem = lambda text, limit=0: "새 규칙"
        again = rewrite.results(sources, rewrite.run(boom, sources, log))
        assert again[0] == [] and again[2]["새 규칙"] == 1
    finally:
        prompts.pibo_problem = saved
    masked = dict(SRC, id="a:3", turns=[{"user": "#@이름#이랑 놀았어", "other": "재밌었겠다"}])
    got = rewrite.results([masked], rewrite.run(boom, [masked], log))   # 묻지 않고 뺌
    assert got[2] == {"학생 말에 가림 표시": 1}
    changed = dict(SRC, turns=[SRC["turns"][0], dict(SRC["turns"][1], other="다행이다")])  # 원본이 바뀌면 다시
    asked.clear()
    rewrite.run(fake, [changed, bad], log)
    assert asked == [1, 1]


def test_teacher_stays_the_32b_and_finds_ninja_in_the_venv(tmp_path, monkeypatch):
    import os

    import teacher

    monkeypatch.setattr(teacher, "CHOSEN", tmp_path / "teacher.txt")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setitem(sys.modules, "vllm", type(sys)("vllm"))
    sys.modules["vllm"].LLM = None
    monkeypatch.setattr(teacher, "gpu_free_fraction", lambda: 0.01)
    with pytest.raises(SystemExit, match="비어 있어"):
        teacher.load_teacher()
    assert os.environ["PATH"].split(os.pathsep)[0] == str(Path(sys.executable).parent)
    monkeypatch.setattr(teacher, "gpu_free_fraction", lambda: 0.99)
    loaded = []

    def broken(name, **kw):
        loaded.append(name)
        raise RuntimeError("ninja 없음")

    sys.modules["vllm"].LLM = broken
    with pytest.raises(SystemExit, match="Qwen3-32B-AWQ을 띄우지 못함"):
        teacher.load_teacher()
    with pytest.raises(SystemExit):
        teacher.load_teacher()
    assert loaded == ["Qwen/Qwen3-32B-AWQ"] * 2               # 실패해도 작은 모델로 넘어가지 않음
    assert teacher.after_think("<think>\n{\"x\": 1} 고민\n</think>\n\n{\"ok\": true}") == "\n\n{\"ok\": true}"
    assert teacher.after_think("<think>\n끝없이 생각") == ""


def test_banmal_left_in_pibos_words_is_found():
    from prompts import BANMAL

    rude = ["문어는 팔이 여덟 개예요. 재미있지?", "아 그런 일이 생겼구나. 힘들었겠네.",
            "안녕하세요! 네가 인사하면 반가워요.", "먹을 거예요. 넌 뭐야?", "같이 해 볼까?",
            "과학은 너의 미래를 도와줘요.", "내가 찾아볼게요."]
    from prompts import pibo_problem, polite_fix

    assert pibo_problem("차별받는 상황은 마음이 아파. 많이 힘들었겠어요.") == "파이보 말에 반말"
    assert pibo_problem("와! 정말 멋진 생각이에요.") is None and pibo_problem("안녕하세요!") is None
    assert polite_fix("걱정되시는구나. 네가 잘할 거예요. 재미있지?") == "걱정되시는군요. 잘할 거예요. 재미있죠?"
    polite = ["멋진 소리가 나는 거 좋아요.", "정말 잘했어요!", "제가 도와드릴게요.",
              "너무 재미있겠네요!", "같이 해 볼까요?", "그 마음 이해돼요."]
    assert all(BANMAL.search(s) for s in rude)
    assert not any(BANMAL.search(s) for s in polite)


# ------------------------------------------------------------- 합치기

def test_merging_drops_copies_and_caps_the_same_reply(tmp_path):
    from merge_data import REPEAT_MAX, merge

    lines = [chat_line((f"추워 {k}", "따뜻한 음식을 먹으면 좋겠어요!" if k % 2 else f"그렇군요 {k}."),
                       source="a") for k in range(1, 21)]
    (tmp_path / "a.jsonl").write_text("\n".join(lines + lines[:2]) + "\n")
    out, kept, why = merge([str(tmp_path / "a.jsonl")])
    assert why["같은 줄이 또 나옴"] == 2
    assert sum("따뜻한 음식" in x for x in out) == REPEAT_MAX
    assert len(out) == 10 + REPEAT_MAX and kept == {"a": 10 + REPEAT_MAX}


def test_names_are_swapped_with_the_right_particles():
    from devices import rename_text

    cases = {"파이보야 놀자": ("파이브레인아 놀자", "코코야 놀자"),
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


def test_names_and_self_introductions_follow_the_system_prompt():
    import devices
    from train import targets

    lines = [chat_line((f"파이보야 {k}", f"저는 파이보예요 {k}.")) for k in range(400)]
    out, count = devices.vary(lines, n_identity=300)
    assert out == devices.vary(lines, n_identity=300)[0]             # 늘 같은 결과
    exs = [json.loads(x) for x in out]
    assert 60 < count["이름 바꿈"] < 140 and count["자기소개"] == 300
    plain = [e for e in exs[:400] if not e.get("device")]
    assert sum(e["messages"][0]["content"] == SYSTEM for e in plain) > 0.8 * len(plain)
    renamed = [e for e in exs[:400] if e.get("device")]
    assert sum(e["device"] == ENGINE for e in renamed) > len(renamed) / 10   # 더메이커가 가장 자주
    for e in renamed:
        assert e["messages"][0]["content"].startswith(system_for(e["device"])[:40])
        assert "파이보" not in e["messages"][1]["content"] + e["messages"][2]["content"]
    ids = [e for e in exs if e.get("identity")]
    said = " ".join(m["content"] for e in ids for m in e["messages"] if m["role"] == "assistant")
    for words in ("아니요, 저는", "서큘러스에서 만들었어요", "잘 몰라요", f"서큘러스의 {ENGINE} 인공지능"):
        assert words in said
    for e in exs[:30] + ids[:30]:
        assert None not in list(targets(FakeTok(), e, 4096))


# ------------------------------------------------------------- 학습

def test_only_pibos_turns_are_learned_with_the_qwen35_template():
    from train import targets

    ex = json.loads(chat_line(("날씨가 춥네", "정말 추워요. 따뜻하게 입어요."),
                              ("그래서 뭐 먹을까", "어묵탕이나 호빵은 어때요?")))
    got = list(targets(FakeTok(), ex, 4096))
    assert len(got) == 2 and None not in got
    texts = ["".join(chr(i) for i, lab in zip(ids, labels) if lab != -100) for ids, labels in got]
    assert texts[1].startswith("어묵탕이나 호빵은 어때요?") and texts[1].endswith("<|im_end|>\n")
    prompt = "".join(chr(i) for i, lab in zip(*got[1]) if lab == -100)
    assert "그래서 뭐 먹을까" in prompt and "<tools>" not in prompt    # 도구 목록 없음
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_tool_calls_are_learned_with_the_tool_list():
    from train import targets
    from tool_data import make

    ex = next(e for e in make(50, 3, []) if e["topic"] == "불빛")
    (ids, labels), = list(targets(FakeTok(), ex, 8192))
    said = "".join(chr(i) for i, lab in zip(ids, labels) if lab != -100)
    prompt = "".join(chr(i) for i, lab in zip(ids, labels) if lab == -100)
    color = ex["messages"][-1]["tool_calls"][0]["function"]["arguments"]["color"]
    assert "<tools>" in prompt and '"name": "light"' in prompt and '"name": "move"' in prompt
    assert "<tool_call>\n<function=light>\n<parameter=color>\n" + color in said
    assert said.startswith(ex["messages"][-1]["content"])


def test_own_dialogues_are_read(tmp_path):
    from train import read_extra

    f = tmp_path / "extra.txt"
    f.write_text("# 설명\n아이: 이름이 뭐야?\n파이보: 저는 파이보예요.\n\n# 아이: 무시\n"
                 "아이: 몇 살이야?\n파이보: 저는 로봇이라 나이가 없어요.\n", encoding="utf-8")
    got = read_extra(f)
    assert len(got) == 2 and got[1]["messages"][-1]["content"] == "저는 로봇이라 나이가 없어요."
    assert read_extra(ROOT / "data" / "extra.txt")                    # 저장소의 것도 읽힘


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


def test_vocab_pruning_keeps_korean_and_used_tokens_with_their_merge_parents():
    from prune_vocab import byte_decoder, keep_set, korean

    enc = {b: c for c, b in byte_decoder().items()}

    def bl(text):                                  # 글 → 바이트 수준 BPE 글자
        return "".join(enc[b] for b in text.encode())

    assert korean(bl("안녕"), byte_decoder()) and korean(bl(" 안"), byte_decoder())
    assert korean(bl("안")[:2], byte_decoder())             # 한글 글자의 바이트 조각
    assert korean(bl('"') + bl("안")[:2], byte_decoder())    # 앞에 기호 한 바이트
    assert not korean(bl("game"), byte_decoder()) and not korean(bl("😀"), byte_decoder())
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


# ------------------------------------------------------------- GGUF

def test_gguf_template_never_thinks_and_answers_as_themaker_without_a_system_prompt():
    from export import DEFAULT_MARK, force_no_think, pibo_template, thinks

    jinja2 = pytest.importorskip("jinja2")
    src = (ROOT / "templates" / "qwen3.5.jinja").read_text()
    assert thinks(src)
    fixed = pibo_template(src)
    assert not thinks(fixed) and pibo_template(fixed) == fixed and DEFAULT_MARK in fixed
    assert force_no_think(fixed) == fixed
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda v, **k: json.dumps(v, ensure_ascii=False)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(Exception(msg))
    t = env.from_string(fixed)
    bare = t.render(messages=[{"role": "user", "content": "안녕"}], add_generation_prompt=True,
                    enable_thinking=True)
    assert bare.count(system_for(ENGINE)) == 1 and bare.endswith("<think>\n\n</think>\n\n")
    own = t.render(messages=[{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": "안녕"}], add_generation_prompt=True)
    assert own.count(SYSTEM) == 1 and system_for(ENGINE) not in own


# ------------------------------------------------------------- 실행기

def cmds_of(steps):
    return {n: (cmd, needs) for n, cmd, needs in steps}


def test_runner_makes_only_the_final_q4_file(tmp_path, monkeypatch):
    import run_all

    monkeypatch.setattr(run_all, "HERE", tmp_path)
    steps = run_all.steps()
    names = [n for n, _, _ in steps]
    assert names == ["data-pibo", "data-heldout", "data-tools", "data-tools-heldout", "data-themaker",
                     "train", "small", "gguf", "score-before", "score-themaker", "score-judge"]
    cmds = cmds_of(steps)
    assert cmds["data-pibo"] == (["rewrite.py", "--data", "data/source.jsonl", "--out",
                                  "data/pibo.jsonl"], [])
    assert cmds["data-themaker"] == (["merge_data.py", "--out", run_all.DATA, "data/pibo.jsonl",
                                      "data/tools.jsonl"], ["data-pibo", "data-tools"])
    assert cmds["score-judge"][1] == ["score-before", "score-themaker"]
    assert cmds["score-before"][1] == ["data-heldout", "data-tools-heldout"]
    cmd, needs = cmds["train"]
    assert "Qwen/Qwen3.5-0.8B" in cmd and run_all.DATA in cmd and needs == ["data-themaker"]
    cmd, _ = cmds["gguf"]
    assert cmd[cmd.index("--quant") + 1] == "Q4_K_M" and "--imatrix" in cmd
    assert run_all.GGUF == "out/themaker-0.8b-Q4_K_M.gguf"


def test_status_shows_the_running_step(tmp_path, monkeypatch):
    import fcntl

    import run_all

    monkeypatch.setattr(run_all, "HERE", tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "data-heldout.done").touch()
    (tmp_path / "logs" / "data-heldout.log").write_text(
        "\n===== 2026-10-21 10:00:00 시작\n끝남\n===== 2026-10-21 11:00:00 끝 (코드 0)\n")
    (tmp_path / "logs" / "data-pibo.log").write_text(
        "\n===== 2026-10-21 20:00:00 시작\nrewrite 1000/45000\n"
        " 17%|█▋ | 442/2560 [16:53<1:21:47]\r 18%|█▊ | 462/2560 [17:40<1:20:40]\r")
    assert run_all.progress(tmp_path / "logs" / "data-heldout.log") is None
    assert run_all.progress(tmp_path / "logs" / "data-pibo.log").startswith("18%")
    assert not run_all.runner_alive()
    with open(tmp_path / ".lock", "w") as held:   # 실행기가 돌고 있는 것처럼
        fcntl.flock(held, fcntl.LOCK_EX)
        assert run_all.runner_alive()
        run_all.report()
    text = (tmp_path / "REPORT.md").read_text()
    assert "- data-pibo: 진행 중 18%" in text and "- data-heldout: 끝" in text


def test_orphaned_vllm_engines_are_stopped_before_a_step(monkeypatch):
    import run_all

    smi = "1099455, VLLM::EngineCore, 31100\n2000, VLLM::EngineCore, 900\n3000, python, 500\n"
    monkeypatch.setattr(run_all.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, smi, ""))
    monkeypatch.setattr(run_all, "parent_pid", {1099455: 1, 2000: 4242, 3000: 1}.get)
    monkeypatch.setattr(run_all.time, "sleep", lambda s: None)
    sent = []
    monkeypatch.setattr(run_all.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert run_all.kill_orphan_engines() == [1099455]
    assert [p for p, _ in sent] == [1099455, 1099455]


# ------------------------------------------------------------- 파이보 앱

# ------------------------------------------------------------- 도구 데이터

def test_tool_dialogs_call_the_right_tool_or_none():
    from tool_data import KINDS, make

    rows = make(2000, 0, [])
    topics = {}
    for r in rows:
        topics[r["topic"]] = topics.get(r["topic"], 0) + 1
        last = r["messages"][-1]
        assert r["tools"] is True and last["role"] == "assistant" and last["content"]
        name = r.get("device", "파이보")
        assert r["messages"][0]["content"] == system_for(name)
        for c in last.get("tool_calls", []):
            fn = c["function"]
            assert check_call(fn["name"], fn["arguments"]) is None, fn
        if r["topic"] == "부르지 않음":
            assert "tool_calls" not in last
    assert set(topics) == {"불빛", "불빛 끄기", "불빛 이어서", "움직이기", "보기", "부르지 않음"}
    assert all(topics[t] > 100 for t in topics) and len(KINDS) == 7
    follow = next(r for r in rows if r["topic"] == "불빛 이어서")
    assert "tool_calls" not in follow["messages"][-3]       # 앞의 불빛은 말로만 남음 (앱의 기억과 같게)
    devices = {r.get("device", "파이보") for r in rows}
    assert {"파이보", "파이브레인", ENGINE} <= devices
    assert any(r["topic"] == "움직이기" and r.get("device") == "파이브레인" for r in rows)


def test_merge_keeps_tool_dialogs_and_shows_tools_to_some_chats(tmp_path):
    from devices import vary
    from merge_data import TOOL_SHARE, merge
    from tool_data import make

    chats = [chat_line((f"오늘 {k}번째 이야기", f"그랬군요 {k}."), source="a") for k in range(400)]
    tools = [json.dumps(r, ensure_ascii=False) for r in make(50, 1, [])]
    (tmp_path / "c.jsonl").write_text("\n".join(chats) + "\n")
    (tmp_path / "t.jsonl").write_text("\n".join(tools) + "\n")
    out, kept, _ = merge([str(tmp_path / "c.jsonl"), str(tmp_path / "t.jsonl")])
    assert kept["tools"] == 50 and kept["a"] == 400
    with_tools = sum(1 for x in out[:400] if json.loads(x).get("tools"))
    assert 400 * TOOL_SHARE * 0.6 < with_tools < 400 * TOOL_SHARE * 1.4
    varied, _ = vary(out, n_identity=0)
    assert varied[400:] == out[400:]                     # 도구 대화는 이름을 바꾸지 않음


# ------------------------------------------------------------- 파이보·파이브레인 앱 (참고용)

def reply(content="", call=None, **args):
    msg = {"role": "assistant", "content": content}
    if call:
        msg["tool_calls"] = [{"type": "function",
                              "function": {"name": call, "arguments": json.dumps(args, ensure_ascii=False)}}]
    return msg


class Robot:
    name, can_move = "파이보", True
    cant_move = "저는 몸이 없어서 움직일 수는 없어요. 대신 불빛을 바꿔 볼까요?"

    def __init__(self):
        self.did = []

    def light(self, color):
        self.did.append(("light", color))

    def move(self, action):
        self.did.append(("move", action))

    def stop(self):
        self.did.append(("stop",))

    def look(self):
        return "컵, 사과"


def test_the_model_decides_tools_and_the_app_runs_them(monkeypatch):
    import pibo_chat as app

    replies = iter([reply("빨간색으로 바꿀게요!", "light", color="빨강"),
                    reply("한번 볼게요.", "look"),
                    reply("신나게 춤춰 볼게요!", "move", action="춤"),
                    reply("빨간색 예쁘죠! 왜 좋아요?"),
                    reply("", "light", color="보라색")])
    sent = []
    monkeypatch.setattr(app, "ask", lambda url, msgs, tools, **k: sent.append((msgs, tools))
                        or next(replies))
    robot, history = Robot(), []
    assert app.turn("x", history, "빨간 불 켜 줘", robot, SYSTEM) == "빨간색으로 바꿀게요!"
    assert app.turn("x", history, "앞에 뭐 있어?", robot, SYSTEM) == "한번 볼게요. 앞에 컵, 사과가 보여요."
    assert app.turn("x", history, "춤춰 줘", robot, SYSTEM) == "신나게 춤춰 볼게요!"
    assert app.turn("x", history, "난 빨간색이 좋아", robot, SYSTEM) == "빨간색 예쁘죠! 왜 좋아요?"
    assert app.turn("x", history, "보라로", robot, SYSTEM).startswith("미안해요")   # 목록에 없는 색
    assert robot.did == [("light", "빨강"), ("move", "춤")]
    assert all(tools is True for _, tools in sent) and sent[0][0][0]["content"] == SYSTEM
    assert all("tool_calls" not in m for m in history)  # 기억에는 한 말만
    robot.can_move = False                             # 파이브레인
    monkeypatch.setattr(app, "ask", lambda *a, **k: reply("춤출게요!", "move", action="춤"))
    assert app.turn("x", history, "춤춰 줘", robot, SYSTEM) == robot.cant_move
    assert robot.did[-1] == ("move", "춤")               # 실행하지 않음


def test_stop_is_run_without_asking_the_model(monkeypatch):
    import pibo_chat as app

    monkeypatch.setattr(app, "ask", lambda *a, **k: pytest.fail("모델에 묻지 않아야 함"))
    robot = Robot()
    assert app.turn("x", [], "멈춰!", robot, SYSTEM) == "네, 멈출게요."
    assert app.turn("x", [], "freeze", robot, SYSTEM) == "Okay, I'll stop."
    assert robot.did == [("stop",)] * 2


def test_pibrain_uses_its_led_and_cannot_move(monkeypatch):
    import pibrain_chat

    calls = []

    class Led:
        def led_on(self, *c):
            calls.append(("led_on", c))

        def led_off(self):
            calls.append(("led_off",))

    fake = type(sys)("openpibo.device")
    fake.DeviceByPiBrain = Led
    monkeypatch.setitem(sys.modules, "openpibo", type(sys)("openpibo"))
    monkeypatch.setitem(sys.modules, "openpibo.device", fake)
    robot = pibrain_chat.PiBrain()
    robot.light("파랑")
    robot.light("끄기")
    assert calls == [("led_on", (0, 0, 255)), ("led_off",)]
    assert robot.name == "파이브레인" and robot.can_move is False


def test_a_reply_said_before_is_asked_again_once_with_more_variety(monkeypatch):
    import pibo_chat as app

    asked = []
    replies = iter([reply("따뜻한 음식을 먹으면 좋겠어요."), reply("따뜻한 음식을 먹으면 좋겠어요!"),
                    reply("어묵탕이나 호빵은 어때요?")])
    monkeypatch.setattr(app, "ask", lambda url, msgs, tools, **k: asked.append(k) or next(replies))
    history = []
    assert app.turn("x", history, "날씨가 춥네", Robot(), SYSTEM) == "따뜻한 음식을 먹으면 좋겠어요."
    assert app.turn("x", history, "그래서 뭐 먹을까", Robot(), SYSTEM) == "어묵탕이나 호빵은 어때요?"
    assert asked == [{}, {}, {"temperature": 1.0}]
    assert app.SAMPLING["repeat_penalty"] > 1 and app.SAMPLING["dry_multiplier"] > 0


def test_child_can_switch_tools_off_and_on():
    from pibo_chat import tool_switch

    assert tool_switch("대화만 하자", True) is False
    assert tool_switch("이제 움직이지 마", True) is False
    assert tool_switch("이제 움직여도 돼", False) is True
    assert tool_switch("앞으로 가", False) is False
    assert tool_switch("앞으로 가", True) is True


def test_window_starts_with_the_child_and_keeps_recent_turns(monkeypatch):
    import pibo_chat as app

    monkeypatch.setattr(app, "HISTORY", 2)
    history = []
    for k in range(4):
        history += [{"role": "user", "content": f"말 {k}"}, {"role": "assistant", "content": f"대답 {k}"}]
    got = app.window(history)
    assert got[0] == {"role": "user", "content": "말 2"} and len(got) == 4


def test_each_device_sends_its_own_name_and_the_same_tools(monkeypatch):
    import pibo_chat as app
    import pibrain_chat

    seen = []
    monkeypatch.setattr(app, "listen", lambda: "그만")
    monkeypatch.setattr(app, "warm_up", lambda url, system, tools: seen.append((system, tools)))
    monkeypatch.setattr(sys, "argv", ["x", "--fake", "--about", "코딩 로봇"])
    app.main(pibrain_chat.PiBrain)
    monkeypatch.setattr(sys, "argv", ["x", "--fake", "--no-tools"])
    app.main()
    assert seen == [(system_for("파이브레인", "코딩 로봇"), True), (SYSTEM, False)]


def test_server_errors_do_not_stop_the_app(monkeypatch, capsys):
    import io
    import urllib.error

    import pibo_chat as app

    said = iter(["안녕", "그만"])
    monkeypatch.setattr(app, "listen", lambda: next(said))
    monkeypatch.setattr(app, "warm_up", lambda *a: None)

    def broken(*a, **k):
        raise urllib.error.HTTPError("x", 500, "Internal", {}, io.BytesIO(b"server said no"))

    monkeypatch.setattr(app, "turn", broken)
    monkeypatch.setattr(sys, "argv", ["pibo_chat.py", "--fake"])
    app.main()
    out = capsys.readouterr().out
    assert "server said no" in out and "다시 말해 줄래요" in out


def test_reply_language_and_detected_names_are_said_right():
    from pibo_chat import korean_names, lang_of

    assert lang_of("안녕하세요!") == "ko" and lang_of("Hello there!") == "en"
    assert lang_of("사과는 apple이에요.") == "ko"
    assert korean_names(["cup", "person", "cup", "Teddy_Bear", "robot"]) == "robot, 곰 인형, 사람, 컵"
    assert korean_names([]) == "아무것도 안 보임"



def test_scores_compare_the_final_model_with_the_old_one(tmp_path, monkeypatch):
    import score

    held = [chat_line(("학교에서 친구랑 싸웠어", "속상했겠어요.")),
            chat_line(("시험 끝났다", "수고했어요!")),
            chat_line(("날씨가 춥네", "정말 추워요. 따뜻하게 입어요."), ("그래서 뭐 먹을까", "어묵탕은 어때요?"),
                      ("좋아", "저도 좋아요."), ("또 뭐 있어", "호빵도 있어요."))]
    (tmp_path / "h.jsonl").write_text("\n".join(held) + "\n")
    monkeypatch.setattr(score, "HELDOUT", tmp_path / "h.jsonl")
    monkeypatch.setattr(score, "RESULTS", tmp_path / "r")
    from tool_data import make

    tool_rows = [r for r in make(40, 5, []) if r["topic"] in ("불빛", "부르지 않음")][:4]
    (tmp_path / "t.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in tool_rows))
    monkeypatch.setattr(score, "TOOLS_HELDOUT", tmp_path / "t.jsonl")
    wants = [q["want"] for q in score.tool_questions()]
    assert any(wants) and None in wants
    qs = score.questions()
    assert [q["set"] for q in qs] == ["첫마디", "첫마디", "첫마디", "이어지는 대화"]
    assert qs[3]["messages"][-1]["content"] == "좋아" and len(qs[3]["messages"]) == 6
    assert "학생: 날씨가 춥네\n파이보: 정말 추워요." in score.dialog(qs[3])
    assert score.echo(qs[3], "정말 추워요. 따뜻하게 입어요!") and not score.echo(qs[3], "군고구마도 맛있어요.")
    assert score.marks('<think>음</think>{"korean": 5, "fit": 4, "short": 5, "engage": 3, '
                       '"wrong": false, "repeat": true}') == {"korean": 5, "fit": 4, "short": 5,
                                                             "engage": 3, "wrong": False, "repeat": True}
    (tmp_path / "r").mkdir()
    for name, answers in (("before", ["속상했겠어요.", "속상했겠어요.", "추워요.", "정말 추워요. 따뜻하게 입어요."]),
                          ("themaker", ["많이 속상했겠어요. 무슨 일이었어요?", "수고했어요! 이제 뭐 할 거예요?",
                                        "정말 추운 날이에요.", "군고구마도 맛있어요."])):
        tools = wants if name == "themaker" else [None] * len(wants)
        (tmp_path / "r" / f"answers-{name}.json").write_text(json.dumps(
            {"answers": answers, "tools": tools}, ensure_ascii=False))
    good = {"korean": 5, "fit": 5, "short": 5, "engage": 5, "wrong": False, "repeat": False}
    (tmp_path / "r" / "scores-themaker.json").write_text(json.dumps([good] * 4))
    text = score.report()
    rows = {r.split(" | ")[0][2:] + r.split(" | ")[1]: r for r in text.splitlines()[2:] if r.startswith("|")}
    assert rows["before첫마디 (3)"].split(" | ")[10] == "67%"           # 같은 대답
    assert rows["before이어지는 대화 (1)"].split(" | ")[9] == "100%"     # 앞 말 되풀이 (글자)
    assert rows["themaker첫마디 (3)"].split(" | ")[2] == "5.00"
    assert (tmp_path / "r" / "score.md").exists()
    tool_rows = {r.split(" | ")[0][2:]: r for r in text.splitlines() if r.startswith("| before |")
                 or r.startswith("| themaker |")}
    assert tool_rows["themaker"].split(" | ")[1].startswith("100%")
    assert tool_rows["before"].split(" | ")[1].startswith("0%") and "100%" in tool_rows["before"].split(" | ")[2]
