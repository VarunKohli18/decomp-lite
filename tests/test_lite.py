"""Tests. The first block needs nothing but Python; the last block needs the
x86-64 toolchain (and Ghidra for the pseudo-C test) and skips without them."""
import json
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from decomplite import benchmark as bench
from decomplite import build as B
from decomplite import config, llm
from decomplite import evaluate as E
from decomplite import representations as R
from decomplite.datasets import load_dataset, select

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Send every artifact to a temp dir instead of the repo's data/ results/."""
    monkeypatch.setattr("decomplite.datasets.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("decomplite.build.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("decomplite.representations.DATA_DIR", tmp_path / "data")
    monkeypatch.setattr("decomplite.benchmark.RESULTS_DIR", tmp_path / "results")
    return tmp_path


# --- config ------------------------------------------------------------------

def make_config(tmp_path, pipelines, **top):
    """A config over the shipped datasets and models with the given pipelines."""
    import yaml
    raw = {"opts": ["O0", "O2"], "model": "vllm-gemma4-31b-it", "pipelines": pipelines,
           "repos": [{"name": "example"}, {"name": "example_jsonl"}],
           "models_file": str(ROOT / "models.yaml"), **top}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))
    return config.load(path)


def test_shipped_config_loads():
    cfg = config.load(ROOT / "benchmark.yaml")
    assert cfg.archs == ("x86_64",)
    assert cfg.opts == ("O0", "O1", "O2", "O3", "Os")
    assert [r.name for r in cfg.repos] == ["example"]
    assert cfg.repos[0].functions[0] == "sum_array"
    # The one shipped pipeline is the simplest one.
    (pl,) = cfg.pipelines
    assert (pl.name, pl.input, pl.recovery, pl.reviewers, pl.feedback, pl.iterations) == \
        ("single_pass", ("asm",), (None,), (), "none", 1)
    # Left out, every setting but `input` takes that same default.
    assert config.parse_pipeline(ROOT, "p", {"input": "asm"}, {}).spec("m") == pl.spec("m")


def test_pipeline_roles_and_spec(tmp_path):
    cfg = make_config(tmp_path, {"p": {
        "input": ["pseudoc", "asm"], "recovery": 2, "reviewers": ["ollama-gemma4-12b"],
        "feedback": "tests", "iterations": 3}})
    pl = cfg.pipelines[0]
    assert pl.input == ("asm", "pseudoc")                      # canonical order
    assert pl.spec("M") == {"input": ["asm", "pseudoc"], "recovery": ["M", "M"],
                            "reviewers": ["ollama-gemma4-12b"], "feedback": "tests",
                            "iterations": 3}


@pytest.mark.parametrize("top,pipelines,needle", [
    ({"archs": ["aarch64"]}, {"p": {"input": "asm"}}, "unknown archs"),
    ({"opts": ["O9"]}, {"p": {"input": "asm"}}, "unknown opts"),
    ({"model": "nope"}, {"p": {"input": "asm"}}, "unknown model"),
    ({}, None, "`pipelines` is required"),
    ({}, ["inputrepr"], "`pipelines` is required"),
    ({}, {"p": {}}, "`input` must be"),
    ({}, {"p": {"input": "hex"}}, "`input` must be"),
    ({}, {"p": {"input": "asm", "feedback": "vibes"}}, "unknown feedback"),
    ({}, {"p": {"input": "asm", "iterations": 0}}, "`iterations` must be"),
    ({}, {"p": {"input": "asm", "recovery": 0}}, "at least one recovery"),
    ({}, {"p": {"input": "asm", "reviewers": ["ghost"]}}, "unknown model 'ghost'"),
    ({}, {"p": {"input": "asm", "reviewer": 1}}, "unknown key"),
    ({}, {"p": {"input": "asm", "recovery": 2}}, "need `feedback` or a reviewer"),
    ({}, {"p": {"input": "asm", "iterations": 2}}, "needs `feedback` or a reviewer"),
    ({}, {"a/b": {"input": "asm"}}, "filename-safe"),
])
def test_config_rejects(tmp_path, top, pipelines, needle):
    with pytest.raises(ValueError, match=needle):
        make_config(tmp_path, pipelines, **top)


# --- datasets ----------------------------------------------------------------

def test_three_dataset_kinds(sandbox):
    d = load_dataset(config.RepoSelection("example"))
    # `<name>_test.c` is the sample's test driver, not a sample of its own.
    assert set(d) == {"sum_array", "count_bits", "reverse_string", "clamp_scale"}
    assert d["sum_array"].test_path.name == "sum_array_test.c" and d["sum_array"].io_path is None

    j = load_dataset(config.RepoSelection("example_jsonl"))
    assert j["str_hash_djb2"].function == "hash"
    assert "5381" in j["str_hash_djb2"].source_path.read_text()
    assert j["gcd"].test_path.name == "gcd_test.c" and j["str_hash_djb2"].test_path is None

    a = load_dataset(config.RepoSelection("example_adapter", cflags=("-DX",)))
    assert a["abs_diff"].cflags == ("-DX",)


def test_select():
    samples = load_dataset(config.RepoSelection("example"))
    chosen, missing = select(
        config.RepoSelection("example", functions=("sum_array", "ghost"), patterns=("c*",)),
        samples)
    assert [s.name for s in chosen] == ["clamp_scale", "count_bits", "sum_array"]
    assert missing == ["ghost"]
    assert len(select(config.RepoSelection("example"), samples)[0]) == 4


# --- asm cleaning, prompt, fences ---------------------------------------------

OBJDUMP = """
libx.stripped.so:     file format elf64-x86-64


Disassembly of section .text:

0000000000001119 <.text+0x19>:
    1119:\tf3 0f 1e fa          \tendbr64
    111d:\te8 2e ff ff ff       \tcall   1050 <strlen@plt>
    1122:\t74 05                \tje     1129 <.text+0x29>
    1123:\t75 e7                \tjne    1140 <strlen@plt+0xf0>
    1124:\t48 8b 05 d5 2e 00 00 \tmov    0x2ed5(%rip),%rax        # 4000 <.data+0x8>
    112b:\tc3                   \tret
"""


def test_clean_asm():
    lines = R.clean_asm(OBJDUMP, 0x1119).splitlines()
    assert lines[0] == "0000000000001119 <sub_1119>:"
    assert len(lines) == 7
    assert lines[2].endswith("call   1050 <strlen@plt>")      # import name kept
    assert lines[3].endswith("je     1129")                   # section-relative noise dropped
    assert lines[4].endswith("jne    1140")
    assert lines[5].endswith("# 4000")
    assert R.clean_asm("no instructions here", 0) == ""


def test_prompt_fills_only_selected_inputs():
    system, user = bench.load_prompt("recover")
    contents = {"asm": "ASM_TEXT", "pseudoc": "PSEUDO_TEXT"}
    p = bench.render_prompt(user, ("asm",), contents)
    assert "ASM_TEXT" in p and "PSEUDO_TEXT" not in p and "{{" not in p
    assert "x86-64 (System V AMD64 ABI)" in p
    both = bench.render_prompt(user, ("asm", "pseudoc"), contents)
    assert "ASM_TEXT" in both and "PSEUDO_TEXT" in both
    assert "{{" not in bench.render_prompt(system, ("asm",), contents)


def test_prompt_slots_are_filled_in_one_pass():
    """C text containing something slot-shaped must not be rescanned."""
    for name in ("repair", "review"):
        _, user = bench.load_prompt(name)
        p = bench.render_prompt(user, ("pseudoc",), {"pseudoc": "P"}, {
            "GENERATED_C": "int a[1][1] = {{PSEUDOC}};", "FEEDBACK_TITLE": "T", "FEEDBACK": "E"})
        assert "int a[1][1] = {{PSEUDOC}};" in p and p.count("{{") == 1


def test_strip_fences():
    assert bench.strip_fences("x\n```c\nint f(void);\n```\ny") == "int f(void);"
    assert bench.strip_fences("```\nint g;\n") == "int g;"
    assert bench.strip_fences("  int h;  ") == "int h;"


# --- LLM transport + the decompile phase, against a stub server ---------------

@pytest.fixture
def stub_server():
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body, self.headers.get("Authorization"),
                         self.headers.get("anthropic-beta") or ""))
            if self.path == "/v1/chat/completions":
                user = body["messages"][-1]["content"]
                served = body["model"]
                code = ("int reviewed(void){return 3;}" if "Reconstruction under review" in user
                        else "int repaired(void){return 4;}" if "failed the check" in user
                        else "int f(void){return 1;}" if served == "m"
                        else "int from_%s(void){return 1;}" % served)
                out = {"choices": [{"message": {"content": "```c\n%s\n```" % code}}],
                       "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
            elif self.path.startswith("/v1/messages"):
                refuse = body["messages"][-1]["content"] == "REFUSE"
                out = {"id": "msg_1", "type": "message", "role": "assistant",
                       "model": body["model"], "stop_sequence": None,
                       "stop_reason": "refusal" if refuse else "end_turn",
                       "content": [] if refuse else [
                           {"type": "thinking", "thinking": "", "signature": "s"},
                           {"type": "text", "text": "int c(void){return 5;}"}],
                       "usage": {"input_tokens": 11, "output_tokens": 4}}
            else:
                out = {"message": {"content": "int f(void){return 2;}"},
                       "prompt_eval_count": 5, "eval_count": 2}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", seen
    srv.shutdown()


def test_llm_providers(stub_server, monkeypatch):
    url, seen = stub_server
    monkeypatch.setenv("MY_KEY", "sekrit")
    r = llm.call({"provider": "vllm", "model": "m", "base_url": url + "/v1", "max_tokens": 9,
                  "api_key_env": "MY_KEY", "extra_body": {"top_k": 1}}, "SYS", "USR")
    assert r["text"].startswith("```c") and r["input_tokens"] == 7
    path, body, auth, _ = seen[-1]
    assert path == "/v1/chat/completions" and auth == "Bearer sekrit"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["max_tokens"] == 9 and body["top_k"] == 1 and body["temperature"] == 0.1

    r = llm.call({"provider": "ollama", "model": "m", "base_url": url, "num_ctx": 4096}, "", "USR")
    assert r["text"] == "int f(void){return 2;}" and r["output_tokens"] == 2
    path, body = seen[-1][:2]
    assert path == "/api/chat" and body["options"]["num_ctx"] == 4096

    # OpenAI: its own key and parameter names; no temperature unless asked for.
    cfg = {"provider": "openai", "model": "m", "base_url": url, "max_tokens": 11}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        llm.call(cfg, "SYS", "USR")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert llm.call(cfg, "SYS", "USR")["output_tokens"] == 3
    path, body, auth, _ = seen[-1]
    assert path == "/v1/chat/completions" and auth == "Bearer sk-test"
    assert body["max_completion_tokens"] == 11 and "temperature" not in body and "max_tokens" not in body

    with pytest.raises(ValueError, match="unknown provider"):
        llm.call({"provider": "gpustack", "model": "m"}, "", "")


def test_anthropic_provider(stub_server, monkeypatch):
    pytest.importorskip("anthropic", minversion="2")
    url, seen = stub_server
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    for name in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    cfg = {"provider": "anthropic", "model": "claude-opus-5-5", "base_url": url,
           "max_tokens": 1234, "effort": "low"}
    r = llm.call(cfg, "SYS", "USR")
    assert r["text"] == "int c(void){return 5;}" and (r["input_tokens"], r["output_tokens"]) == (11, 4)
    assert r["served_by"] == "claude-opus-5-5"
    path, body = seen[-1][:2]
    assert path.startswith("/v1/messages") and body["system"] == "SYS"
    assert body["messages"] == [{"role": "user", "content": "USR"}]
    assert body["max_tokens"] == 1234 and body["output_config"] == {"effort": "low"}
    assert "temperature" not in body and "fallbacks" not in body

    llm.call(dict(cfg, fallbacks="default"), "", "USR")
    path, body = seen[-1][:2]
    assert body["fallbacks"] == "default" and "system" not in body
    assert "server-side-fallback-2026-07-01" in seen[-1][3]

    with pytest.raises(RuntimeError, match="declined the request"):
        llm.call(cfg, "", "REFUSE")


def fake_run(tmp_path, monkeypatch, url, pipelines, passes=lambda text: True):
    """A config whose models all hit the stub server, with build and input
    extraction stubbed out. `passes(text)` is the feedback check's verdict."""
    cfg = make_config(tmp_path, pipelines)
    for key in cfg.model_registry:
        cfg.model_registry[key] = {"provider": "vllm", "base_url": url,
                                   "model": "m" if key == cfg.models[0] else "other"}
    monkeypatch.setattr(B, "build", lambda sample, opt, force=False: B.Build(
        repo=sample.repo, name=sample.name, opt=opt, ok=sample.name != "gcd",
        error="compile failed", stripped="x.so", function=sample.name))
    monkeypatch.setattr(R, "materialize", lambda b, chans, force=False: {
        c: ("" if (c == "pseudoc" and b.opt == "O0" and b.name == "count_bits")
            else f"<{c} {b.name} {b.opt}>") for c in chans})
    monkeypatch.setattr(R, "channel_error", lambda b, c: "ghidra exploded")
    monkeypatch.setattr(E, "feedback", lambda kind, sample, text, fn, opt: {
        "ok": passes(text), "title": "Compiler diagnostics", "source": kind,
        "text": "" if passes(text) else "candidate.c:1: error: BOOM"})
    return cfg


def cell(cfg, pipeline, fn="sum_array", opt="O2"):
    return bench.resolve(cfg, {"fn": fn, "opt": opt, "pipeline": pipeline})[2][0]


def roles(c):
    return [(t["role"], t["iteration"]) for t in c.meta()["turns"]]


SIMPLE = {"single": {"input": "pseudoc"}}


def test_simplest_pipeline_is_one_call_per_cell(sandbox, stub_server, monkeypatch):
    url, seen = stub_server
    cfg = fake_run(sandbox, monkeypatch, url, SIMPLE)
    cells = bench.resolve(cfg, {"fn": "sum_array,count_bits,gcd"})[2]
    assert len(cells) == 3 * 2 * 1 * 1
    bench.phase_decompile(cfg, cells, force=False, workers=2)

    ok = cell(cfg, "single")
    assert ok.out_path.read_text() == "int f(void){return 1;}\n"
    assert "results/single/vllm-gemma4-31b-it/example/x86_64-O2" in ok.out_path.as_posix()
    meta = ok.meta()
    assert meta["status"] == "ok" and meta["llm_calls"] == 1 and roles(ok) == [("recovery 1", 1)]
    assert meta["spec"]["recovery"] == ["vllm-gemma4-31b-it"]
    assert "<pseudoc sum_array O2>" in meta["turns"][0]["user_prompt"]
    assert "<asm" not in meta["turns"][0]["user_prompt"]

    # No input / failed build: recorded, no output, no LLM call.
    for c, needle in ((cell(cfg, "single", "count_bits", "O0"), "ghidra exploded"),
                      (cell(cfg, "single", "gcd"), "build failed")):
        assert not c.out_path.exists() and needle in c.meta()["error"]
    assert len(seen) == 3          # sum_array O0+O2, count_bits O2

    # Second run: finished cells are cached, failed ones are retried.
    bench.phase_decompile(cfg, cells, force=False, workers=1)
    assert len(seen) == 3


def test_reviewer_and_shared_first_answer(sandbox, stub_server, monkeypatch):
    url, seen = stub_server
    cfg = fake_run(sandbox, monkeypatch, url, {
        **SIMPLE, "review": {"input": "pseudoc", "reviewers": 1},
        "review_x3": {"input": "pseudoc", "reviewers": 2, "iterations": 2}})
    review = cell(cfg, "review")
    assert bench.decompile_cell(cfg, review, force=False) == "ok"
    assert len(seen) == 2 and roles(review) == [("recovery 1", 1), ("reviewer 1", 1)]
    assert review.out_path.read_text() == "int reviewed(void){return 3;}\n"
    user = review.meta()["turns"][1]["user_prompt"]
    assert "int f(void){return 1;}" in user and "<pseudoc sum_array O2>" in user
    assert "{{" not in user and "colleague" in review.meta()["turns"][1]["system_prompt"]

    # Another pipeline that starts the same way starts from the SAME answer,
    # without paying for it again.
    single = cell(cfg, "single")
    assert bench.decompile_cell(cfg, single, force=False) == "ok"
    assert len(seen) == 2 and single.meta()["turns"][0]["reused"] is True
    assert single.out_path.read_text() == "int f(void){return 1;}\n"

    # 2 reviewers x 2 iterations = 4 review calls on top of the shared answer.
    x3 = cell(cfg, "review_x3")
    assert bench.decompile_cell(cfg, x3, force=False) == "ok"
    assert len(seen) == 6
    assert roles(x3) == [("recovery 1", 1), ("reviewer 1", 1), ("reviewer 2", 1),
                         ("reviewer 1", 2), ("reviewer 2", 2)]


def test_feedback_repairs_only_while_the_check_fails(sandbox, stub_server, monkeypatch):
    url, seen = stub_server
    recipe = {"fix": {"input": "asm", "feedback": "compiler", "iterations": 3}}

    # Passes first time: no extra call.
    cfg = fake_run(sandbox, monkeypatch, url, recipe)
    c = cell(cfg, "fix")
    assert bench.decompile_cell(cfg, c, force=False) == "ok"
    assert len(seen) == 1 and roles(c) == [("recovery 1", 1)]
    assert c.meta()["stopped"] == "passed the compiler check before iteration 2"
    assert c.meta()["feedback_passed"] is True

    # Fails, the repair passes: one repair call, shown the diagnostics.
    cfg = fake_run(sandbox, monkeypatch, url, recipe, passes=lambda text: "repaired" in text)
    c = cell(cfg, "fix", "count_bits")
    assert bench.decompile_cell(cfg, c, force=False) == "ok"
    assert roles(c) == [("recovery 1", 1), ("repair", 2)]
    user = c.meta()["turns"][1]["user_prompt"]
    assert "error: BOOM" in user and "==== Compiler diagnostics ====" in user
    assert "<asm count_bits O2>" in user and "int f(void){return 1;}" in user and "{{" not in user
    assert c.out_path.read_text() == "int repaired(void){return 4;}\n"

    # Never passes: 3 iterations = the first answer + 2 repairs, then it stops.
    cfg = fake_run(sandbox, monkeypatch, url, recipe, passes=lambda text: False)
    c = cell(cfg, "fix", "clamp_scale")
    assert bench.decompile_cell(cfg, c, force=False) == "ok"
    assert roles(c) == [("recovery 1", 1), ("repair", 2), ("repair", 3)]
    assert c.meta()["stopped"] == "after iteration 3" and c.meta()["feedback_passed"] is False


def test_several_recovery_llms(sandbox, stub_server, monkeypatch):
    url, seen = stub_server
    # Two different models; the check only accepts the second one's answer.
    cfg = fake_run(sandbox, monkeypatch, url, {
        "pick": {"input": "asm", "recovery": ["vllm-gemma4-31b-it", "ollama-gemma4-12b"],
                 "feedback": "compiler"},
        "judge": {"input": "asm", "recovery": 2, "reviewers": 1}},
        passes=lambda text: "from_other" in text)
    pick = cell(cfg, "pick")
    assert bench.decompile_cell(cfg, pick, force=False) == "ok"
    assert roles(pick) == [("recovery 1", 1), ("recovery 2", 1)]
    assert pick.out_path.read_text() == "int from_other(void){return 1;}\n"

    # Same model twice = two separate answers; the reviewer sees both.
    judge = cell(cfg, "judge")
    assert bench.decompile_cell(cfg, judge, force=False) == "ok"
    assert roles(judge) == [("recovery 1", 1), ("recovery 2", 1), ("reviewer 1", 1)]
    assert judge.meta()["turns"][0]["reused"] is True          # shared with `pick`
    assert "reused" not in judge.meta()["turns"][1]
    user = judge.meta()["turns"][2]["user_prompt"]
    assert "---- Candidate 1 of 2 ----" in user and "---- Candidate 2 of 2 ----" in user


def test_changed_recipe_is_stale_not_reused(sandbox, stub_server, monkeypatch):
    url, seen = stub_server
    cfg = fake_run(sandbox, monkeypatch, url, {"p": {"input": "pseudoc"}})
    assert bench.decompile_cell(cfg, cell(cfg, "p"), force=False) == "ok"
    # Same name, different recipe: neither reused nor overwritten...
    cfg = fake_run(sandbox, monkeypatch, url, {"p": {"input": "pseudoc", "reviewers": 1}})
    c = cell(cfg, "p")
    assert not c.is_current()
    assert bench.decompile_cell(cfg, c, force=False) == "stale"
    assert c.out_path.read_text() == "int f(void){return 1;}\n"
    # ...until --force.
    assert bench.decompile_cell(cfg, c, force=True) == "ok"
    assert c.is_current() and c.out_path.read_text() == "int reviewed(void){return 3;}\n"


# --- real toolchain ------------------------------------------------------------

def _have_toolchain():
    try:
        B.toolchain()
        return True
    except B.ToolchainError:
        return False


needs_toolchain = pytest.mark.skipif(not _have_toolchain(), reason="no x86-64 Linux toolchain")
needs_ghidra = pytest.mark.skipif(
    not os.environ.get("GHIDRA_INSTALL_DIR") or shutil.which("java") is None,
    reason="GHIDRA_INSTALL_DIR not set")


@needs_toolchain
@pytest.mark.parametrize("opt", config.OPT_LEVELS)
def test_build_and_asm_every_opt(sandbox, opt):
    samples = load_dataset(config.RepoSelection("example"))
    for name in samples:
        b = B.build(samples[name], opt)
        assert b.ok, b.error
        assert b.function == name and b.size > 0
        asm = R.materialize(b, ["asm"])["asm"]
        assert asm.startswith("%016x <sub_%x>:" % (b.vaddr, b.vaddr))
        assert "ret" in asm
        assert name not in asm                       # stripped: no symbol name leaks
    # Cached on the second call.
    assert B.build(samples["sum_array"], opt).ok


@needs_toolchain
def test_symbol_resolution_and_failures(sandbox):
    j = load_dataset(config.RepoSelection("example_jsonl"))
    assert B.build(j["str_hash_djb2"], "O2").function == "hash"

    bad = sandbox / "bad"
    bad.mkdir()
    (bad / "broken.c").write_text("int broken( {")
    (bad / "two.c").write_text("int a(void){return 1;}\nint b(void){return 2;}\n")
    (bad / "solo.c").write_text("int other_name(int x){return x+1;}\n")
    d = load_dataset(config.RepoSelection("bad", path=bad))
    assert "compile failed" in B.build(d["broken"], "O0").error
    assert "cannot tell which function" in B.build(d["two"], "O0").error
    assert B.build(d["solo"], "O0").function == "other_name"


@needs_toolchain
@needs_ghidra
def test_pseudoc(sandbox):
    samples = load_dataset(config.RepoSelection("example"))
    b = B.build(samples["reverse_string"], "O2")
    got = R.materialize(b, ["pseudoc"])
    assert got["pseudoc"], R.channel_error(b, "pseudoc")
    assert "FUN_" in got["pseudoc"] and "reverse_string" not in got["pseudoc"]


needs_harness = pytest.mark.skipif(
    not _have_toolchain() or not B.Toolchain("", "", "").can_execute(),
    reason="needs an x86-64 Linux host to build and run test harnesses")


@needs_harness
def test_rc_rci_re_on_the_example_dataset(sandbox):
    pytest.importorskip("tree_sitter_c")
    s = load_dataset(config.RepoSelection("example"))["sum_array"]
    ref = s.source_path.read_text()

    # The reference itself, under another name: renamed, builds, passes.
    good = E.harness(s, ref.replace("sum_array", "FUN_00101119"), "sum_array")
    assert (good["compile_ok"], good["test_passed"], good["status"]) == (1, 1, "ok")
    assert (good["test_pass"], good["test_total"]) == (1, 1)

    # Right interface, wrong behaviour: RCI passes, RE fails.
    wrong = E.harness(s, "int f(const int *v, int n) { return n; }", "sum_array")
    assert (wrong["compile_ok"], wrong["test_passed"]) == (1, 0)

    # Compiles alone (RC) but not against its callers (RCI): wrong arity.
    arity = "int f(const int *v) { return v[0]; }"
    assert E.standalone_compile(arity, "O2")[0]
    res = E.harness(s, arity, "sum_array")
    assert res["compile_ok"] == 0 and res["test_passed"] == 0
    assert "sum_array" in res["compile_error"]
    fb = E.feedback("interface", s, arity, "sum_array", "O2")
    assert (fb["ok"], fb["source"], fb["title"]) == (False, "interface", "Compiler diagnostics")
    assert "too many arguments" in fb["text"]
    # `compiler` feedback cannot see that; `tests` reports behaviour.
    assert E.feedback("compiler", s, arity, "sum_array", "O2")["ok"]
    fb = E.feedback("tests", s, "int f(const int *v, int n) { return n; }", "sum_array", "O2")
    assert (fb["ok"], fb["source"], fb["title"]) == (False, "tests", "Test results")
    assert "0/1 passed" in fb["text"]
    assert E.feedback("tests", s, ref, "sum_array", "O2") == {
        "ok": True, "title": "", "text": "", "source": "tests"}

    # Not C at all: fails everything.
    assert not E.standalone_compile("int broken( {", "O0")[0]
    assert E.harness(s, "int broken( {", "sum_array")["compile_ok"] == 0

    # A sample without a harness: RC only; feedback falls back to the compiler.
    j = load_dataset(config.RepoSelection("example_jsonl"))["str_hash_djb2"]
    assert E.harness(j, ref, "hash")["status"] == "no_harness"
    fb = E.feedback("tests", j, "int f(void) { return g(); }", "hash", "O0")
    assert (fb["ok"], fb["source"]) == (False, "compiler") and "implicit declaration" in fb["text"]


@needs_harness
def test_io_pair_protocol(sandbox):
    """ExeBench's layout: `<name>_test.*` run once per pair of `<name>_io.json`."""
    pytest.importorskip("tree_sitter_c")
    d = sandbox / "io"
    d.mkdir()
    (d / "twice.c").write_text("int twice(int x)\n{\n    return 2 * x;\n}\n")
    (d / "twice_test.c").write_text(
        '#include "twice.c"\n'
        "int main(int argc, char **argv)\n{\n"
        '    int x = 0; FILE *in = fopen(argv[1], "r");\n'
        '    if (!in || fscanf(in, "{\\"x\\": %d}", &x) != 1) return 1;\n'
        '    FILE *out = fopen(argv[2], "w");\n'
        '    fprintf(out, "{\\"ret\\": %d}", twice(x));\n'
        "    return 0;\n}\n")
    (d / "twice_io.json").write_text(json.dumps(
        [{"input": {"x": i}, "output": {"ret": 2 * i}} for i in range(5)]))
    s = load_dataset(config.RepoSelection("io", path=d))["twice"]
    assert s.io_path.name == "twice_io.json"

    good = E.harness(s, "int g(int a) { return a + a; }", "twice")
    assert (good["test_pass"], good["test_total"], good["test_passed"]) == (5, 5, 1)
    # x*x agrees with 2x only at 0 and 2: a partial pass rate, RE = 0.
    part = E.harness(s, "int g(int a) { return a * a; }", "twice")
    assert (part["test_pass"], part["test_total"], part["test_passed"]) == (2, 5, 0)
    # An infinite loop is a failed test, not a hung eval.
    monkey = E.IO_TIMEOUT_S
    E.IO_TIMEOUT_S = 1
    try:
        hang = E.harness(s, "int g(int a) { for (;;) a++; return a; }", "twice")
    finally:
        E.IO_TIMEOUT_S = monkey
    assert hang["compile_ok"] == 1 and hang["test_passed"] == 0
