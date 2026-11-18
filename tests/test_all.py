#!/usr/bin/env python3
"""API 키 없이 돌릴 수 있는 검증. 실행: python tests/test_all.py

generate.py 검증은 이 컴퓨터 안에 띄운 가짜 서버로 요청 내용만 확인한다.
가짜 서버의 응답은 시험용 문장이고 실험 결과가 아니다.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import analyze  # noqa: E402

LEX = analyze.Lexicon(ROOT / "lexicon.json")
DIRECTIONAL = {"A가 B보다 높습니다.", "A가 B보다 낮습니다."}


def counts(text: str) -> dict:
    doc = analyze.build_doc("A", "all", "t", text, LEX)
    return {raw: len(pos) for raw, pos in doc.hits.items()}


def test_homonyms_are_separated():
    c = counts("행의 개수를 센다. 세 가지가 있다. 힘이 세다. 21세기다.")
    assert c.get("세/VV") == 1, c
    c = counts("자료를 둘로 가른다. 학생을 가르친다.")
    assert c.get("가르/VV") == 1, c
    c = counts("그 자리에 선다. 서울에서는 선을 넘지 않는다.")
    assert c.get("서/VV") == 1 and c.get("자리/NNG") == 1, c
    c = counts("해결할 길이 없다. 길이가 길다.")
    assert c.get("길/NNG") == 1, c
    c = counts("깜짝 놀랐다. 짝을 짓는다. 오른쪽으로 간다. 어느 쪽이 나은지 본다.")
    assert c.get("짝/NNG") == 1 and c.get("쪽/NNB") == 1, c
    c = counts("결국 이익인 셈이다. 이자를 셈한다.")
    assert c.get("셈/NNB") == 1 and c.get("셈/NNG") == 1, c


def test_sequence_and_exclusion_entries():
    c = counts("두 알고리즘의 빠르기를 견준다. 속도가 빠르다.")
    assert c.get("빠르/VA+기/ETN!때문/NNB") == 1 and c.get("속도/NNG") == 1 and c.get("견주/VV") == 1, c
    assert "빠르/VA+기/ETN!때문/NNB" not in counts("디스크에서 읽는 일이 훨씬 빠르기 때문에 금방 뜬다.")
    c = counts("무리 지어 다닌다. 무리한 요구다.")
    assert c.get("무리/NNG!하/XS*") == 1, c


def test_markdown_is_removed():
    text = "## 제목\n- 까닭을 `까닭`으로 쓴다.\n```\n까닭 = 1\n```\nhttps://example.com/까닭"
    assert counts(text).get("까닭/NNG") == 1


def test_chunks_keep_sentences():
    text = " ".join(f"문장 {i}번은 이렇게 끝난다." for i in range(100))
    chunks = analyze.split_chunks(text, 50)
    assert len(chunks) == 8 and all(c.endswith(".") for c in chunks), len(chunks)
    assert analyze.split_chunks(text, 0) == [text]


def simulate(rng, lam_a: float, lam_b: float, strata: int = 6, per_cell: int = 5):
    """층마다 설정별 문서 per_cell개. 문서마다 어절 200개, 표지 횟수는 포아송 분포."""
    is_a = np.tile(np.repeat([True, False], per_cell), strata)
    strat = np.repeat(np.arange(strata), 2 * per_cell)
    num = rng.poisson(np.where(is_a, lam_a, lam_b))
    return num, np.full(len(num), 200.0), is_a, strat


def test_no_difference_is_rarely_called_a_difference():
    rng = np.random.default_rng(1)
    false_alarms, covered, runs = 0, 0, 200
    for _ in range(runs):
        r = analyze.compare_rates(*simulate(rng, 3.0, 3.0), rng, 300, 200)
        false_alarms += r["p"] < 0.05
        covered += r["ci"][0] <= 1 <= r["ci"][1]
    assert false_alarms / runs < 0.10, false_alarms
    assert covered / runs > 0.88, covered


def test_real_difference_is_detected():
    rng = np.random.default_rng(2)
    hits = sum(analyze.verdict(analyze.compare_rates(*simulate(rng, 6.0, 3.0), rng, 300, 200)) == "A가 B보다 높습니다." for _ in range(50))
    assert hits >= 45, hits


def test_equal_rates_reach_equivalence_with_enough_data():
    rng = np.random.default_rng(3)
    r = analyze.compare_rates(*simulate(rng, 8.0, 8.0, strata=12, per_cell=10), rng, 500, 500)
    assert analyze.verdict(r) == "두 묶음의 비율이 사실상 같습니다.", r


def test_too_few_hits_is_not_judged():
    rng = np.random.default_rng(4)
    r = analyze.compare_rates(*simulate(rng, 0.05, 0.05), rng, 300, 200)
    assert analyze.verdict(r).startswith("판단할 수 없습니다"), r


def make_docs(swap_in_a: bool) -> list:
    base = "이 방법은 속도가 빠르다. 그 이유는 계산 단계가 적기 때문이다. 단어의 의미를 비교한다. 자료의 구조를 바꾼다."
    docs = []
    for s in range(4):
        for k in range(6):
            for group in "AB":
                text = " ".join([base] * (3 + (k + s) % 3))
                if swap_in_a and group == "A":
                    text = text.replace("이유", "까닭").replace("단어", "낱말")
                docs.append(analyze.build_doc(group, f"p{s}", f"p{s}#{group}{k}", text, LEX))
    return docs


def run_compare(docs) -> dict:
    args = SimpleNamespace(seed=0, perm=400, boot=300, show_hits=1)
    with contextlib.redirect_stdout(io.StringIO()):
        return analyze.compare(docs, LEX, ("A", "B"), args)


def test_full_report_on_identical_and_altered_texts():
    same = run_compare(make_docs(swap_in_a=False))
    assert analyze.verdict(same["pair_share"]) not in DIRECTIONAL
    assert same["native_rate"]["num_a"] == same["native_rate"]["num_b"] == 0

    altered = run_compare(make_docs(swap_in_a=True))
    assert analyze.verdict(altered["native_rate"]) == "A가 B보다 높습니다.", altered["native_rate"]
    assert analyze.verdict(altered["pair_share"]) == "A가 B보다 높습니다.", altered["pair_share"]
    assert altered["vocabulary"]["p"] < 0.05
    assert {w["word"] for w in altered["vocabulary"]["more_a"][:2]} == {"까닭/NNG", "낱말/NNG"}


def test_probe_answers_and_distribution_test():
    assert analyze.extract_answer("37", "first_int") == "37"
    assert analyze.extract_answer("내가 고른 수는 다음과 같다: 42", "last_int") == "42"
    assert analyze.extract_answer("내가 고른 과일은 다음과 같다: 사과.", "last_word") == "사과"
    assert analyze.extract_answer("...", "first_int") is None
    rng = np.random.default_rng(5)
    same_a = list(rng.choice(["37", "42", "73"], 100, p=[0.5, 0.3, 0.2]))
    same_b = list(rng.choice(["37", "42", "73"], 100, p=[0.5, 0.3, 0.2]))
    other = list(rng.choice(["37", "42", "73"], 100, p=[0.2, 0.3, 0.5]))
    assert analyze.tv_test(same_a, same_b, rng, 1000)[1] > 0.05
    assert analyze.tv_test(same_a, other, rng, 1000)[1] < 0.01


class FakeApi(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        FakeApi.seen.append(({k.lower(): v for k, v in self.headers.items()}, self.path, body))
        payload = json.dumps({
            "id": "msg_test", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": "시험용 응답입니다."}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("request-id", "req_test")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_generate_sends_the_intended_requests():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeApi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("ANTHROPIC_", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"))}
    env |= {"ANTHROPIC_API_KEY": "test-key", "NO_PROXY": "127.0.0.1,localhost"}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "main.jsonl"
        command = [sys.executable, str(ROOT / "generate.py"), "main", "--n", "2", "--limit-prompts", "2",
                   "--out", str(out), "--base-url", f"http://127.0.0.1:{server.server_port}"]
        first = subprocess.run(command, env=env, capture_output=True, text=True)
        assert first.returncode == 0, first.stderr
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        again = subprocess.run(command, env=env, capture_output=True, text=True)
        assert again.returncode == 0, again.stderr
        mixed = subprocess.run(command + ["--model", "claude-sonnet-4-6"], env=env, capture_output=True, text=True)
        assert mixed.returncode != 0 and "다른 모델" in mixed.stderr, mixed.stderr   # 모델이 섞이지 않게 막는다
    server.shutdown()

    assert len(FakeApi.seen) == 8, len(FakeApi.seen)          # 다시 실행했을 때 새 호출이 없어야 한다
    assert sorted(r["arm"] for r in rows) == ["A"] * 4 + ["B"] * 4
    suffix = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8"))["suffix"]
    with_temperature = 0
    for headers, path, body in FakeApi.seen:
        assert path == "/v1/messages", path
        assert headers["anthropic-version"] == "2023-06-01", headers
        assert headers["x-api-key"] == "test-key"
        assert not {"top_p", "top_k", "thinking"} & set(body), body
        assert body["messages"][0]["content"].endswith(suffix)
        if "temperature" in body:
            assert body["temperature"] == 0.99
            with_temperature += 1
    assert with_temperature == 4, with_temperature           # 설정 B만 temperature를 보낸다
    assert all(r["params"] == ({"temperature": 0.99} if r["arm"] == "B" else None) for r in rows)


class RejectsSampling(BaseHTTPRequestHandler):
    """샘플링 파라미터가 있는 요청을 400으로 거부하는 서버 (4.7 이후 모델의 동작)."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if "temperature" in body:
            status, reply = 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "temperature is not supported"}}
        else:
            status, reply = 200, {
                "id": "msg_test", "type": "message", "role": "assistant", "model": body["model"],
                "content": [{"type": "text", "text": "시험용 응답입니다."}],
                "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        payload = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_generate_stops_when_the_model_rejects_sampling_parameters():
    server = ThreadingHTTPServer(("127.0.0.1", 0), RejectsSampling)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("ANTHROPIC_", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"))}
    env |= {"ANTHROPIC_API_KEY": "test-key", "NO_PROXY": "127.0.0.1,localhost"}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "main.jsonl"
        done = subprocess.run(
            [sys.executable, str(ROOT / "generate.py"), "main", "--n", "3", "--limit-prompts", "2", "--workers", "1",
             "--out", str(out), "--base-url", f"http://127.0.0.1:{server.server_port}"],
            env=env, capture_output=True, text=True,
        )
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()] if out.exists() else []
    server.shutdown()
    assert done.returncode != 0
    assert "요청이 거부됐습니다" in done.stderr, done.stderr
    assert all(r["arm"] == "A" for r in rows) and len(rows) < 6, rows   # 첫 거부 뒤에는 더 호출하지 않는다


if __name__ == "__main__":
    tests =[(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"통과  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"실패  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 통과")
    sys.exit(1 if failed else 0)
