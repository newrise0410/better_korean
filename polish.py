#!/usr/bin/env python3
"""판정 기록(labels.jsonl)을 기준으로 한국어 원고의 어색한 문장을 고친다.

  python polish.py run 원고.md --out 고친원고.md     원고를 고친다
  python polish.py review 고친원고.md --n 20          고친 줄 20개를 뽑아 보여 준다
  python polish.py record 고친원고.md --bad 3,7       review 번호 가운데 아직 어색한 번호를 기록한다

단어 목록이나 규칙은 없다. 기준은 편집자가 문장마다 내린 판정('어색' / '자연')과
확인된 수정 쌍뿐이다. record로 판정을 쌓을수록 다음 실행이 편집자의 기준에 가까워진다.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHUNK_LINES = 30


def load_labels(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def system_prompt(labels: list[dict]) -> str:
    judged = "\n".join(f"- {r['text']} → {r['label']}" for r in labels)
    pairs = "\n".join(f"- 원문: {r['text']}\n  수정: {r['fixed']}" for r in labels if r.get("fixed"))
    return (
        "당신은 한국어 원고를 고치는 편집 보조다. 아래는 이 원고의 편집자가 문장마다 직접 내린 판정이다.\n\n"
        f"{judged}\n\n"
        "아래는 편집자가 받아들인 수정이다.\n\n"
        f"{pairs}\n\n"
        "편집자는 판정의 이유를 설명하지 않았다. 판정과 수정에서 편집자의 기준을 스스로 파악한다.\n\n"
        "사용자가 원고의 한 구간을 '줄 번호<탭>내용' 형식으로 보낸다. 줄마다 편집자가 '어색'이라고 판정할 문장이 "
        "있는지 보고, 있으면 편집자가 '자연'이라고 판정할 문장으로 고친다.\n"
        "- 뜻과 정보량은 그대로 둔다. 구간에 없는 내용을 더하지 않는다. 생략된 말은 같은 구간에서 찾아 채운다.\n"
        "- 줄의 형식은 그대로 둔다. 마크다운 기호, 표의 세로선(|) 개수, 줄 맨 앞의 표지('훅:', '관점:' 등)를 바꾸지 않는다.\n"
        "- 제목은 제목 길이로, 질문은 질문으로 둔다.\n"
        "- 편집자가 '자연'이라고 판정할 줄은 건드리지 않는다. 링크, 코드, 그림 지시(svg:, flow: 등)도 건드리지 않는다.\n\n"
        "고친 줄만 '줄 번호<탭>고친 내용' 형식으로 한 줄에 하나씩 출력한다. 고칠 줄이 없으면 '없음'이라고만 출력한다."
    )


def chunks(lines: list[str]) -> list[tuple[int, int]]:
    """빈 줄에서 끊어 CHUNK_LINES줄 안팎의 구간으로 나눈다. (시작, 끝) 줄 번호, 끝은 포함하지 않는다."""
    out, start = [], 0
    for i, line in enumerate(lines):
        if i - start >= CHUNK_LINES and not line.strip():
            out.append((start, i))
            start = i
    out.append((start, len(lines)))
    return out


def run(args) -> None:
    import anthropic

    lines = args.file.read_text(encoding="utf-8").split("\n")
    first, last = (args.lines or (1, len(lines)))
    system = system_prompt(load_labels(args.labels))
    client = anthropic.Anthropic(base_url="https://api.anthropic.com", max_retries=5, timeout=600.0)
    todo = [(a, b) for a, b in chunks(lines) if b > first - 1 and a < last]
    print(f"{len(lines):,}줄 가운데 {len(todo)}개 구간을 검토합니다.")

    def work(span: tuple[int, int]) -> dict[int, str]:
        a, b = span
        body = "\n".join(f"{n + 1}\t{lines[n]}" for n in range(a, b) if lines[n].strip())
        if not body:
            return {}
        try:
            response = client.messages.create(
                model=args.model,
                max_tokens=32000,
                cache_control={"type": "ephemeral"},   # 판정 기록이 든 system은 호출마다 같으므로 캐시한다
                system=system,
                messages=[{"role": "user", "content": body}],
            )
        except anthropic.APIStatusError as e:
            print(f"  줄 {a + 1}~{b}: 오류 {e.status_code}, 원문을 유지합니다.", file=sys.stderr)
            return {}
        except anthropic.APIConnectionError:
            print(f"  줄 {a + 1}~{b}: 네트워크 오류, 원문을 유지합니다.", file=sys.stderr)
            return {}
        if response.stop_reason != "end_turn":
            return {}
        changed = {}
        for row in "".join(x.text for x in response.content if x.type == "text").split("\n"):
            m = re.match(r"\s*(\d+)\t(.*)$", row)
            if not m:
                continue
            n, new = int(m.group(1)) - 1, m.group(2)
            # 구간 밖의 줄, 내용이 같은 줄, 표 구조가 달라진 줄은 받지 않는다.
            if a <= n < b and new.strip() and new != lines[n] and new.count("|") == lines[n].count("|"):
                changed[n] = new
        return changed

    changes: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(work, todo):
            changes.update(result)

    out_lines = [changes.get(n, line) for n, line in enumerate(lines)]
    args.out.write_text("\n".join(out_lines), encoding="utf-8")
    log = [{"line": n + 1, "before": lines[n], "after": new} for n, new in sorted(changes.items())]
    log_path = args.out.with_suffix(".changes.json")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(log)}줄을 고쳤습니다 -> {args.out} (바뀐 줄 목록: {log_path})")


def sample(args) -> list[dict]:
    log = json.loads(args.file.with_suffix(".changes.json").read_text(encoding="utf-8"))
    random.Random(args.seed).shuffle(log)
    return log[: args.n]


def review(args) -> None:
    for i, row in enumerate(sample(args), 1):
        print(f"{i}. (줄 {row['line']})\n   원문: {row['before'].strip()}\n   수정: {row['after'].strip()}\n")
    print("아직 어색한 번호를 골라 'python polish.py record 파일 --bad 번호,번호'로 기록하세요. 모두 괜찮으면 --bad 없이 실행합니다.")


def record(args) -> None:
    bad = {int(x) for x in args.bad.split(",") if x.strip()} if args.bad else set()
    rows = sample(args)
    strip = lambda s: re.sub(r"^[\s|>*#-]+|[\s|]+$", "", s)
    with args.labels.open("a", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            if i in bad:
                entry = {"text": strip(row["after"]), "label": "어색"}
            else:
                entry = {"text": strip(row["before"]), "label": "어색", "fixed": strip(row["after"]), "fixed_by": "model, 사용자 확인"}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"판정 {len(rows)}개를 {args.labels}에 기록했습니다 (아직 어색함 {len(bad)}개). 통과율 {100 * (len(rows) - len(bad)) / max(1, len(rows)):.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=ROOT / "labels.jsonl")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("file", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--lines", type=int, nargs=2, metavar=("시작", "끝"), help="이 줄 범위만 검토한다")
    p.add_argument("--workers", type=int, default=4)
    for name in ("review", "record"):
        p = sub.add_parser(name)
        p.add_argument("file", type=Path, help="run의 --out 파일")
        p.add_argument("--n", type=int, default=20)
        p.add_argument("--seed", type=int, default=0)
        if name == "record":
            p.add_argument("--bad", default="", help="review 번호 가운데 아직 어색한 번호 (쉼표로 구분)")
    args = parser.parse_args()
    {"run": run, "review": review, "record": record}[args.command](args)


if __name__ == "__main__":
    main()
