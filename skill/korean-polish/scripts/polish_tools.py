#!/usr/bin/env python3
"""korean-polish 스킬의 보조 도구. 표준 라이브러리만 쓴다.

  chunks 파일                       구간을 '시작줄 끝줄'로 출력
  brief 파일 시작줄 끝줄            하위 에이전트에게 줄 지시문 출력
  apply 파일 답.tsv ... --out 새파일  하위 에이전트의 답을 반영
  sample 새파일 --n 20              바뀐 줄을 뽑아 출력
  record 새파일 --n 20 --bad 3,7    사용자의 확인을 판정 기록에 추가
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parent.parent
BASE_LABELS = SKILL / "labels.jsonl"
PROJECT_LABELS = Path(".korean-polish") / "labels.jsonl"
CHUNK_LINES = 30


def load_labels() -> list[dict]:
    rows = []
    for path in (BASE_LABELS, PROJECT_LABELS):
        if path.exists():
            rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").split("\n")


def chunk_spans(lines: list[str]) -> list[tuple[int, int]]:
    """빈 줄에서 끊어 CHUNK_LINES줄 안팎으로 나눈다. 줄 번호는 1부터, 끝줄 포함."""
    spans, start = [], 0
    for i, line in enumerate(lines):
        if i - start >= CHUNK_LINES and not line.strip():
            spans.append((start + 1, i))
            start = i
    spans.append((start + 1, len(lines)))
    return spans


def brief(path: Path, first: int, last: int) -> str:
    labels = load_labels()
    lines = read_lines(path)
    judged = "\n".join(f"- {r['text']} → {r['label']}" for r in labels)
    pairs = "\n".join(f"- 원문: {r['text']}\n  수정: {r['fixed']}" for r in labels if r.get("fixed"))
    body = "\n".join(f"{n}\t{lines[n - 1]}" for n in range(first, min(last, len(lines)) + 1) if lines[n - 1].strip())
    return (
        "한국어 원고의 한 구간을 고치는 일이다. 다른 파일을 읽거나 도구를 쓸 필요가 없다. 아래 내용만으로 답한다.\n\n"
        "아래는 이 원고의 편집자가 문장마다 직접 내린 판정이다.\n\n"
        f"{judged}\n\n"
        "아래는 편집자가 받아들인 수정이다.\n\n"
        f"{pairs}\n\n"
        "편집자는 판정의 이유를 설명하지 않았다. 판정과 수정에서 편집자의 기준을 스스로 파악한다.\n\n"
        "아래 구간의 줄마다 편집자가 '어색'이라고 판정할 문장이 있는지 보고, 있으면 편집자가 '자연'이라고 판정할 문장으로 고친다.\n"
        "- 뜻과 정보량은 그대로 둔다. 구간에 없는 내용을 더하지 않는다. 생략된 말은 같은 구간에서 찾아 채운다.\n"
        "- 줄의 형식은 그대로 둔다. 마크다운 기호, 표의 세로선(|) 개수, 줄 맨 앞의 표지('훅:', '관점:' 등)를 바꾸지 않는다.\n"
        "- 제목은 제목 길이로, 질문은 질문으로 둔다.\n"
        "- 편집자가 '자연'이라고 판정할 줄은 건드리지 않는다. 링크, 코드, 그림 지시(svg:, flow: 등)도 건드리지 않는다.\n\n"
        "고친 줄만 '줄 번호<탭>고친 내용' 형식으로 한 줄에 하나씩 답한다. 설명은 붙이지 않는다. 고칠 줄이 없으면 '없음'이라고만 답한다.\n\n"
        f"구간 (줄 번호<탭>내용):\n{body}\n"
    )


def apply(path: Path, answers: list[Path], out: Path) -> None:
    lines = read_lines(path)
    changes, dropped = {}, 0
    for answer in answers:
        for row in answer.read_text(encoding="utf-8").split("\n"):
            m = re.match(r"\s*(\d+)\t(.*)$", row)
            if not m:
                continue
            n, new = int(m.group(1)), m.group(2)
            ok = 1 <= n <= len(lines) and new.strip() and new != lines[n - 1] and new.count("|") == lines[n - 1].count("|")
            if ok:
                changes[n] = new
            else:
                dropped += 1
    out.write_text("\n".join(changes.get(n, line) for n, line in enumerate(lines, 1)), encoding="utf-8")
    log = [{"line": n, "before": lines[n - 1], "after": new} for n, new in sorted(changes.items())]
    out.with_suffix(".changes.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(log)}줄을 고쳤습니다 (형식이 맞지 않아 버린 답 {dropped}개) -> {out}")


def pick(out: Path, n: int, seed: int) -> list[dict]:
    log = json.loads(out.with_suffix(".changes.json").read_text(encoding="utf-8"))
    random.Random(seed).shuffle(log)
    return log[:n]


def record(out: Path, n: int, seed: int, bad: str, fixes: list[str]) -> None:
    bad_set = {int(x) for x in bad.split(",") if x.strip()}
    fix_map = {int(k): v for k, v in (f.split("=", 1) for f in fixes)}
    strip = lambda s: re.sub(r"^[\s|>*#-]+|[\s|]+$", "", s)
    rows = pick(out, n, seed)
    PROJECT_LABELS.parent.mkdir(exist_ok=True)
    with PROJECT_LABELS.open("a", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            if i in fix_map:
                entries = [{"text": strip(row["before"]), "label": "어색", "fixed": fix_map[i], "fixed_by": "user"}]
                if i in bad_set:
                    entries.append({"text": strip(row["after"]), "label": "어색"})
            elif i in bad_set:
                entries = [{"text": strip(row["after"]), "label": "어색"}]
            else:
                entries = [{"text": strip(row["before"]), "label": "어색", "fixed": strip(row["after"]), "fixed_by": "model, 사용자 확인"}]
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    passed = len(rows) - len(bad_set)
    print(f"판정 {len(rows)}개를 {PROJECT_LABELS}에 기록했습니다. 통과율 {100 * passed / max(1, len(rows)):.0f}% ({passed}/{len(rows)})")
    print("이 파일을 커밋해야 다음 세션에 남습니다.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("chunks"); p.add_argument("file", type=Path)
    p = sub.add_parser("brief"); p.add_argument("file", type=Path); p.add_argument("first", type=int); p.add_argument("last", type=int)
    p = sub.add_parser("apply"); p.add_argument("file", type=Path); p.add_argument("answers", type=Path, nargs="+"); p.add_argument("--out", type=Path, required=True)
    for name in ("sample", "record"):
        p = sub.add_parser(name); p.add_argument("file", type=Path); p.add_argument("--n", type=int, default=20); p.add_argument("--seed", type=int, default=0)
        if name == "record":
            p.add_argument("--bad", default=""); p.add_argument("--fix", action="append", default=[], metavar="번호=고친 문장")
    args = parser.parse_args()

    if args.command == "chunks":
        for first, last in chunk_spans(read_lines(args.file)):
            print(first, last)
    elif args.command == "brief":
        sys.stdout.write(brief(args.file, args.first, args.last))
    elif args.command == "apply":
        apply(args.file, args.answers, args.out)
    elif args.command == "sample":
        for i, row in enumerate(pick(args.file, args.n, args.seed), 1):
            print(f"{i}. (줄 {row['line']})\n   원문: {row['before'].strip()}\n   수정: {row['after'].strip()}\n")
    else:
        record(args.file, args.n, args.seed, args.bad, args.fix)


if __name__ == "__main__":
    main()
