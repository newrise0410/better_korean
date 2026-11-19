#!/usr/bin/env python3
"""한국어 글에서 영어 문장 틀을 옮긴 것으로 보이는 문장을 찾아 줄 번호와 함께 보여 준다.

  python lint.py 원고.md                 걸린 문장을 모두 출력
  python lint.py 원고.md --summary       규칙별 개수만 출력
  python lint.py 원고.md --rule 의문사   한 규칙만 출력

API를 쓰지 않는다. 규칙은 형태소 분석 결과에 대한 구조 검사이고, 규칙이 참조하는
단어는 lint_rules.json에 있다. 걸린 문장이 모두 틀린 문장은 아니다. 고칠지는 사람이 정한다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from analyze import clean_text, tokenize

ROOT = Path(__file__).resolve().parent
PREDICATE_TAGS = {"VV", "VA", "VX", "VCP", "VCN", "XSV", "XSA"}


class Rules:
    def __init__(self, path: Path):
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.skip = tuple(raw["skip_prefixes"])
        self.wh = set(raw["wh_words"])
        self.object_verbs = set(raw["object_verbs"])
        self.physical_verbs = set(raw["physical_verbs"])
        self.human_verbs = set(raw["human_verbs"])
        self.human_subjects = set(raw["human_subjects"])
        nouns, mods = "|".join(raw["overview_nouns"]), "|".join(raw["overview_modifiers"])
        # '한 장의 지도', '공통 언어'처럼 꾸미는 말이 붙었거나, '지도로 보여 준다'처럼 수단으로 쓰인 경우
        self.overview = re.compile(rf"({mods})(의)? ?({nouns})|({nouns})(로|으로) (보여|정리|묶|제공|그려)")


def check(sentence: str, rules: Rules) -> list[str]:
    """문장이 걸린 규칙 이름 목록."""
    tokens = tokenize(sentence)
    found = []

    # 1. 의문사 둘 이상이 서술어 하나를 나눠 쓰는 질문 ('무엇으로, 얼마나, 누구와 주고받을까?')
    if sentence.rstrip().endswith("?"):
        since_last_wh, seen_wh = None, False
        for form, tag, _, _ in tokens:
            if form in rules.wh:
                if seen_wh and since_last_wh == 0:
                    found.append("의문사")
                    break
                seen_wh, since_last_wh = True, 0
            elif tag in PREDICATE_TAGS and seen_wh:
                since_last_wh += 1

    # 2. 목적어가 필요한 동사인데 문장에 목적격 조사가 없다
    if any(f in rules.object_verbs and t == "VV" for f, t, _, _ in tokens) and not any(t == "JKO" for _, t, _, _ in tokens):
        found.append("목적어")

    # 3. 개요나 공통점을 사물에 빗댄 명사
    if rules.overview.search(sentence):
        found.append("사물비유")

    # 4. 추상적인 관계를 물리 동작에 빗댄 동사
    if any(f in rules.physical_verbs and t in ("VV", "VA") for f, t, _, _ in tokens):
        found.append("동작비유")

    # 5. 사람의 동작을 하는 사물 (문장에 사람을 가리키는 말이 없을 때만)
    forms = {f for f, _, _, _ in tokens}
    if forms & rules.human_verbs and not forms & rules.human_subjects:
        found.append("의인화")

    # 6. 명사를 기호로 이어 붙인 구
    if sentence.count("·") >= 3 or sentence.count("→") >= 2 or sentence.count(" / ") >= 2:
        found.append("기호나열")
    return found


def sentences(path: Path, rules: Rules):
    """(줄 번호, 문장)을 차례로 낸다. 코드 블록과 작업 지시 칸은 건너뛴다."""
    in_code = False
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        for cell in line.split("|"):
            if "http" in cell:
                continue
            text = clean_text(cell).strip(" -*>")
            if re.match(rf"^({'|'.join(map(re.escape, rules.skip))})\s*:", text):
                continue
            for sentence in re.split(r"(?<=[.?!])\s+", text):
                if len(sentence.strip()) >= 12:
                    yield number, sentence.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path)
    parser.add_argument("--rules", type=Path, default=ROOT / "lint_rules.json")
    parser.add_argument("--rule", help="이 규칙에 걸린 문장만 출력 (의문사, 목적어, 사물비유, 동작비유, 의인화, 기호나열)")
    parser.add_argument("--summary", action="store_true", help="규칙별 개수만 출력")
    parser.add_argument("--json", type=Path, help="결과를 JSON 파일로 저장")
    args = parser.parse_args()

    rules = Rules(args.rules)
    total, counts, rows = 0, Counter(), []
    for number, sentence in sentences(args.file, rules):
        total += 1
        hits = check(sentence, rules)
        counts.update(hits)
        if hits and (not args.rule or args.rule in hits):
            rows.append({"line": number, "rules": hits, "sentence": sentence})

    if not args.summary:
        for row in rows:
            print(f"줄 {row['line']} [{', '.join(row['rules'])}] {row['sentence']}")
        print()
    flagged = sum(1 for _ in rows) if not args.rule else len(rows)
    print(f"문장 {total:,}개 가운데 {flagged:,}개가 걸렸습니다 ({100 * flagged / max(1, total):.1f}%).")
    for name, n in counts.most_common():
        print(f"  {name}: {n}개 (문장 1,000개당 {1000 * n / max(1, total):.1f})")
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
