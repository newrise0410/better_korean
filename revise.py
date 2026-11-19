#!/usr/bin/env python3
"""generate.py가 만든 글을 검토 지시문에 따라 한 번 고쳐서 새 JSONL 파일로 저장한다.

  python revise.py data/main-opus-5-5.jsonl --out data/main-opus-5-5-reviewed.jsonl

글마다 대화 기록이 없는 새 호출로 검토한다. 중간에 멈춰도 다시 실행하면 빠진 글만 채운다.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--review-file", type=Path, default=ROOT / "review_ko.txt")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if r.get("mode") == "main" and r.get("stop_reason") == "end_turn"]
    key = lambda r: (r["prompt_id"], r["arm"], r["idx"])
    done = set()
    if args.out.exists():
        done = {key(json.loads(line)) for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [r for r in rows if key(r) not in done]
    print(f"검토할 글 {len(rows)}편 가운데 {len(todo)}편을 호출합니다 -> {args.out}")
    if not todo:
        return

    review = args.review_file.read_text(encoding="utf-8")
    client = anthropic.Anthropic(base_url="https://api.anthropic.com", max_retries=5, timeout=300.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock, failures, count = threading.Lock(), [], [0, 0]

    def run(row: dict) -> None:
        try:
            response = client.messages.create(
                model=args.model,
                max_tokens=16000,
                system=review,
                messages=[{"role": "user", "content": row["text"]}],
            )
        except anthropic.RateLimitError:
            failures.append(f"{key(row)}: 요청 한도를 넘었습니다.")
        except anthropic.APIStatusError as e:
            failures.append(f"{key(row)}: 오류 {e.status_code} {e.message}")
        except anthropic.APIConnectionError:
            failures.append(f"{key(row)}: 네트워크 오류")
        else:
            revised = dict(
                row,
                original_text=row["text"],
                text="".join(b.text for b in response.content if b.type == "text"),
                stop_reason=response.stop_reason,
                review_model=response.model,
            )
            with lock:
                with args.out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(revised, ensure_ascii=False) + "\n")
                count[0] += 1
                count[1] += response.usage.output_tokens

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, todo))
    print(f"완료 {count[0]}편, 실패 {len(failures)}편, 출력 {count[1]:,}토큰")
    for line in failures[:10]:
        print("  실패:", line, file=sys.stderr)
    if failures:
        sys.exit("실패한 호출이 있습니다. 같은 명령을 다시 실행하면 빠진 글만 채웁니다.")


if __name__ == "__main__":
    main()
