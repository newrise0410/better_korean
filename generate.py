#!/usr/bin/env python3
"""같은 프롬프트를 두 가지 샘플링 설정으로 여러 번 생성해 JSONL 파일로 저장한다.

  python generate.py probe              샘플링 확인용 짧은 응답 (비용이 거의 들지 않는다)
  python generate.py main               본 실험용 글
  python generate.py main --dry-run     호출 계획만 출력하고 끝낸다

설정 A는 샘플링 파라미터를 보내지 않는다. 워터마크가 실리는 기본 상태다.
설정 B는 temperature 0.99를 보낸다. Anthropic 문서에 따르면 4.x 모델에서는
기본값이 아닌 샘플링 파라미터를 보내면 워터마크가 실리지 않을 수 있다.

중간에 멈춰도 같은 명령을 다시 실행하면 빠진 응답만 채운다.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent

# 실험은 Anthropic 직접 API에서만 의미가 있으므로 환경 변수 ANTHROPIC_BASE_URL을 따르지 않는다.
API_BASE_URL = "https://api.anthropic.com"

# 샘플링 파라미터를 받는 모델 (Anthropic 문서 기준, 2026년 11월)
SAMPLING_MODELS = {"claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"}

DEFAULT_ARMS = {"A": None, "B": {"temperature": 0.99}}
DEFAULTS = {"main": {"n": 10, "max_tokens": 4096}, "probe": {"n": 100, "max_tokens": 64}}


def parse_arms(specs: list[str] | None) -> dict:
    if not specs:
        return DEFAULT_ARMS
    arms = {}
    for spec in specs:
        name, _, body = spec.partition("=")
        arms[name] = json.loads(body) if body and body != "null" else None
    return arms


def plan(prompts: dict, mode: str, n: int, arms: dict, limit: int | None) -> list[dict]:
    items = prompts[mode][:limit] if limit else prompts[mode]
    suffix = prompts.get("suffix", "") if mode == "main" else ""
    return [
        {"mode": mode, "prompt_id": p["id"], "arm": arm, "idx": i, "prompt": p["prompt"] + suffix, "params": params}
        for p in items
        for i in range(n)
        for arm, params in arms.items()
    ]


def key(row: dict) -> tuple:
    return row["mode"], row["prompt_id"], row["arm"], row["idx"]


def call(client: anthropic.Anthropic, task: dict, args) -> dict:
    request = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "messages": [{"role": "user", "content": task["prompt"]}],
    }
    if args.system:
        request["system"] = args.system
    if task["params"]:
        # SDK 1.x의 messages.create()는 temperature 인자를 받지 않으므로 요청 본문에 직접 넣는다.
        request["extra_body"] = task["params"]
    response = client.messages.create(**request)
    return {
        "mode": task["mode"],
        "prompt_id": task["prompt_id"],
        "arm": task["arm"],
        "idx": task["idx"],
        "model": args.model,
        "served_model": response.model,
        "params": task["params"],
        "text": "".join(block.text for block in response.content if block.type == "text"),
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "request_id": response._request_id,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sdk": anthropic.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["main", "probe"])
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--n", type=int, help="프롬프트마다 설정별 생성 횟수 (main 10, probe 100)")
    parser.add_argument("--out", type=Path, help="저장할 파일 (기본: data/<mode>.jsonl)")
    parser.add_argument("--prompts", type=Path, default=ROOT / "prompts.json")
    parser.add_argument("--arm", action="append", metavar="이름=JSON",
                        help="설정을 직접 지정한다. 예: --arm A=null --arm B='{\"top_k\": 500}'")
    parser.add_argument("--limit-prompts", type=int, help="앞에서부터 이 개수의 프롬프트만 쓴다 (시험 실행용)")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--system-file", type=Path, help="시스템 프롬프트로 쓸 파일")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-url", default=API_BASE_URL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if int(anthropic.__version__.split(".")[0]) >= 2:
        sys.exit(
            f"anthropic SDK {anthropic.__version__}은 API 버전 2026-10-21을 보내고, 이 버전은 temperature를 거부합니다.\n"
            "pip install 'anthropic>=1,<2' 로 설치한 뒤 다시 실행하세요."
        )
    if args.model not in SAMPLING_MODELS:
        print(f"주의: {args.model}은 샘플링 파라미터를 받지 않을 수 있습니다. 받는 모델: {sorted(SAMPLING_MODELS)}", file=sys.stderr)

    args.n = args.n or DEFAULTS[args.mode]["n"]
    args.max_tokens = args.max_tokens or DEFAULTS[args.mode]["max_tokens"]
    args.out = args.out or ROOT / "data" / f"{args.mode}.jsonl"
    args.system = args.system_file.read_text(encoding="utf-8") if args.system_file else None
    arms = parse_arms(args.arm)

    tasks = plan(json.loads(args.prompts.read_text(encoding="utf-8")), args.mode, args.n, arms, args.limit_prompts)
    done = set()
    if args.out.exists():
        done = {key(json.loads(line)) for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [t for t in tasks if key(t) not in done]
    random.Random(0).shuffle(todo)   # 두 설정의 호출이 시간상 섞이게 한다

    print(f"모델 {args.model}, 설정 {arms}")
    print(f"계획 {len(tasks)}회 가운데 {len(tasks) - len(todo)}회는 이미 있고 {len(todo)}회를 호출합니다 -> {args.out}")
    if args.dry_run or not todo:
        return

    client = anthropic.Anthropic(base_url=args.base_url, max_retries=5, timeout=180.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock, stop = threading.Lock(), threading.Event()
    stats, failures = Counter(), []

    def run(task: dict) -> None:
        if stop.is_set():
            return
        label = f"{task['prompt_id']}/{task['arm']}{task['idx']}"
        try:
            row = call(client, task, args)
        # 다시 시도해도 고쳐지지 않는 오류는 전체를 멈춘다.
        except anthropic.BadRequestError as e:
            stop.set()
            failures.append(f"{label}: 요청이 거부됐습니다. 모델이 이 파라미터를 받는지 확인하세요. {e.message}")
        except anthropic.AuthenticationError:
            stop.set()
            failures.append(f"{label}: API 키가 올바르지 않습니다.")
        except anthropic.PermissionDeniedError as e:
            stop.set()
            failures.append(f"{label}: 이 API 키로는 요청할 수 없습니다. {e.message}")
        except anthropic.NotFoundError:
            stop.set()
            failures.append(f"{label}: 모델 {args.model}을 찾을 수 없습니다.")
        # 아래 오류는 SDK가 이미 5번 다시 시도한 뒤다. 기록만 하고 계속한다.
        except anthropic.RateLimitError:
            failures.append(f"{label}: 요청 한도를 넘었습니다. --workers 값을 줄여 다시 실행하세요.")
        except anthropic.APIStatusError as e:
            failures.append(f"{label}: 서버 오류 {e.status_code}")
        except anthropic.APIConnectionError:
            failures.append(f"{label}: 네트워크 오류")
        except anthropic.AnthropicError as e:   # 인증 정보를 찾지 못한 경우 등
            stop.set()
            failures.append(f"{label}: {e}")
        else:
            with lock:
                with args.out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["ok"] += 1
                stats["input"] += row["input_tokens"]
                stats["output"] += row["output_tokens"]
                stats[f"model:{row['served_model']}"] += 1
                if stats["ok"] % 20 == 0:
                    print(f"  {stats['ok']}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, task) for task in todo]):
            future.result()

    print(f"완료 {stats['ok']}회, 실패 {len(failures)}회. 입력 {stats['input']:,}토큰, 출력 {stats['output']:,}토큰")
    served = {k.removeprefix("model:"): v for k, v in stats.items() if k.startswith("model:")}
    if served:
        print(f"응답한 모델: {served}")
    for line in failures[:10]:
        print("  실패:", line, file=sys.stderr)
    if failures:
        sys.exit("실패한 호출이 있습니다. 원인을 고친 뒤 같은 명령을 다시 실행하면 빠진 응답만 채웁니다.")


if __name__ == "__main__":
    main()
