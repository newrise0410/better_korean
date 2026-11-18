#!/usr/bin/env python3
"""두 묶음의 한국어 글에서 단어 선택을 비교한다.

  python analyze.py ab data/main.jsonl        API 실험 결과 (설정 A 대 설정 B)
  python analyze.py dirs --a 이전 --b 이후     폴더 두 개 (날짜 비교, 사람 글 대조)
  python analyze.py probe data/probe.jsonl    두 설정의 샘플링이 실제로 다른지 확인
  python analyze.py tokens 파일.txt           형태소 분석과 집계 항목 확인

비교 단위는 문서다. 유의확률은 문서의 묶음 표지를 층(프롬프트) 안에서 섞는
순열 검정으로 구하고, 신뢰구간은 층과 문서를 복원 추출하는 부트스트랩으로 구한다.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

# 어휘 전체 비교에 쓰는 품사 (조사, 어미, 기호는 제외)
CONTENT_TAGS = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "MAG", "MM", "XR", "SL"}

# 판정 기준. 비율의 비(A/B)에 대한 95% 신뢰구간이 이 범위 안에 들어오면 "사실상 같다"고 본다.
EQUIV_LOW, EQUIV_HIGH = 0.8, 1.25
MIN_HITS = 20        # 두 묶음을 합친 등장 횟수가 이보다 적으면 판단하지 않는다
TTR_WINDOW = 150     # 어휘 다양도를 계산할 때 쓰는 내용어 개수


# ---------------------------------------------------------------- 글 정리와 형태소 분석

_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_URL = re.compile(r"https?://\S+")
_LINE_MARK = re.compile(r"^[ \t]*(#{1,6}|[-*+]|\d+\.|>)[ \t]+", re.M)


def clean_text(text: str) -> str:
    """코드 블록, URL, 마크다운 기호를 지운다. 문장은 그대로 둔다."""
    text = _FENCE.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _LINE_MARK.sub("", text)
    return text.replace("**", "").replace("__", "").replace("|", " ").strip()


_kiwi = None


def tokenize(text: str) -> list[tuple[str, str, int, int]]:
    """(형태, 품사, 시작 위치, 길이) 목록. 품사의 불규칙 활용 표시(-I, -R)는 뗀다."""
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi

        _kiwi = Kiwi()
    return [(t.form, t.tag.split("-")[0], t.start, t.len) for t in _kiwi.tokenize(text)]


# ---------------------------------------------------------------- 집계 항목


@dataclass(frozen=True)
class Entry:
    raw: str
    parts: tuple[tuple[str, str], ...]
    not_next: tuple[str, str] | None


def parse_entry(raw: str) -> Entry:
    def one(piece: str) -> tuple[str, str]:
        form, _, tag = piece.rpartition("/")
        if not form or not tag:
            raise ValueError(f"집계 항목 형식이 잘못됐습니다: {raw!r} ('형태/품사'여야 합니다)")
        return form, tag

    body, _, neg = raw.partition("!")
    return Entry(raw, tuple(one(p) for p in body.split("+")), one(neg) if neg else None)


def find_entry(tokens, index, entry: Entry) -> list[int]:
    """항목이 나타난 형태소 위치 목록."""
    found = []
    first_form, first_tag = entry.parts[0]
    for i in index.get(first_form, ()):
        if not fnmatchcase(tokens[i][1], first_tag):
            continue
        end = i + len(entry.parts)
        if end > len(tokens):
            continue
        if any(
            tokens[i + k][0] != form or not fnmatchcase(tokens[i + k][1], tag)
            for k, (form, tag) in enumerate(entry.parts[1:], 1)
        ):
            continue
        if entry.not_next and end < len(tokens):
            form, tag = entry.not_next
            if tokens[end][0] == form and fnmatchcase(tokens[end][1], tag):
                continue
        found.append(i)
    return found


class Lexicon:
    def __init__(self, path: Path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.pairs = [
            (p["id"], [parse_entry(e) for e in p["native"]], [parse_entry(e) for e in p["standard"]])
            for p in raw["pairs"]
        ]
        groups = raw.get("groups", {})
        pair_native = [e for _, native, _ in self.pairs for e in native]
        self.native = _unique(pair_native + [parse_entry(e) for e in groups.get("native_extra", [])])
        self.rare_sino = _unique([parse_entry(e) for e in groups.get("rare_sino", [])])
        self.demonstratives = _unique([parse_entry(e) for e in groups.get("demonstratives", [])])
        pair_standard = [e for _, _, standard in self.pairs for e in standard]
        self.all = _unique(self.native + pair_standard + self.rare_sino + self.demonstratives)


def _unique(entries: list[Entry]) -> list[Entry]:
    seen, out = set(), []
    for e in entries:
        if e.raw not in seen:
            seen.add(e.raw)
            out.append(e)
    return out


# ---------------------------------------------------------------- 문서


@dataclass
class Doc:
    group: str      # "A" 또는 "B"
    stratum: str    # 프롬프트 id. 폴더 비교에서는 모두 "all"
    name: str
    text: str
    tokens: list
    n_eojeol: int
    hits: dict      # 항목 원문 -> 형태소 위치 목록

    def count(self, entries: list[Entry]) -> int:
        return sum(len(self.hits.get(e.raw, ())) for e in entries)

    def content_lemmas(self) -> list[str]:
        return [f"{form}/{tag}" for form, tag, _, _ in self.tokens if tag in CONTENT_TAGS]


def build_doc(group: str, stratum: str, name: str, text: str, lex: Lexicon) -> Doc:
    text = clean_text(text)
    tokens = tokenize(text)
    index: dict[str, list[int]] = {}
    for i, tok in enumerate(tokens):
        index.setdefault(tok[0], []).append(i)
    hits = {}
    for entry in lex.all:
        found = find_entry(tokens, index, entry)
        if found:
            hits[entry.raw] = found
    return Doc(group, stratum, name, text, tokens, len(text.split()), hits)


def load_jsonl(path: Path, lex: Lexicon, arms: tuple[str, str]) -> list[Doc]:
    docs, skipped = [], Counter()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("mode") != "main" or row.get("arm") not in arms:
            continue
        if row.get("stop_reason") != "end_turn":
            skipped[row.get("stop_reason")] += 1   # 잘린 글과 거절된 응답은 뺀다
            continue
        group = "A" if row["arm"] == arms[0] else "B"
        docs.append(build_doc(group, row["prompt_id"], f"{row['prompt_id']}#{row['arm']}{row['idx']}", row["text"], lex))
    if skipped:
        print(f"주의: 정상 종료하지 않은 응답 {sum(skipped.values())}개를 뺐습니다 {dict(skipped)}", file=sys.stderr)
    return docs


_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")


def split_chunks(text: str, size: int) -> list[str]:
    """문장 경계를 지키면서 어절 size개 안팎의 조각으로 나눈다."""
    if size <= 0:
        return [text]
    chunks, current, n = [], [], 0
    for sent in _SENT_SPLIT.split(text):
        if not sent.strip():
            continue
        current.append(sent)
        n += len(sent.split())
        if n >= size:
            chunks.append(" ".join(current))
            current, n = [], 0
    if current and (n >= size // 2 or not chunks):
        chunks.append(" ".join(current))   # 너무 짧은 끝 조각은 버린다
    return chunks


def load_dir(folder: Path, group: str, lex: Lexicon, chunk: int) -> list[Doc]:
    files = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in {".txt", ".md"})
    if not files:
        sys.exit(f"{folder} 안에 .txt 또는 .md 파일이 없습니다.")
    docs = []
    for path in files:
        text = clean_text(path.read_text(encoding="utf-8"))
        for k, piece in enumerate(split_chunks(text, chunk)):
            docs.append(build_doc(group, "all", f"{path.name}#{k}", piece, lex))
    return docs


# ---------------------------------------------------------------- 통계


def perm_labels(is_a: np.ndarray, strata: np.ndarray, n_perm: int, rng) -> np.ndarray:
    """층 안에서만 묶음 표지를 섞은 0/1 행렬 (n_perm x 문서 수)."""
    out = np.empty((n_perm, len(is_a)))
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[:, idx] = rng.permuted(np.tile(is_a[idx].astype(float), (n_perm, 1)), axis=1)
    return out


def boot_weights(is_a: np.ndarray, strata: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """층을 복원 추출하고, 뽑힌 층 안에서 묶음별로 문서를 복원 추출한 횟수 행렬."""
    names = np.unique(strata)
    picks = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=n_boot)
    out = np.zeros((n_boot, len(is_a)))
    for k, s in enumerate(names):
        for flag in (True, False):
            idx = np.flatnonzero((strata == s) & (is_a == flag))
            if len(idx):
                out[:, idx] = rng.multinomial(picks[:, k] * len(idx), np.full(len(idx), 1 / len(idx)))
    return out


def compare_rates(num, den, is_a, strata, rng, n_perm, n_boot) -> dict:
    """묶음별 비율(분자 합 / 분모 합)을 비교한다."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    a = is_a.astype(float)
    na, da, nb, db = (num * a).sum(), (den * a).sum(), (num * (1 - a)).sum(), (den * (1 - a)).sum()
    out = {"num_a": int(na), "den_a": int(da), "num_b": int(nb), "den_b": int(db)}
    if da == 0 or db == 0:
        return out | {"rate_a": None, "rate_b": None, "ratio": None, "ci": None, "p": None}
    observed = na / da - nb / db

    with np.errstate(divide="ignore", invalid="ignore"):
        labels = perm_labels(is_a, strata, n_perm, rng)
        pna, pda = labels @ num, labels @ den
        diffs = np.nan_to_num(pna / pda - (num.sum() - pna) / (den.sum() - pda))
        p = (1 + np.sum(np.abs(diffs) >= abs(observed) - 1e-12)) / (n_perm + 1)

        w = boot_weights(is_a, strata, n_boot, rng)
        bna, bda, bnb, bdb = w @ (num * a), w @ (den * a), w @ (num * (1 - a)), w @ (den * (1 - a))
        ratios = ((bna + 0.5) / bda) / ((bnb + 0.5) / bdb)
        ratios = ratios[np.isfinite(ratios)]

    ci = [float(x) for x in np.percentile(ratios, [2.5, 97.5])] if len(ratios) else None
    return out | {
        "rate_a": na / da,
        "rate_b": nb / db,
        "ratio": float(((na + 0.5) / da) / ((nb + 0.5) / db)),   # 0회일 때를 대비해 0.5를 더한다
        "ci": ci,
        "p": float(p),
    }


def verdict(r: dict) -> str:
    if r["ratio"] is None or r["ci"] is None:
        return "판단할 수 없습니다 (한 묶음에 해당 단어가 나올 자리가 없습니다)."
    if r["num_a"] + r["num_b"] < MIN_HITS:
        return f"판단할 수 없습니다 (해당 단어가 두 묶음을 합쳐 {r['num_a'] + r['num_b']}회만 나왔습니다)."
    low, high = r["ci"]
    if low >= EQUIV_LOW and high <= EQUIV_HIGH:
        return "두 묶음의 비율이 사실상 같습니다."
    if low > 1 and r["p"] < 0.05:
        return "A가 B보다 높습니다."
    if high < 1 and r["p"] < 0.05:
        return "A가 B보다 낮습니다."
    return "판단을 보류합니다 (차이가 있다고도 없다고도 말하기 어렵습니다. 표본을 늘려야 합니다)."


def compare_means(values, is_a, strata, rng, n_perm) -> dict:
    values = np.asarray(values, float)
    keep = np.isfinite(values)
    values, is_a, strata = values[keep], is_a[keep], strata[keep]
    if is_a.sum() == 0 or (~is_a).sum() == 0:
        return {"mean_a": None, "mean_b": None, "p": None}
    a = is_a.astype(float)
    mean_a, mean_b = values[is_a].mean(), values[~is_a].mean()
    labels = perm_labels(is_a, strata, n_perm, rng)
    pa = (labels @ values) / a.sum()
    pb = (values.sum() - labels @ values) / (len(values) - a.sum())
    p = (1 + np.sum(np.abs(pa - pb) >= abs(mean_a - mean_b) - 1e-12)) / (n_perm + 1)
    return {"mean_a": float(mean_a), "mean_b": float(mean_b), "p": float(p)}


def _jsd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """행마다 두 빈도 분포의 젠슨-섀넌 발산(비트)."""
    p = a / a.sum(axis=1, keepdims=True)
    q = b / b.sum(axis=1, keepdims=True)
    m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0, p * np.log2(p / m), 0).sum(axis=1)
        kl_q = np.where(q > 0, q * np.log2(q / m), 0).sum(axis=1)
    return (kl_p + kl_q) / 2


def compare_vocabulary(docs: list[Doc], is_a, strata, lex: Lexicon, rng, n_perm: int, top: int = 10) -> dict:
    """내용어 전체의 빈도 분포가 두 묶음에서 다른지 본다."""
    counts = [Counter(d.content_lemmas()) for d in docs]
    total = Counter()
    for c in counts:
        total.update(c)
    vocab = sorted(w for w, n in total.items() if n >= 5)
    if not vocab or is_a.sum() == 0 or (~is_a).sum() == 0:
        return {"jsd": None, "p": None, "more_a": [], "more_b": []}
    col = {w: j for j, w in enumerate(vocab)}
    x = np.zeros((len(docs), len(vocab)))
    for i, c in enumerate(counts):
        for w, n in c.items():
            if w in col:
                x[i, col[w]] = n
    a = is_a.astype(float)
    ya, yb = a @ x, (1 - a) @ x
    observed = float(_jsd(ya[None, :], yb[None, :])[0])

    extreme, done = 0, 0
    while done < n_perm:                      # 메모리를 아끼려고 500개씩 나눠 계산한다
        step = min(500, n_perm - done)
        pa = perm_labels(is_a, strata, step, rng) @ x
        extreme += int(np.sum(_jsd(pa, x.sum(axis=0) - pa) >= observed - 1e-12))
        done += step
    p = (1 + extreme) / (n_perm + 1)

    # 단어별 로그 오즈비의 z 점수 (Monroe 외 2008, 사전분포는 전체 빈도에 비례)
    prior = (ya + yb) * 500 / (ya + yb).sum() + 0.01
    la = np.log((ya + prior) / (ya.sum() + prior.sum() - ya - prior))
    lb = np.log((yb + prior) / (yb.sum() + prior.sum() - yb - prior))
    z = (la - lb) / np.sqrt(1 / (ya + prior) + 1 / (yb + prior))
    native = {e.raw for e in lex.native if len(e.parts) == 1 and not e.not_next}

    def rows(order, sign):
        return [
            {"word": vocab[j], "a": int(ya[j]), "b": int(yb[j]), "z": float(z[j]), "native": vocab[j] in native}
            for j in order[:top]
            if sign * z[j] >= 1.5   # 우연으로 볼 만한 작은 차이는 싣지 않는다
        ]

    order = np.argsort(-z)
    return {"jsd": observed, "p": float(p), "vocab": len(vocab), "more_a": rows(order, 1), "more_b": rows(order[::-1], -1)}


def type_token_ratio(doc: Doc) -> float:
    lemmas = doc.content_lemmas()
    if len(lemmas) < TTR_WINDOW:
        return math.nan
    return len(set(lemmas[:TTR_WINDOW])) / TTR_WINDOW


# ---------------------------------------------------------------- 보고서


def _width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _width(s))


def _fmt_rate(r: dict, scale: float, unit: str) -> str:
    if r["rate_a"] is None:
        return "계산할 수 없음"
    ci = f"{r['ci'][0]:.2f}~{r['ci'][1]:.2f}" if r["ci"] else "-"
    return (
        f"A {r['rate_a'] * scale:.2f}{unit} ({r['num_a']}회)   B {r['rate_b'] * scale:.2f}{unit} ({r['num_b']}회)   "
        f"A/B {r['ratio']:.2f} (95% 신뢰구간 {ci})   p={r['p']:.3f}"
    )


def _context(doc: Doc, pos: int, span: int = 22) -> str:
    _, _, start, length = doc.tokens[pos]
    left = doc.text[max(0, start - span):start].replace("\n", " ")
    right = doc.text[start + length:start + length + span].replace("\n", " ")
    return f"…{left}[{doc.text[start:start + length]}]{right}…"


def compare(docs: list[Doc], lex: Lexicon, labels: tuple[str, str], args) -> dict:
    rng = np.random.default_rng(args.seed)
    is_a = np.array([d.group == "A" for d in docs])
    strata = np.array([d.stratum for d in docs])
    eojeol = np.array([d.n_eojeol for d in docs], float)
    if is_a.sum() == 0 or (~is_a).sum() == 0:
        sys.exit("두 묶음 중 하나에 문서가 없습니다.")
    result: dict = {"labels": {"A": labels[0], "B": labels[1]}}

    print("== 표본 ==")
    for flag, key in ((True, "A"), (False, "B")):
        mask = is_a == flag
        print(f"{key} ({result['labels'][key]}): 문서 {int(mask.sum())}개, 어절 {int(eojeol[mask].sum()):,}개")
    print(f"층(프롬프트) {len(set(strata))}개, 순열 {args.perm:,}회, 부트스트랩 {args.boot:,}회")
    result["docs"] = {"A": int(is_a.sum()), "B": int((~is_a).sum())}

    def rate_section(title: str, key: str, entries: list[Entry]) -> None:
        r = compare_rates([d.count(entries) for d in docs], eojeol, is_a, strata, rng, args.perm, args.boot)
        result[key] = r
        print(f"\n== {title} (어절 1,000개당 횟수) ==")
        print(_fmt_rate(r, 1000, ""))
        print("판정:", verdict(r))

    rate_section("1. 고유어 표지", "native_rate", lex.native)
    by_entry = sorted(
        ((e.raw, sum(len(d.hits.get(e.raw, ())) for d in docs if d.group == "A"),
          sum(len(d.hits.get(e.raw, ())) for d in docs if d.group == "B")) for e in lex.native),
        key=lambda row: -(row[1] + row[2]),
    )
    shown = [row for row in by_entry if row[1] + row[2] > 0][:15]
    if shown:
        print("자주 나온 표지 (A회 / B회): " + ", ".join(f"{raw} {a}/{b}" for raw, a, b in shown))
    result["native_by_entry"] = [{"entry": raw, "a": a, "b": b} for raw, a, b in by_entry if a + b]

    print("\n== 2. 쌍별 고유어 선택 ==")
    print(_pad("쌍", 8) + _pad("A 고유어/표준", 16) + _pad("B 고유어/표준", 16) + _pad("A 고유어 비율", 15) + "B 고유어 비율")
    native_all, both_all, pair_rows = np.zeros(len(docs)), np.zeros(len(docs)), []
    for pid, native, standard in lex.pairs:
        n = np.array([d.count(native) for d in docs], float)
        s = np.array([d.count(standard) for d in docs], float)
        native_all += n
        both_all += n + s
        na, sa, nb, sb = n[is_a].sum(), s[is_a].sum(), n[~is_a].sum(), s[~is_a].sum()
        pair_rows.append({"pair": pid, "a_native": int(na), "a_standard": int(sa), "b_native": int(nb), "b_standard": int(sb)})
        if na + sa + nb + sb == 0:
            continue
        share = lambda x, y: f"{100 * x / (x + y):.1f}%" if x + y else "-"
        print(_pad(pid, 8) + _pad(f"{int(na)} / {int(sa)}", 16) + _pad(f"{int(nb)} / {int(sb)}", 16) + _pad(share(na, sa), 15) + share(nb, sb))
    r = compare_rates(native_all, both_all, is_a, strata, rng, args.perm, args.boot)
    result["pairs"], result["pair_share"] = pair_rows, r
    print("전체 고유어 선택 비율: " + _fmt_rate(r, 100, "%"))
    print("판정:", verdict(r))

    rate_section("3. 드문 한자어, 대조군", "rare_sino_rate", lex.rare_sino)
    rate_section("4. 지시어", "demonstrative_rate", lex.demonstratives)

    print("\n== 5. 어휘 전체 비교 ==")
    v = compare_vocabulary(docs, is_a, strata, lex, rng, min(args.perm, 2000))
    result["vocabulary"] = v
    if v["jsd"] is None:
        print("어휘가 너무 적어서 계산하지 않았습니다.")
    else:
        note = "두 묶음의 단어 빈도 분포가 다릅니다." if v["p"] < 0.05 else "두 묶음의 단어 빈도 분포에서 차이를 찾지 못했습니다."
        print(f"젠슨-섀넌 발산 {v['jsd']:.4f} (단어 {v['vocab']:,}종), p={v['p']:.3f}: {note}")
        for key, title in (("more_a", "A에서 더 자주 나온 단어"), ("more_b", "B에서 더 자주 나온 단어")):
            words = ", ".join(f"{w['word']}{'*' if w['native'] else ''} {w['a']}/{w['b']}" for w in v[key])
            print(f"{title} (A회/B회, *는 고유어 표지): {words or '없음'}")

    print("\n== 6. 어휘 다양도 ==")
    t = compare_means([type_token_ratio(d) for d in docs], is_a, strata, rng, args.perm)
    result["type_token_ratio"] = t
    if t["mean_a"] is None:
        print(f"내용어가 {TTR_WINDOW}개 이상인 문서가 부족해서 계산하지 않았습니다.")
    else:
        print(f"첫 내용어 {TTR_WINDOW}개 가운데 서로 다른 단어의 비율: A {t['mean_a']:.3f}   B {t['mean_b']:.3f}   p={t['p']:.3f}")

    if args.show_hits:
        print(f"\n== 고유어 표지가 나온 문맥 (항목마다 최대 {args.show_hits}개) ==")
        for e in lex.native:
            found = [(d, pos) for d in docs for pos in d.hits.get(e.raw, ())][: args.show_hits]
            for d, pos in found:
                print(f"{_pad(e.raw, 14)} {d.group} {_pad(d.name, 22)} {_context(d, pos)}")
    return result


# ---------------------------------------------------------------- 샘플링 확인 (probe)


def extract_answer(text: str, how: str) -> str | None:
    if how in ("first_int", "last_int"):
        found = re.findall(r"\d+", text)
        return (found[0] if how == "first_int" else found[-1]) if found else None
    found = re.findall(r"[가-힣A-Za-z]+", text)
    return found[-1] if found else None


def tv_test(a: list[str], b: list[str], rng, n_perm: int) -> tuple[float, float]:
    """두 답 분포의 총변동거리와 순열 검정 유의확률."""
    cats = {c: i for i, c in enumerate(sorted(set(a) | set(b)))}
    x = np.array([cats[v] for v in a + b])

    def tv(values):
        pa = np.bincount(values[: len(a)], minlength=len(cats)) / len(a)
        pb = np.bincount(values[len(a):], minlength=len(cats)) / len(b)
        return 0.5 * np.abs(pa - pb).sum()

    observed = tv(x)
    extreme = sum(tv(rng.permutation(x)) >= observed - 1e-12 for _ in range(n_perm))
    return float(observed), (1 + extreme) / (n_perm + 1)


def _entropy(values: list[str]) -> float:
    n = len(values)
    return -sum(c / n * math.log2(c / n) for c in Counter(values).values())


def probe(path: Path, prompts_path: Path, arms: tuple[str, str], args) -> dict:
    how = {p["id"]: p.get("extract", "last_word") for p in json.loads(prompts_path.read_text(encoding="utf-8"))["probe"]}
    answers: dict = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("mode") != "probe" or row.get("arm") not in arms or row["prompt_id"] not in how:
            continue
        answer = extract_answer(row["text"], how[row["prompt_id"]])
        if answer is not None:
            answers.setdefault(row["prompt_id"], {arms[0]: [], arms[1]: []})[row["arm"]].append(answer)
    if not answers:
        sys.exit("probe 응답이 없습니다. 먼저 'python generate.py probe'를 실행하세요.")

    rng = np.random.default_rng(args.seed)
    result, smallest = {}, 1.0
    print("== 같은 프롬프트에 대한 답의 분포 ==")
    for pid, by_arm in answers.items():
        a, b = by_arm[arms[0]], by_arm[arms[1]]
        if not a or not b:
            continue
        tv, p = tv_test(a, b, rng, args.perm)
        smallest = min(smallest, p)
        result[pid] = {"n_a": len(a), "n_b": len(b), "tv": tv, "p": p, "entropy_a": _entropy(a), "entropy_b": _entropy(b)}
        ca, cb = Counter(a), Counter(b)
        top = sorted(set(ca) | set(cb), key=lambda k: -(ca[k] + cb[k]))[:8]
        print(f"\n[{pid}] {arms[0]} {len(a)}개, {arms[1]} {len(b)}개")
        print(f"  답 ({arms[0]}회/{arms[1]}회): " + ", ".join(f"{k} {ca[k]}/{cb[k]}" for k in top))
        print(f"  엔트로피 {arms[0]} {_entropy(a):.2f}비트, {arms[1]} {_entropy(b):.2f}비트   총변동거리 {tv:.3f}   p={p:.4f}")
    if not result:
        sys.exit("두 설정 모두에 답이 있는 probe 프롬프트가 없습니다.")

    adjusted = min(1.0, smallest * len(result))   # 프롬프트 개수만큼 보정 (본페로니)
    result["adjusted_p"] = adjusted
    print("\n== 판정 ==")
    if adjusted < 0.01:
        print(f"두 설정의 샘플링 결과가 다릅니다 (보정 p={adjusted:.4f}).")
        print("temperature 1.0과 0.99의 차이만으로는 이런 차이가 생기지 않습니다.")
        print("기본값이 아닌 파라미터가 샘플링 경로를 바꾼다는 뜻이고, 워터마크가 한쪽에서만 켜진 상황과 부합합니다.")
    else:
        print(f"두 설정의 샘플링 결과에서 차이를 찾지 못했습니다 (보정 p={adjusted:.4f}).")
        print("두 설정 모두 워터마크가 켜져 있거나, 이 프롬프트에는 워터마크가 적용되지 않았을 수 있습니다.")
        print("이 경우 본 실험의 '차이 없음'은 약한 증거로만 읽어야 합니다.")
    return result


# ---------------------------------------------------------------- 명령줄


def show_tokens(path: Path, lex: Lexicon) -> None:
    doc = build_doc("A", "all", path.name, path.read_text(encoding="utf-8"), lex)
    print(" ".join(f"{form}/{tag}" for form, tag, _, _ in doc.tokens))
    print()
    for title, entries in (("고유어 표지", lex.native), ("드문 한자어", lex.rare_sino), ("지시어", lex.demonstratives)):
        found = [(e.raw, len(doc.hits[e.raw])) for e in entries if e.raw in doc.hits]
        print(f"{title}: " + (", ".join(f"{raw} {n}회" for raw, n in found) or "없음"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lexicon", type=Path, default=ROOT / "lexicon.json")
    parser.add_argument("--perm", type=int, default=5000, help="순열 횟수")
    parser.add_argument("--boot", type=int, default=2000, help="부트스트랩 횟수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-hits", type=int, default=0, metavar="N", help="고유어 표지가 나온 문맥을 항목마다 N개 출력")
    parser.add_argument("--json", type=Path, help="계산 결과를 JSON 파일로 저장")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ab = sub.add_parser("ab", help="generate.py main 결과 비교")
    p_ab.add_argument("data", type=Path)
    p_ab.add_argument("--arms", nargs=2, default=["A", "B"], metavar=("A", "B"))

    p_dirs = sub.add_parser("dirs", help="폴더 두 개 비교")
    p_dirs.add_argument("--a", type=Path, required=True)
    p_dirs.add_argument("--b", type=Path, required=True)
    p_dirs.add_argument("--chunk", type=int, default=300, help="문서를 나누는 어절 수 (0이면 파일 하나가 문서 하나)")

    p_probe = sub.add_parser("probe", help="generate.py probe 결과로 샘플링 차이 확인")
    p_probe.add_argument("data", type=Path)
    p_probe.add_argument("--arms", nargs=2, default=["A", "B"], metavar=("A", "B"))
    p_probe.add_argument("--prompts", type=Path, default=ROOT / "prompts.json")

    p_tokens = sub.add_parser("tokens", help="형태소 분석 결과와 집계 항목 확인")
    p_tokens.add_argument("file", type=Path)

    args = parser.parse_args()
    if args.command == "probe":
        result = probe(args.data, args.prompts, tuple(args.arms), args)
    else:
        lex = Lexicon(args.lexicon)
        if args.command == "tokens":
            show_tokens(args.file, lex)
            return
        if args.command == "ab":
            docs = load_jsonl(args.data, lex, tuple(args.arms))
            labels = (f"설정 {args.arms[0]}", f"설정 {args.arms[1]}")
        else:
            docs = load_dir(args.a, "A", lex, args.chunk) + load_dir(args.b, "B", lex, args.chunk)
            labels = (str(args.a), str(args.b))
        result = compare(docs, lex, labels, args)
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n계산 결과를 {args.json}에 저장했습니다.")


if __name__ == "__main__":
    main()
