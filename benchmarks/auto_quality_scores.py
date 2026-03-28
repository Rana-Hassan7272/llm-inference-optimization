import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = ROOT / "README.md"

TIER_FILES = {
    "4-bit (fast)": RESULTS / "4bit_results.json",
    "8-bit (balanced)": RESULTS / "8bit_results.json",
    "FP16 (quality)": RESULTS / "fp16_results.json",
}


def load_results(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def heuristic_quality(question: str, answer: str) -> float:
    """
    Returns 0.0-1.0 correctness for a subset of factual prompts.
    If no rule matches, returns 0.0 (unknown).
    """
    q = (question or "").strip().lower()
    a = (answer or "").strip().lower()

    def contains_any(s: str, terms: List[str]) -> bool:
        return any(t in s for t in terms)

    # Capital of France → Paris
    if "capital of france" in q:
        return 1.0 if "paris" in a else 0.0

    # Boiling point of water in Celsius → 100
    if "boiling point of water" in q and ("celsius" in q or "°c" in q or "c" in q):
        return 1.0 if re.search(r"\b100\b", a) else 0.0

    # Who wrote Hamlet → Shakespeare
    if "hamlet" in q and contains_any(q, ["who wrote", "who was the author"]):
        return 1.0 if "shakespeare" in a else 0.0

    # 17 multiplied by 13 → 221
    if contains_any(q, ["17 multiplied by 13", "17 x 13", "17 * 13"]):
        return 1.0 if re.search(r"\b221\b", a) else 0.0

    # Largest planet → Jupiter
    if "largest planet" in q and "solar system" in q:
        return 1.0 if "jupiter" in a else 0.0

    # HTTP 404 → Not Found
    if "http status code 404" in q or ("404" in q and "http" in q):
        return 1.0 if "not found" in a else 0.0

    # GPU stands for → Graphics Processing Unit
    if "what does gpu stand for" in q or ("gpu" in q and "stand for" in q):
        return 1.0 if ("graphics processing unit" in a) else 0.0

    # Year WW2 ended → 1945
    if contains_any(q, ["what year did world war ii end", "what year did world war 2 end"]):
        return 1.0 if re.search(r"\b1945\b", a) else 0.0

    # Fallback: unknown
    return 0.0


def tokenize(s: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", s.lower(), flags=re.UNICODE)


def bleu1_score(hypo: str, refs: List[str]) -> float:
    """
    Simple BLEU-1 (unigram precision with brevity penalty), scaled 0..1.
    """
    hyp_toks = tokenize(hypo)
    if not hyp_toks:
        return 0.0
    # Choose the best reference for precision overlap
    best = 0.0
    for ref in refs:
        ref_toks = tokenize(ref)
        ref_counts: Dict[str, int] = {}
        for t in ref_toks:
            ref_counts[t] = ref_counts.get(t, 0) + 1
        match = 0
        used: Dict[str, int] = {}
        for t in hyp_toks:
            c = used.get(t, 0)
            if c < ref_counts.get(t, 0):
                match += 1
                used[t] = c + 1
        precision = match / len(hyp_toks)
        # Brevity penalty
        r = len(ref_toks)
        c = len(hyp_toks)
        bp = 1.0 if c > r else (0.0 if r == 0 else min(1.0, c / r))
        score = bp * precision
        best = max(best, score)
    return best


# Minimal reference set for factual prompts
BLEU_REFS: Dict[str, List[str]] = {
    "What is the capital of France?".lower(): ["Paris"],
    "What is the boiling point of water at sea level in Celsius?".lower(): ["100", "100 c", "100 °c", "100 degrees celsius"],
    "Who wrote the play Hamlet?".lower(): ["William Shakespeare", "Shakespeare"],
    "What is 17 multiplied by 13?".lower(): ["221", "17 x 13 is 221", "17 * 13 is 221"],
    "What is the largest planet in our solar system?".lower(): ["Jupiter"],
    "What is HTTP status code 404?".lower(): ["Not Found", "404 Not Found"],
    "What year did World War II end?".lower(): ["1945"],
    "What does GPU stand for and why is it useful for AI?".lower(): ["Graphics Processing Unit"],
}


def aggregate_tier(entries: List[Dict]) -> Tuple[float, float, float, float, float, float]:
    ttft = [float(e.get("ttft_ms", 0.0)) for e in entries if isinstance(e.get("ttft_ms", None), (int, float))]
    tps = [float(e.get("tok_per_sec", 0.0)) for e in entries if isinstance(e.get("tok_per_sec", None), (int, float))]
    vram = [float(e.get("mem_gb", 0.0)) for e in entries if isinstance(e.get("mem_gb", None), (int, float))]

    # Heuristic quality scaled to 1–5
    hq = []
    bleu_scores: List[float] = []
    for e in entries:
        q = e.get("question", "")
        a = e.get("answer", "")
        hq.append(heuristic_quality(q, a) * 5.0)
        # BLEU-1 against reference set when available
        refs = BLEU_REFS.get((q or "").strip().lower())
        if refs:
            bleu_scores.append(bleu1_score(a or "", refs))

    # BLEU reported on 0..100 scale; also compute coverage
    bleu_avg_pct = mean(bleu_scores) * 100.0 if bleu_scores else 0.0
    bleu_cov = (len(bleu_scores) / len(entries)) * 100.0 if entries else 0.0

    return mean(vram), mean(ttft), mean(tps), mean(hq), bleu_avg_pct, bleu_cov


def build_table_data() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for tier, path in TIER_FILES.items():
        data = load_results(path)
        if not data:
            out[tier] = {}
            continue
        vram, ttft, tps, qual, bleu1, bleu_cov = aggregate_tier(data)
        out[tier] = {
            "vram_gb": round(vram, 2),
            "ttft_ms": round(ttft, 0),
            "tps": round(tps, 2),
            "auto_quality": round(qual, 2),
            "bleu1": round(bleu1, 2),
            "bleu_coverage_pct": round(bleu_cov, 1),
        }
    return out


def write_manifest(table: Dict[str, Dict[str, float]]) -> Path:
    out_path = RESULTS / "ablation_table.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    return out_path


START_MARK = "<!-- ABLATION_TABLE_START -->"
END_MARK = "<!-- ABLATION_TABLE_END -->"


def render_markdown(table: Dict[str, Dict[str, float]]) -> str:
    # If any tier is missing, we will note it inline.
    lines = []
    lines.append("### Ablation — Precision vs VRAM, TTFT, TPS, Quality (Auto + BLEU-1)")
    lines.append("")
    lines.append("Computed from `results/*_results.json`. Quality is a heuristic factual-correctness score (1–5). BLEU-1 uses a small reference set for factual prompts only.")
    lines.append("")
    lines.append("| Tier | VRAM (GB) | Avg TTFT (ms) | Avg TPS (tok/s) | Auto Quality (1–5) | BLEU-1 (0–100) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    order = ["4-bit (fast)", "8-bit (balanced)", "FP16 (quality)"]
    for tier in order:
        row = table.get(tier, {})
        if not row:
            lines.append(f"| {tier} | — | — | — | — | — |")
        else:
            lines.append(
                f"| {tier} | {row['vram_gb']:.2f} | {row['ttft_ms']:.0f} | {row['tps']:.2f} | {row['auto_quality']:.2f} | {row['bleu1']:.2f} |"
            )
    lines.append("")
    # Coverage note
    covs = [table.get(t, {}).get("bleu_coverage_pct") for t in order if table.get(t)]
    cov_note = f"BLEU coverage: ~{round(mean([c for c in covs if isinstance(c, (int, float))]), 1)}% of prompts have references." if covs else "BLEU coverage: 0% (no matching prompts)."
    lines.append(f"Notes: This is a lightweight proxy; BLEU computed only on factual prompts. {cov_note}")
    return "\n".join(lines)


def update_readme(rendered: str) -> None:
    text = README.read_text(encoding="utf-8")
    if START_MARK in text and END_MARK in text:
        pre = text.split(START_MARK, 1)[0]
        post = text.split(END_MARK, 1)[1]
        new_text = f"{pre}{START_MARK}\n\n{rendered}\n\n{END_MARK}{post}"
    else:
        # Append at the end if markers are missing
        new_text = f"{text.rstrip()}\n\n{START_MARK}\n\n{rendered}\n\n{END_MARK}\n"
    README.write_text(new_text, encoding="utf-8")


def main():
    table = build_table_data()
    write_manifest(table)
    rendered = render_markdown(table)
    update_readme(rendered)
    print("Ablation table generated and README updated.")


if __name__ == "__main__":
    main()

