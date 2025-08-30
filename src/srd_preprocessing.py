from __future__ import annotations


import json
import re


from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SRC_JSON  = Path("./data/raw/5esrd.json")
DST_JSONL = Path("./data/processed/ft/srd_chat_normal.jsonl")
DROP_KEYS      = {"content", "description", "text"}       
SCALARS        = (str, int, float, bool)              
REMOVE_PREFIXES = (
    r"^Appendix MM-B:\s*",
    r"^Appendix MM-A:\s*",
    r"^Appendix PH-A:\s*",
)


def markdown_table(tbl: Dict[str, List[str]]) -> str:
    headers = list(tbl)
    rows    = zip(*(tbl[h] for h in headers))
    lines   = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *[ "| " + " | ".join(r) + " |" for r in rows ],
    ]
    return "\n".join(lines) + "\n\n"


def breadcrumb(trail: List[str], key: str | None) -> List[str]:
    return [*trail, key] if key and key.lower() not in DROP_KEYS else trail


def squish_newlines(txt: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", txt)


def traverse(node: Any, trail: List[str] | None = None
             ) -> Iterable[Tuple[str, str]]:
    trail = trail or []

    if isinstance(node, dict):
        if set(node) == {"table"}:
            yield " > ".join(trail), markdown_table(node["table"])
            return
        for k, v in node.items():
            yield from traverse(v, breadcrumb(trail, k))
        return

    if isinstance(node, list):
        segs: List[str] = []
        for item in node:
            if isinstance(item, SCALARS) or item is None:
                segs.append("" if item is None else str(item))
            elif isinstance(item, dict) and set(item) == {"table"}:
                segs.append(markdown_table(item["table"]))
            else:
                yield from traverse(item, trail)
        if segs:
            joined = squish_newlines("\n".join(segs).rstrip())
            if joined:
                yield " > ".join(trail), joined
        return

    yield " > ".join(trail), str(node) if node is not None else ""


def sanitize_q(q: str) -> str | None:
    q = q.strip()

    # Keep the first “Monsters > ”, strip only the letter-group part
    q = re.sub(
        r"^Monsters\s*>\s*Monsters\s*\([A-Z]\)\s*>\s*",
        "Monsters > ",
        q,
    )

    # Strip other simple prefixes
    for patt in REMOVE_PREFIXES:
        q = re.sub(patt, "", q)

    return q.strip() or None


def build_pairs(tree: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for q_raw, a_raw in traverse(tree):
        q = sanitize_q(q_raw)
        a = a_raw.strip()
        if q and a and not q.startswith("Legal Information"):
            pairs.append((q, a))
    return pairs


def write_jsonl(pairs: List[Tuple[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for q, a in pairs:
            fh.write(json.dumps({"messages": [
                {"role": "user",      "content": q},
                {"role": "assistant", "content": a},
            ]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print("\nLoading SRD JSON")
    srd_tree = json.loads(SRC_JSON.read_text(encoding="utf-8"))

    print("\nTraversing & building pairs")
    qa_pairs = build_pairs(srd_tree)

    print(f"\nWriting {len(qa_pairs):,} pairs to {DST_JSONL}")
    write_jsonl(qa_pairs, DST_JSONL)

    print("\nDone")
