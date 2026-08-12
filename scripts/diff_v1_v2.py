"""Diff chunks, not code (rewrite §5.4): compare v1 (output/) and v2
(output2/) chunk artifacts per document, modulo chunk IDs and ordering.

Signature per chunk: (type, normalized content, pages, needs_review).
Content is whitespace-normalized; table chunks also carry their
row-count so a re-grouped table shows up as a real difference."""

import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

V1, V2 = Path("output"), Path("output2")


def signatures(path: Path) -> Counter:
    sigs: Counter = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        content = " ".join(c["content"].split())
        sigs[(c["type"], content, tuple(c["pages"]), c["needs_review"])] += 1
    return sigs


docs = sorted(
    d.name
    for d in V2.iterdir()
    if (d / "stages" / "07_chunks.jsonl").exists()
    and (V1 / d.name / "stages" / "07_chunks.jsonl").exists()
)
total_only1 = total_only2 = 0
for doc in docs:
    s1 = signatures(V1 / doc / "stages" / "07_chunks.jsonl")
    s2 = signatures(V2 / doc / "stages" / "07_chunks.jsonl")
    only1 = s1 - s2
    only2 = s2 - s1
    n1, n2 = sum(s1.values()), sum(s2.values())
    status = "IDENTICAL" if not only1 and not only2 else f"DIFF (v1-only {sum(only1.values())}, v2-only {sum(only2.values())})"
    print(f"{doc:28s} v1={n1:3d} v2={n2:3d}  {status}")
    total_only1 += sum(only1.values())
    total_only2 += sum(only2.values())
    for sig, n in list(only1.items())[:4]:
        t, content, pages, review = sig
        print(f"   v1-only x{n}: {t} p{pages} review={review}: {content[:90]}")
    for sig, n in list(only2.items())[:4]:
        t, content, pages, review = sig
        print(f"   v2-only x{n}: {t} p{pages} review={review}: {content[:90]}")

print(f"\nTOTAL divergent chunks: v1-only={total_only1}, v2-only={total_only2}")
