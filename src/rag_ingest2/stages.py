"""The generic Stage abstraction — hand-rolled checkpointing, deleted.

v1's pipeline.run() spent ~150 of its 500 lines on bespoke run-vs-load
branches; nothing enforced that the run branch and the load branch
produced equivalent state (theme E). Here the orchestrator owns
run-vs-load, timing, and artifact I/O uniformly: a stage defines run /
serialize / deserialize, and resume becomes impossible to get wrong
per-stage.

Guardrail (rewrite §6): the orchestrator knows about artifacts and
ordering, NEVER about page kinds, units, or tables. If a layer's logic
starts leaking in here, that is v1's disease returning under a new name.

Artifact format: ``serialize`` returns a JSON-able object; a list is
written as JSONL (greppable line-per-record), anything else as pretty
JSON. Every stage writes its complete artifact before the next stage
reads it — the files remain the learning instrument.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import IngestConfig

log = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """What every stage can see. ``results`` accumulates each stage's
    (deserialized) output under its stage name."""

    pdf_path: Path
    doc_out: Path
    cfg: IngestConfig
    results: dict[str, object] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class StageSpec:
    """One pipeline layer, as the orchestrator sees it."""

    name: str
    artifact: str  # file name under stages/
    run: Callable[[PipelineContext], object]
    serialize: Callable[[object], object]  # result -> JSON-able
    deserialize: Callable[[object, PipelineContext], object]  # JSON -> result


def _write_artifact(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, list):
        with path.open("w", encoding="utf-8") as f:
            for row in payload:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("stage artifact -> %s (%d records)", path, len(payload))
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("stage artifact -> %s", path)


def _read_artifact(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def run_stages(specs: list[StageSpec], ctx: PipelineContext, from_stage: int = 1) -> None:
    """Run (or load) every stage in order. Stage N is 1-based;
    ``from_stage=N`` loads artifacts for stages 1..N-1 and runs the rest.
    Loading goes through the same deserialize the artifact was written
    for, so a resumed run and a fresh run hand downstream stages the
    same shapes by construction."""
    for i, spec in enumerate(specs, start=1):
        path = ctx.doc_out / "stages" / spec.artifact
        if i < from_stage:
            ctx.results[spec.name] = spec.deserialize(_read_artifact(path), ctx)
            log.info("stage %d (%s) skipped, loaded artifact", i, spec.name)
            continue
        t0 = time.perf_counter()
        result = spec.run(ctx)
        ctx.timings[spec.name] = round(time.perf_counter() - t0, 3)
        ctx.results[spec.name] = result
        _write_artifact(path, spec.serialize(result))
