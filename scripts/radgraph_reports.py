"""RadGraph (modern-radgraph-xl) over the FINDINGS + IMPRESSION text of every MIMIC report.

Output: data/features/radgraph_reports/shard_NNNN.jsonl.gz, one JSON line per study:
  {study_id, subject_id, section_source, text (RadGraph's tokenised text), entities}
`entities` is RadGraph's raw output (tokens, label "Type::certainty", start_ix, end_ix, relations),
so negation scope and sentence membership can be resolved later from token positions.

Used for retrieval facts (Stage 5), concept selection and the Stage 7 round-trip check. Sections
other than FINDINGS/IMPRESSION are never parsed (AUDIT.md defect 11); `section_source` flags the 5%
of reports that needed the fallback (see nesy/reports.py).

Resumable: studies are processed in fixed shards (sorted by study_id); finished shards are skipped.
CPU: ~12 reports/s with 16 threads, batch 32 (runs/*_radgraph-speedtest).

  python -u scripts/radgraph_reports.py --threads 16 --batch 32
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from nesy import paths as P  # noqa: E402
from nesy.reports import ReportZip  # noqa: E402
from nesy.runlog import Progress, Run  # noqa: E402

OUT = P.FEATURES / "radgraph_reports"
SHARD = 2000
MODEL = "modern-radgraph-xl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--split", default=None, help="parse only this split's studies (study table) into radgraph_<split>_refs/")
    ap.add_argument("--cuda", type=int, default=-1, help="GPU index, -1 for CPU")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()
    global OUT
    if args.split:
        OUT = P.FEATURES / f"radgraph_{args.split}_refs"
    with Run("radgraph-reports", {**vars(args), "model": MODEL, "shard": SHARD, "out": str(OUT)}, args.run_dir) as run:
        import torch
        torch.set_num_threads(args.threads)
        from radgraph import RadGraph
        import radgraph
        run.config["radgraph_version"] = getattr(radgraph, "__version__", "0.1.18")

        studies = (pd.read_parquet(P.MANIFESTS / "mimic.parquet", columns=["subject_id", "study_id"])
                   .drop_duplicates("study_id").sort_values("study_id").reset_index(drop=True))
        if args.split:
            from nesy import data as D
            keep = set(D.study_table().query("split == @args.split").study_id)
            studies = studies[studies.study_id.isin(keep)].reset_index(drop=True)
            run.log(f"split {args.split}: {len(studies):,} studies (the pipeline's study table)")
        n_shards = (len(studies) + SHARD - 1) // SHARD
        if args.limit_shards:
            n_shards = min(n_shards, args.limit_shards)
        total = min(len(studies), n_shards * SHARD)
        OUT.mkdir(parents=True, exist_ok=True)
        done_shards = {int(p.stem.split("_")[1].split(".")[0]) for p in OUT.glob("shard_*.jsonl.gz")}
        run.log(f"studies {len(studies):,} in {n_shards} shards of {SHARD}; already done {len(done_shards & set(range(n_shards)))}")

        t = time.time()
        rg = RadGraph(model_type=MODEL, batch_size=args.batch, cuda=args.cuda)
        run.log(f"RadGraph loaded in {time.time() - t:.0f}s")
        rz = ReportZip()
        prog = Progress(run, total, "reports parsed", every_s=300)
        n_done = n_empty = 0
        src_counts: dict[str, int] = {}
        for si in range(n_shards):
            part = studies.iloc[si * SHARD:(si + 1) * SHARD]
            if si in done_shards:
                n_done += len(part)
                prog.update(n_done)
                continue
            secs = [rz.sections(s, st) for s, st in zip(part.subject_id, part.study_id)]
            texts = [x.text for x in secs]
            idx = [i for i, tx in enumerate(texts) if tx]
            n_empty += len(texts) - len(idx)
            parsed = rg([texts[i] for i in idx]) if idx else {}
            res = {i: parsed[str(k)] for k, i in enumerate(idx)}
            path = OUT / f"shard_{si:04d}.jsonl.gz"
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                for i, (s, st) in enumerate(zip(part.subject_id, part.study_id)):
                    r = res.get(i, {"text": "", "entities": {}})
                    src_counts[secs[i].source] = src_counts.get(secs[i].source, 0) + 1
                    fh.write(json.dumps({"study_id": int(st), "subject_id": int(s), "section_source": secs[i].source,
                                         "text": r.get("text", ""), "entities": r.get("entities", {})}) + "\n")
            tmp.replace(path)
            n_done += len(part)
            run.metric(shard=si, n=len(part), empty=len(texts) - len(idx))
            prog.update(n_done)
        prog.update(n_done, force=True)
        run.log(f"parsed {n_done:,} studies this or earlier runs; empty text skipped (this run) {n_empty}; "
                f"section sources (this run) {src_counts}")
        run.log(f"shards present: {len(list(OUT.glob('shard_*.jsonl.gz')))} / {n_shards}")


if __name__ == "__main__":
    main()
