"""Command-line interface for SiteGuard V4."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from . import __version__
from .assets import validate_asset_root
from .decision import LEVELS, decide_highest_supported_resolution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="siteguard",
        description="Evidence-calibrated enzyme catalytic-function annotation transfer",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="validate frozen model and reference assets")
    doctor.add_argument("--asset-root", type=Path, required=True)

    embed = commands.add_parser("embed", help="compute exact-sequence ESM2-t33 query embeddings")
    embed.add_argument("--input-fasta", type=Path, required=True)
    embed.add_argument("--model", type=Path, required=True)
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--index", type=Path, required=True)

    predict = commands.add_parser("predict", help="retrieve references, score candidates, and safely abstain")
    predict.add_argument("--input-fasta", type=Path, required=True)
    predict.add_argument("--asset-root", type=Path, required=True)
    predict.add_argument("--query-embeddings", type=Path, required=True)
    predict.add_argument("--embedding-index", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--workdir", type=Path, required=True)
    predict.add_argument("--query-pfam", type=Path)
    predict.add_argument("--mmseqs", default="mmseqs")
    predict.add_argument("--threads", type=int, default=8)
    predict.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    propose = commands.add_parser(
        "propose-reactions", help="rank the complete known Rhea catalog with the locked ChemBridge model"
    )
    propose.add_argument("--asset-root", type=Path, required=True)
    propose.add_argument("--query-embeddings", type=Path, required=True)
    propose.add_argument("--embedding-index", type=Path, required=True)
    propose.add_argument("--output", type=Path, required=True)
    propose.add_argument("--input-fasta", type=Path)
    propose.add_argument("--top-k", type=int, default=10)
    propose.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    decide = commands.add_parser("decide", help="apply frozen hierarchy to precomputed top-label probabilities")
    decide.add_argument("--input", type=Path, required=True)
    decide.add_argument("--calibration-config", type=Path, required=True)
    decide.add_argument("--output", type=Path, required=True)

    judge = commands.add_parser(
        "evidence-judge",
        help="score HIT-EC candidate reliability and defer when cohort applicability is not established",
    )
    judge.add_argument("--input", type=Path, required=True, help="83-feature Parquet, TSV, or CSV table")
    judge.add_argument("--output", type=Path, required=True, help="judgement Parquet, TSV, or CSV table")
    judge.add_argument("--model-spec", type=Path, help="optional portable EvidenceJudge JSON spec")
    judge.add_argument(
        "--cohort-status", choices=["validated-similar", "unknown", "shifted"], default="unknown",
        help="cohort-level applicability state; unknown/shifted forces final DEFER",
    )
    judge.add_argument(
        "--cohort-audit", type=Path,
        help="optional hash-bound audit JSON; a shifted audit overrides any requested status",
    )

    audit = commands.add_parser(
        "evidence-judge-audit",
        help="run a truth-free cohort-shift audit that can veto, but never enable, acceptance",
    )
    audit.add_argument("--reference", type=Path, required=True, help="validated reference feature table")
    audit.add_argument("--target", type=Path, required=True, help="new cohort feature table")
    audit.add_argument("--output", type=Path, required=True, help="hash-bound applicability-audit JSON")
    audit.add_argument("--model-spec", type=Path, help="optional portable EvidenceJudge JSON spec")
    audit.add_argument("--permutations", type=int, default=1000)
    audit.add_argument("--bootstraps", type=int, default=10000)
    audit.add_argument("--seed", type=int, default=20261111)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        result = validate_asset_root(args.asset_root)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 2
    if args.command == "embed":
        from .embed import embed_fasta

        embed_fasta(args.input_fasta, args.model, args.output, args.index)
        return 0
    if args.command == "predict":
        from .predictor import predict_fasta

        result = predict_fasta(
            args.input_fasta, args.asset_root, args.query_embeddings, args.embedding_index,
            args.output, args.workdir, args.query_pfam, args.mmseqs, args.threads, args.device,
        )
        print(json.dumps({
            "status": "PASS", "queries": len(result),
            "resolution_counts": result["final_resolution"].value_counts().to_dict(),
            "output": str(args.output.resolve()),
        }, indent=2))
        return 0
    if args.command == "propose-reactions":
        from .chembridge import propose_reactions

        result = propose_reactions(
            args.asset_root, args.query_embeddings, args.embedding_index, args.output,
            args.top_k, args.input_fasta, args.device,
        )
        print(json.dumps({
            "status": "PASS", "queries": int(result["query_protein_id"].nunique()),
            "panel_rows": len(result), "top_k": args.top_k,
            "model": str(result["model"].iloc[0]) if len(result) else None,
            "output": str(args.output.resolve()),
        }, indent=2))
        return 0
    if args.command == "decide":
        calibration = json.loads(args.calibration_config.read_text(encoding="utf-8"))
        thresholds = calibration["thresholds"]
        with args.input.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            input_columns = list(reader.fieldnames or [])
            input_rows = list(reader)
        rows: list[dict[str, object]] = []
        for row in input_rows:
            probabilities = {level: float(row[f"top_probability_{level}"]) for level in LEVELS}
            labels = {level: str(row.get(f"top_label_{level}", "")) for level in LEVELS}
            rows.append({**row, **decide_highest_supported_resolution(probabilities, labels, thresholds)})
        decision_columns = [
            "final_resolution", "final_label", "final_probability", "abstention_reason",
            *[f"accepted_{level}" for level in LEVELS],
        ]
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*input_columns, *decision_columns], delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        return 0
    if args.command == "evidence-judge":
        from .evidencejudge import judge_frame, load_spec, read_feature_table, write_judgements

        source = read_feature_table(args.input)
        _, _, spec_sha256 = load_spec(args.model_spec)
        effective_status = args.cohort_status
        audit_sha256 = None
        if args.cohort_audit is not None:
            from .applicability import apply_audit_veto

            effective_status, _, audit_sha256 = apply_audit_veto(
                args.cohort_status,
                args.cohort_audit,
                target_path=args.input,
                model_spec_sha256=spec_sha256,
            )
        result = judge_frame(source, cohort_status=effective_status, spec_path=args.model_spec)
        result["requested_cohort_status"] = args.cohort_status
        result["cohort_audit"] = str(args.cohort_audit.resolve()) if args.cohort_audit else ""
        result["cohort_audit_sha256"] = audit_sha256 or ""
        write_judgements(result, args.output)
        print(json.dumps({
            "status": "PASS",
            "rows": len(result),
            "row_threshold_accepts": int(result["frozen_threshold_accept"].sum()),
            "final_accepts": int(result["final_decision"].eq("ACCEPT").sum()),
            "final_defers": int(result["final_decision"].eq("DEFER").sum()),
            "requested_cohort_status": args.cohort_status,
            "effective_cohort_status": effective_status,
            "cohort_audit_sha256": audit_sha256,
            "output": str(args.output.resolve()),
            "warning": "A row score is not a portable 95% guarantee; unknown or shifted cohorts are deferred.",
        }, indent=2))
        return 0
    if args.command == "evidence-judge-audit":
        from .applicability import audit_cohort_files

        result = audit_cohort_files(
            args.reference,
            args.target,
            output_path=args.output,
            spec_path=args.model_spec,
            seed=args.seed,
            permutations=args.permutations,
            bootstraps=args.bootstraps,
        )
        print(json.dumps({
            "status": "PASS",
            "automatic_cohort_status": result["automatic_cohort_status"],
            "scope": result["automatic_status_scope"],
            "output": str(args.output.resolve()),
            "warning": "This audit can veto acceptance but cannot establish validated-similar status.",
        }, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
