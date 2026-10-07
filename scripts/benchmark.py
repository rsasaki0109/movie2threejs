"""Convert measured run reports to a Markdown benchmark; never fill missing times."""
import argparse
import json
from pathlib import Path


def render(report, description):
    source = report["source"]
    lines = ["# Measured capture run", "", description, "",
             f"- Status: `{report['status']}`",
             f"- Started (UTC): `{report['started_utc']}`",
             f"- GPU: {report['gpu'] or 'not detected'}",
             f"- Video: `{source['name']}`",
             f"- Video SHA-256: `{source['sha256']}`", "",
             "| Stage | Status | Measured seconds |", "|---|---|---:|"]
    for stage in ["frames", "poses", "train", "segment", "world"]:
        item = report["stages"].get(stage, {})
        seconds = item.get("seconds")
        lines.append(f"| {stage} | {item.get('status', 'not run')} | {seconds if seconds is not None else '—'} |")
    stats = report.get("stats", {})
    lines += ["", f"Objects: {stats.get('objects', 'not measured')}. Static colliders: {stats.get('colliders', 'not measured')}."]
    if report.get("total_seconds") is not None:
        lines.append(f"Total for this invocation: {report['total_seconds']} s (setup and reused stages excluded).")
    lines += ["", "## Settings", "", "```json", json.dumps(report["settings"], indent=2), "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--description", required=True, help="actual capture ownership/scene description")
    parser.add_argument("--out", default=Path("docs/benchmarks.md"), type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(json.loads(args.report.read_text()), args.description), encoding="utf-8")
