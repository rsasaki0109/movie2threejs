"""Run pinned VGGT with bounded tracking batches, preserving all query points."""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--vggt-dir", type=Path, required=True)
    parser.add_argument("--track-budget", type=int, default=32768)
    args, upstream_args = parser.parse_known_args()
    if args.track_budget < 1:
        raise ValueError("tracking budget must be positive")
    sys.path.insert(0, str(args.vggt_dir.resolve()))
    import demo_colmap
    import torch
    import numpy as np

    # Upstream chunks query points using frames * points. Bound that workload
    # for every query, including its fallback for poorly visible frames.
    demo_colmap.predict_tracks = partial(demo_colmap.predict_tracks, max_points_num=args.track_budget)
    sys.argv = ["demo_colmap.py", *upstream_args]
    configuration = demo_colmap.parse_args()
    original = demo_colmap.run_VGGT
    def measured_prediction(*prediction_args, **prediction_kwargs):
        output = original(*prediction_args, **prediction_kwargs)
        confidence = output[3]
        report = {
            "confidence_percentiles": dict(zip(["min", "p10", "p50", "p90", "p99", "max"],
                                                np.percentile(confidence, [0, 10, 50, 90, 99, 100]).tolist())),
            "confidence_threshold": configuration.conf_thres_value,
            "samples_above_threshold": int(np.count_nonzero(confidence >= configuration.conf_thres_value)),
            "finite_depth": bool(np.isfinite(output[2]).all()),
            "tracking_batch_budget": args.track_budget if configuration.use_ba else None,
        }
        (Path(configuration.scene_dir) / "pose-diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
        print("VGGT diagnostics:", json.dumps(report), flush=True)
        return output
    demo_colmap.run_VGGT = measured_prediction
    print(f"VGGT tracking batch budget: {args.track_budget} frame-point pairs", flush=True)
    with torch.no_grad():
        demo_colmap.demo_fn(configuration)


if __name__ == "__main__":
    main()
