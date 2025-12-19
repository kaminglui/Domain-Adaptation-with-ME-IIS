from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional

from src.camelyon17.runner import Camelyon17Runner, Camelyon17RunnerConfig, Camelyon17SplitConfig


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run Camelyon17 (WILDS) experiments.")
    p.add_argument("--dataset", type=str, default="camelyon17", choices=["camelyon17"])
    p.add_argument(
        "--methods",
        action="append",
        default=[],
        help="Comma-separated or repeatable list: source_only,dann,me_iis",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_mode", type=str, default="uda_target", choices=["uda_target", "align_val"])
    p.add_argument(
        "--adapt_split",
        type=str,
        default="auto",
        choices=["auto", "val_unlabeled", "test_unlabeled"],
        help="Target unlabeled split (auto derives from split_mode).",
    )
    p.add_argument(
        "--eval_split",
        type=str,
        default="auto",
        choices=["auto", "val", "test"],
        help="Labeled eval split (auto derives from split_mode).",
    )
    p.add_argument("--force_rerun", action="store_true")

    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--runs_root", type=str, default=str(Path("outputs") / "runs"))

    p.add_argument("--batch_size", type=str, default="64")
    p.add_argument("--num_workers", type=str, default="auto")
    p.add_argument("--epochs_source", type=int, default=5)
    p.add_argument("--epochs_adapt", type=int, default=5)
    p.add_argument("--max_epochs", type=int, default=0, help="Convenience alias: sets both epochs_source and epochs_adapt")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--dry_run_max_batches", type=int, default=0)
    p.add_argument("--dry_run_max_samples", type=int, default=0)
    p.add_argument("--disable_checkpointing", action="store_true")
    p.add_argument("--skip_download", action="store_true")
    args = p.parse_args(argv)

    batch_size_raw = str(args.batch_size).strip()
    if batch_size_raw.lower() == "auto":
        batch_size: int | str = "auto"
    else:
        batch_size = int(batch_size_raw)
    epochs_source = int(args.epochs_source)
    epochs_adapt = int(args.epochs_adapt)
    if int(args.max_epochs) > 0:
        epochs_source = int(args.max_epochs)
        epochs_adapt = int(args.max_epochs)

    dry_run_max_batches = int(args.dry_run_max_batches)
    dry_run_max_samples = int(args.dry_run_max_samples)

    if bool(args.smoke_test):
        epochs_source = 1
        epochs_adapt = 1
        dry_run_max_batches = 2
        # Reduce OOM risk in smoke tests.
        # Reduce OOM risk in smoke tests.
        batch_size = min(int(16 if isinstance(batch_size, str) else batch_size), 16)
        # Keep sample caps small so ME-IIS source/target passes stay fast.
        if dry_run_max_samples <= 0:
            dry_run_max_samples = int(dry_run_max_batches) * max(1, int(batch_size))

    split_mode = str(args.split_mode)
    adapt_split = str(args.adapt_split)
    eval_split = str(args.eval_split)
    if adapt_split == "auto":
        adapt_split = "val_unlabeled" if split_mode == "align_val" else "test_unlabeled"
    if eval_split == "auto":
        eval_split = "val" if split_mode == "align_val" else "test"

    split = Camelyon17SplitConfig(
        split_mode=split_mode,  # type: ignore[arg-type]
        adapt_split=adapt_split,  # type: ignore[arg-type]
        eval_split=eval_split,  # type: ignore[arg-type]
    )
    cfg = Camelyon17RunnerConfig(
        dataset=str(args.dataset),
        data_root=(None if str(args.data_root).strip() == "" else str(args.data_root)),
        download=not bool(args.skip_download),
        runs_root=Path(str(args.runs_root)),
        seed=int(args.seed),
        num_workers=(str(args.num_workers).strip().lower() if str(args.num_workers).strip() != "" else "auto"),
        epochs_source=int(epochs_source),
        epochs_adapt=int(epochs_adapt),
        dry_run_max_batches=int(dry_run_max_batches),
        dry_run_max_samples=int(dry_run_max_samples),
        disable_checkpointing=bool(args.disable_checkpointing),
        meiis=(
            replace(Camelyon17RunnerConfig().meiis, debug=True) if bool(args.smoke_test) else Camelyon17RunnerConfig().meiis
        ),
    )
    runner = Camelyon17Runner(cfg=cfg, split=split)
    try:
        res = runner.run(methods=args.methods, batch_size=batch_size, force_rerun=bool(args.force_rerun))
    except Exception as exc:
        if bool(args.skip_download):
            print("[DATA][MISSING] Camelyon17 WILDS dataset not found and --skip_download was set.")
            print(
                "Provide an existing dataset root via `--data_root ...` or set `WILDS_DATA_ROOT`, "
                "or rerun without `--skip_download` to allow WILDS to download."
            )
            print(f"Error: {type(exc).__name__}: {exc}")
            return 2
        raise

    if bool(args.smoke_test):
        from src.models import build_backbone

        bb = build_backbone(
            cfg.backbone,
            pretrained=bool(cfg.pretrained),
            replace_batchnorm_with_instancenorm=bool(cfg.replace_batchnorm_with_instancenorm),
        )
        feature_dim = int(bb.feature_dim)
        if "me_iis" in res:
            import json

            run_dir = Path(str(res["me_iis"]["run_dir"]))
            payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
            updates = ((payload.get("meiis", {}) or {}).get("weight_updates", []) or [])
            last = None
            for u in reversed(updates):
                if isinstance(u, dict) and u.get("status") == "updated":
                    last = u
                    break
            target_selected = None if last is None else last.get("target_selected")
            ess = None if last is None else last.get("ess")
            obj = [] if last is None else (last.get("iis_objective") or [])
            print(
                "SMOKE TEST PASSED "
                f"feature_dim={feature_dim} "
                f"split_mode={split.split_mode} adapt_split={split.adapt_split} eval_split={split.eval_split} "
                f"Pg_selected_count={target_selected} iis_iters={len(obj)} ESS={ess}"
            )
        else:
            print(
                "SMOKE TEST PASSED "
                f"feature_dim={feature_dim} "
                f"split_mode={split.split_mode} adapt_split={split.adapt_split} eval_split={split.eval_split}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
