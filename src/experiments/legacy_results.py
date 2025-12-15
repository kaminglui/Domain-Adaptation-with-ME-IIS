from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple

from src.experiments.metrics import utc_timestamp
from src.experiments.run_config import compute_run_id, config_to_canonical_dict


def _pick(args: Mapping[str, Any], keys: Tuple[str, ...]) -> Dict[str, Any]:
    return {k: args.get(k) for k in keys}


def legacy_train_payload(args: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Canonical payload (and thus run_id) for `scripts/train_source.py`.

    Intentionally excludes runtime-only values like resume paths and dump_config.
    """
    payload = _pick(
        args,
        (
            "dataset_name",
            "source_domain",
            "target_domain",
            "num_epochs",
            "batch_size",
            "lr_backbone",
            "lr_classifier",
            "weight_decay",
            "seed",
            "deterministic",
            "dry_run_max_batches",
            "dry_run_max_samples",
        ),
    )
    payload.update({"entrypoint": "scripts/train_source.py", "method": "source_only"})
    return payload


def legacy_adapt_method_tag(args: Mapping[str, Any]) -> str:
    if bool(args.get("use_pseudo_labels", False)):
        return "me_iis_pl"
    if str(args.get("gmm_selection_mode", "fixed")) == "bic":
        return "me_iis_bic"
    return "me_iis"


def legacy_adapt_payload(args: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Canonical payload (and thus run_id) for `scripts/adapt_me_iis.py`.

    Intentionally excludes runtime-only values like resume paths and checkpoint paths.
    """
    payload = _pick(
        args,
        (
            "dataset_name",
            "source_domain",
            "target_domain",
            "batch_size",
            "num_workers",
            "num_latent_styles",
            "components_per_layer",
            "gmm_selection_mode",
            "gmm_bic_min_components",
            "gmm_bic_max_components",
            "cluster_backend",
            "vmf_kappa",
            "cluster_clean_ratio",
            "kmeans_n_init",
            "feature_layers",
            "source_prob_mode",
            "iis_iters",
            "iis_tol",
            "adapt_epochs",
            "finetune_backbone",
            "backbone_lr_scale",
            "classifier_lr",
            "weight_decay",
            "use_pseudo_labels",
            "pseudo_conf_thresh",
            "pseudo_max_ratio",
            "pseudo_loss_weight",
            "dry_run_max_samples",
            "dry_run_max_batches",
            "seed",
            "deterministic",
        ),
    )
    payload.update(
        {
            "entrypoint": "scripts/adapt_me_iis.py",
            "method": legacy_adapt_method_tag(args),
        }
    )
    return payload


def legacy_eval_source_only_payload(args: Mapping[str, Any]) -> Dict[str, Any]:
    payload = _pick(
        args,
        (
            "dataset_name",
            "domain",
            "batch_size",
            "num_workers",
            "seed",
            "deterministic",
        ),
    )
    payload.update({"entrypoint": "scripts/eval_source_only.py", "method": "source_only_self"})
    return payload


def legacy_run_id_and_config_json(payload: Mapping[str, Any]) -> Tuple[str, str]:
    rid = compute_run_id(payload)
    cfg_json = json.dumps(config_to_canonical_dict(payload), sort_keys=True, ensure_ascii=False)
    return rid, cfg_json


def legacy_timestamp_utc() -> str:
    return utc_timestamp()

