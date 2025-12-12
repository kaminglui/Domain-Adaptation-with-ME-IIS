import argparse
import unittest

from src.cli.args import AdaptConfig, build_adapt_parser


def _action_map(parser: argparse.ArgumentParser):
    actions = {}
    for group in parser._action_groups:  # type: ignore[attr-defined]
        for action in group._group_actions:  # type: ignore[attr-defined]
            if action.option_strings and "-h" in action.option_strings:
                continue
            actions[action.dest] = action
    return actions


class TestAdaptCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_adapt_parser()
        self.base_args = [
            "--source_domain",
            "Ar",
            "--target_domain",
            "Cl",
            "--checkpoint",
            "/tmp/ckpt.pth",
        ]

    def test_flags_present(self) -> None:
        actions = _action_map(self.parser)
        expected = {
            "dataset_name",
            "data_root",
            "source_domain",
            "target_domain",
            "checkpoint",
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
            "resume_adapt_from",
            "save_adapt_every",
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
            "deterministic",
            "seed",
            "dump_config",
        }
        self.assertTrue(expected.issubset(set(actions.keys())))

    def test_defaults_and_config(self) -> None:
        parsed = self.parser.parse_args(self.base_args)
        cfg = AdaptConfig(**vars(parsed))
        self.assertEqual(cfg.cluster_backend, "gmm")
        self.assertEqual(cfg.gmm_selection_mode, "fixed")
        self.assertEqual(cfg.iis_iters, 15)
        self.assertFalse(cfg.finetune_backbone)
        self.assertIsNone(cfg.dump_config)

    def test_each_flag_parses(self) -> None:
        actions = _action_map(self.parser)
        for dest, action in actions.items():
            if dest in {"source_domain", "target_domain", "checkpoint"}:
                continue
            with self.subTest(flag=dest):
                args_list = list(self.base_args)
                if action.nargs == 0:
                    args_list.append(action.option_strings[0])
                else:
                    if action.choices:
                        value = list(action.choices)[0]
                    elif action.type is int:
                        value = (action.default or 0) + 1
                    elif action.type is float:
                        value = (action.default or 0.0) + 0.5
                    else:
                        value = "/tmp/value" if "path" in dest or "checkpoint" in dest else "value"
                    args_list.extend([action.option_strings[0], str(value)])
                parsed = self.parser.parse_args(args_list)
                if action.nargs == 0:
                    self.assertTrue(getattr(parsed, dest))


if __name__ == "__main__":
    unittest.main()
