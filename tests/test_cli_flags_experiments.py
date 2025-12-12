import argparse
import unittest

from src.cli.args import ExperimentConfig, build_experiments_parser


def _action_map(parser: argparse.ArgumentParser):
    actions = {}
    for group in parser._action_groups:  # type: ignore[attr-defined]
        for action in group._group_actions:  # type: ignore[attr-defined]
            if action.option_strings and "-h" in action.option_strings:
                continue
            actions[action.dest] = action
    return actions


class TestExperimentCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_experiments_parser()
        self.base_args = [
            "--source_domain",
            "Ar",
            "--target_domain",
            "Cl",
            "--experiment_family",
            "gmm",
        ]

    def test_flags_present(self) -> None:
        actions = _action_map(self.parser)
        expected = {
            "dataset_name",
            "source_domain",
            "target_domain",
            "base_data_root",
            "seeds",
            "experiment_family",
            "output_csv",
            "num_epochs",
            "batch_size",
            "num_workers",
            "lr_backbone",
            "lr_classifier",
            "weight_decay",
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
            "pseudo_conf_thresh",
            "pseudo_max_ratio",
            "pseudo_loss_weight",
            "deterministic",
            "dry_run_max_samples",
            "dry_run_max_batches",
            "dump_config",
        }
        self.assertTrue(expected.issubset(set(actions.keys())))

    def test_defaults_and_config(self) -> None:
        parsed = self.parser.parse_args(self.base_args)
        cfg = ExperimentConfig(**vars(parsed))
        self.assertEqual(cfg.dataset_name, "office_home")
        self.assertEqual(cfg.gmm_selection_mode, "fixed")
        self.assertEqual(cfg.pseudo_max_ratio, 0.3)
        self.assertIsNone(cfg.dump_config)

    def test_each_flag_parses(self) -> None:
        actions = _action_map(self.parser)
        for dest, action in actions.items():
            if dest in {"source_domain", "target_domain", "experiment_family"}:
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
                        value = "value"
                    args_list.extend([action.option_strings[0], str(value)])
                parsed = self.parser.parse_args(args_list)
                if action.nargs == 0:
                    self.assertTrue(getattr(parsed, dest))


if __name__ == "__main__":
    unittest.main()
