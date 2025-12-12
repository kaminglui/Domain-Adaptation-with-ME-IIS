import argparse
import unittest

from src.cli.args import TrainConfig, build_train_parser


def _action_map(parser: argparse.ArgumentParser):
    actions = {}
    for group in parser._action_groups:  # type: ignore[attr-defined]
        for action in group._group_actions:  # type: ignore[attr-defined]
            if action.option_strings and "-h" in action.option_strings:
                continue
            actions[action.dest] = action
    return actions


class TestTrainCLI(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = build_train_parser()
        self.base_args = ["--source_domain", "Ar", "--target_domain", "Cl"]

    def test_flags_present(self) -> None:
        actions = _action_map(self.parser)
        expected = {
            "dataset_name",
            "data_root",
            "source_domain",
            "target_domain",
            "num_epochs",
            "resume_from",
            "save_every",
            "batch_size",
            "lr_backbone",
            "lr_classifier",
            "weight_decay",
            "num_workers",
            "deterministic",
            "seed",
            "dry_run_max_batches",
            "dry_run_max_samples",
            "eval_on_source_self",
            "eval_results_csv",
            "dump_config",
        }
        self.assertTrue(expected.issubset(set(actions.keys())))

    def test_defaults_and_config(self) -> None:
        parsed = self.parser.parse_args(self.base_args)
        cfg = TrainConfig(**vars(parsed))
        self.assertEqual(cfg.dataset_name, "office_home")
        self.assertEqual(cfg.num_epochs, 50)
        self.assertEqual(cfg.lr_classifier, 1e-2)
        self.assertEqual(cfg.deterministic, False)
        self.assertIsNone(cfg.dump_config)

    def test_each_flag_parses(self) -> None:
        actions = _action_map(self.parser)
        for dest, action in actions.items():
            if dest in {"source_domain", "target_domain"}:
                continue  # provided in base args
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
                        value = "/tmp/value" if dest.endswith("root") or "resume" in dest else "value"
                    args_list.extend([action.option_strings[0], str(value)])
                parsed = self.parser.parse_args(args_list)
                if action.nargs == 0:
                    self.assertTrue(getattr(parsed, dest))


if __name__ == "__main__":
    unittest.main()
