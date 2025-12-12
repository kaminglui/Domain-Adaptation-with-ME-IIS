import unittest

from clustering.factory import create_backend
from clustering.gmm_backend import GMMBackend
from clustering.vmf_softmax_backend import VMFSoftmaxBackend
from src.cli.args import build_adapt_parser, build_experiments_parser, build_train_parser


class TestCLIInvalidArgs(unittest.TestCase):
    def test_invalid_choice_raises(self) -> None:
        train_parser = build_train_parser()
        with self.assertRaises(SystemExit):
            train_parser.parse_args(["--dataset_name", "invalid", "--source_domain", "Ar", "--target_domain", "Cl"])

        adapt_parser = build_adapt_parser()
        with self.assertRaises(SystemExit):
            adapt_parser.parse_args(
                [
                    "--source_domain",
                    "Ar",
                    "--target_domain",
                    "Cl",
                    "--checkpoint",
                    "/tmp/x.pth",
                    "--cluster_backend",
                    "invalid",
                ]
            )

        exp_parser = build_experiments_parser()
        with self.assertRaises(SystemExit):
            exp_parser.parse_args(
                ["--source_domain", "Ar", "--target_domain", "Cl", "--experiment_family", "bad_family"]
            )

    def test_backend_factory_selection(self) -> None:
        gmm = create_backend("gmm", n_components=2, seed=0)
        self.assertIsInstance(gmm, GMMBackend)
        vmf = create_backend("vmf_softmax", n_components=3, seed=0, vmf_kappa=10.0)
        self.assertIsInstance(vmf, VMFSoftmaxBackend)

    def test_adapt_parser_accepts_vmf_flags(self) -> None:
        parser = build_adapt_parser()
        args = parser.parse_args(
            [
                "--source_domain",
                "Ar",
                "--target_domain",
                "Cl",
                "--checkpoint",
                "/tmp/x.pth",
                "--cluster_backend",
                "vmf_softmax",
                "--vmf_kappa",
                "25",
                "--cluster_clean_ratio",
                "0.8",
            ]
        )
        self.assertEqual(args.cluster_backend, "vmf_softmax")
        self.assertAlmostEqual(args.vmf_kappa, 25.0)
        self.assertAlmostEqual(args.cluster_clean_ratio, 0.8)


if __name__ == "__main__":
    unittest.main()
