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


if __name__ == "__main__":
    unittest.main()
