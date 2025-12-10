import unittest

from scripts.run_me_iis_experiments import parse_seeds
from utils.experiment_utils import (
    build_components_map,
    dataset_tag,
    normalize_dataset_name,
    parse_feature_layers,
)


class TestDatasetNormalization(unittest.TestCase):
    def test_normalize_dataset_name_variants(self) -> None:
        self.assertEqual(normalize_dataset_name("Office-Home"), "officehome")
        self.assertEqual(normalize_dataset_name("office_home"), "officehome")
        self.assertEqual(normalize_dataset_name("Office Home"), "officehome")

    def test_dataset_tag_office_home(self) -> None:
        self.assertEqual(dataset_tag("office_home"), "office-home")
        self.assertEqual(dataset_tag("Office-Home"), "office-home")
        self.assertEqual(dataset_tag("Office Home"), "office-home")

    def test_dataset_tag_office31(self) -> None:
        self.assertEqual(dataset_tag("office31"), "office-31")
        self.assertEqual(dataset_tag("Office-31"), "office-31")
        self.assertEqual(dataset_tag("Office 31"), "office-31")

    def test_dataset_tag_unknown(self) -> None:
        self.assertEqual(dataset_tag("cifar10"), "cifar10")


class TestParseFeatureLayers(unittest.TestCase):
    def test_parse_layers_basic(self) -> None:
        self.assertEqual(parse_feature_layers("layer3,layer4,avgpool"), ["layer3", "layer4", "avgpool"])

    def test_parse_layers_whitespace(self) -> None:
        self.assertEqual(parse_feature_layers("layer3, layer4, avgpool"), ["layer3", "layer4", "avgpool"])

    def test_parse_layers_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_feature_layers("")
        with self.assertRaises(ValueError):
            parse_feature_layers("   ")


class TestBuildComponentsMap(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_layers = ["layer3", "layer4", "avgpool"]

    def test_no_override(self) -> None:
        expected = {"layer3": 5, "layer4": 5, "avgpool": 5}
        self.assertEqual(build_components_map(self.feature_layers, 5, None), expected)

    def test_explicit_pairs(self) -> None:
        expected = {"layer3": 10, "layer4": 3, "avgpool": 5}
        self.assertEqual(build_components_map(self.feature_layers, 5, "layer3:10,layer4:3"), expected)

    def test_shorthand_list(self) -> None:
        expected = {"layer3": 7, "layer4": 8, "avgpool": 9}
        self.assertEqual(build_components_map(self.feature_layers, 5, "7,8,9"), expected)

    def test_mismatch_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_components_map(self.feature_layers, 5, "7,8")

    def test_unknown_layer_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_components_map(self.feature_layers, 5, "layerX:10")


class TestParseSeeds(unittest.TestCase):
    def test_parse_seeds_basic(self) -> None:
        self.assertEqual(parse_seeds("0,1,2"), [0, 1, 2])

    def test_parse_seeds_whitespace(self) -> None:
        self.assertEqual(parse_seeds(" 0 , 2 , 4 "), [0, 2, 4])

    def test_parse_seeds_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_seeds("")


if __name__ == "__main__":
    unittest.main()
