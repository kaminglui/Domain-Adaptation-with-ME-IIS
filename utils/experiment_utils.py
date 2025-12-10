from pathlib import Path
from typing import Dict, List, Optional, Sequence


def normalize_dataset_name(name: str) -> str:
    """
    Normalize dataset name by lowercasing and stripping spaces, hyphens, underscores.
    E.g. "office_home", "Office-Home", "Office Home" -> "officehome".
    """
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def dataset_tag(name: str) -> str:
    """
    Return the short tag used in CSVs/results:
      - "office_home" (or variants) -> "office-home"
      - "office31" (or variants) -> "office-31"
      - otherwise return name as-is (lowercased).
    Use normalize_dataset_name internally.
    """
    norm = normalize_dataset_name(name)
    if norm == "officehome":
        return "office-home"
    if norm == "office31":
        return "office-31"
    return name.lower()


def parse_feature_layers(layers_str: str) -> List[str]:
    """
    Parse a comma-separated feature layer string into a list of layer names.
    - Strip whitespace.
    - Ignore empty segments.
    - If the input is empty or only whitespace, raise ValueError.
    Example:
      "layer3, layer4,avgpool" -> ["layer3", "layer4", "avgpool"].
    """
    layers = [layer.strip() for layer in layers_str.split(",") if layer.strip()]
    if not layers:
        raise ValueError("At least one feature layer must be specified (e.g., 'layer3,layer4,avgpool').")
    return layers


def build_components_map(
    feature_layers: Sequence[str], default_components: int, override_str: Optional[str]
) -> Dict[str, int]:
    """
    Build a mapping layer_name -> num_components.

    Behavior:
    - Start with all layers mapped to default_components.
    - If override_str is None or empty, just return the default map.

    If override_str is given, support TWO forms:

    (1) Explicit layer:count pairs, comma-separated:
        e.g. "layer3:10,layer4:5"
        - Split by comma.
        - Each item must contain exactly one ':'.
        - Layer names must be present in feature_layers.
        - Counts must be positive integers.

    (2) Shorthand comma-separated counts with the same length as feature_layers:
        e.g. for feature_layers=["layer3","layer4","avgpool"], override_str="5,10,5"
        - Split into counts.
        - If len(counts) != len(feature_layers), raise ValueError.
        - Map feature_layers[i] -> int(counts[i]).

    If an unknown layer name is given, or a count is invalid, raise ValueError.
    """
    comp_map: Dict[str, int] = {layer: int(default_components) for layer in feature_layers}
    if override_str is None or not str(override_str).strip():
        _validate_component_counts(comp_map)
        return comp_map

    override = str(override_str).strip()
    if ":" in override:
        items = [item.strip() for item in override.split(",") if item.strip()]
        for item in items:
            if ":" not in item:
                raise ValueError(f"Invalid components_per_layer entry '{item}'. Use 'layer:count'.")
            name, count_str = item.split(":", maxsplit=1)
            name = name.strip()
            if name not in comp_map:
                raise ValueError(f"Got components override for unknown layer '{name}'.")
            try:
                comp_map[name] = int(count_str)
            except ValueError as exc:
                raise ValueError(f"Invalid component count '{count_str}' for layer '{name}'.") from exc
    else:
        counts = [c.strip() for c in override.split(",") if c.strip()]
        if len(counts) != len(feature_layers):
            raise ValueError(
                f"components_per_layer override must match number of layers ({len(feature_layers)}), "
                f"got {len(counts)} entries."
            )
        for layer, count_str in zip(feature_layers, counts):
            try:
                comp_map[layer] = int(count_str)
            except ValueError as exc:
                raise ValueError(f"Invalid component count '{count_str}' for layer '{layer}'.") from exc

    _validate_component_counts(comp_map)
    return comp_map


def _validate_component_counts(comp_map: Dict[str, int]) -> None:
    for layer, count in comp_map.items():
        if int(count) <= 0:
            raise ValueError(f"Number of components for {layer} must be positive, got {count}.")


def build_source_ckpt_path(
    source_domain: str, target_domain: str, seed: int, base_dir: str = "checkpoints"
) -> Path:
    """
    Construct the canonical source-only checkpoint path:
      checkpoints/source_only_{source}_to_{target}_seed{seed}.pth
    """
    return Path(base_dir) / f"source_only_{source_domain}_to_{target_domain}_seed{seed}.pth"
