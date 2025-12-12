"""
Generate a Markdown CLI reference from the argparse metadata used across scripts.

Run:
    python scripts/generate_cli_reference.py --out docs/cli_reference.md
"""

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cli.args import build_adapt_parser, build_experiments_parser, build_train_parser


def _type_name(action: argparse.Action) -> str:
    if action.nargs == 0:
        return "flag"
    if action.type is None:
        return "str"
    return getattr(action.type, "__name__", str(action.type))


def _format_default(value) -> str:
    if value is argparse.SUPPRESS:
        return ""
    if isinstance(value, str):
        return value
    return repr(value)


def _render_group(title: str, actions: List[argparse.Action]) -> List[str]:
    lines: List[str] = []
    if not actions:
        return lines
    lines.append(f"### {title}")
    for act in actions:
        if act.option_strings and "-h" in act.option_strings:
            continue
        names = act.option_strings if act.option_strings else [act.dest]
        flag_str = ", ".join(names)
        tname = _type_name(act)
        default = _format_default(act.default)
        choices = ""
        if getattr(act, "choices", None):
            choices = f"; choices: {list(act.choices)}"
        help_text = act.help or ""
        lines.append(f"- `{flag_str}` (type: {tname}; default: {default}{choices}) — {help_text}")
    lines.append("")  # blank line after group
    return lines


def _render_parser(name: str, parser: argparse.ArgumentParser) -> List[str]:
    lines: List[str] = [f"## {name}", "", f"Usage: `{parser.format_usage().strip()}`", ""]
    for group in parser._action_groups:  # type: ignore[attr-defined]
        group_lines = _render_group(group.title or "Arguments", group._group_actions)  # type: ignore[attr-defined]
        lines.extend(group_lines)
    return lines


def generate_reference(out_path: Path) -> None:
    sections: List[str] = [
        "# CLI Reference (auto-generated)",
        "",
        "Generated from argparse metadata. Regenerate with:",
        "`python scripts/generate_cli_reference.py --out docs/cli_reference.md`",
        "",
    ]
    train_parser = build_train_parser()
    train_parser.prog = "python scripts/train_source.py"
    adapt_parser = build_adapt_parser()
    adapt_parser.prog = "python scripts/adapt_me_iis.py"
    exp_parser = build_experiments_parser()
    exp_parser.prog = "python scripts/run_me_iis_experiments.py"

    sections.extend(_render_parser("Train (source-only)", train_parser))
    sections.extend(_render_parser("Adapt (ME-IIS)", adapt_parser))
    sections.extend(_render_parser("Experiment Runner", exp_parser))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"[CLI] Wrote reference to {out_path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate Markdown CLI reference from parsers.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("docs") / "cli_reference.md",
        help="Output markdown path.",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_reference(args.out)
