from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_PIPELINE_ENTRYPOINTS = [
    "tools/run_experiment.py",
    "src/experiments/runner.py",
    "notebooks/Run_All_Experiments.ipynb",
]


@dataclass(frozen=True)
class Graph:
    edges: Dict[Path, Set[Path]]


def _iter_python_files(root: Path) -> List[Path]:
    skip_dirs = {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "env",
        "legacy",
        "checkpoints",
        "results",
        "outputs",
    }
    out: List[Path] = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if any(part in skip_dirs for part in rel.parts):
            continue
        out.append(path)
    return sorted(out)


def _module_name_for_file(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _build_module_index(root: Path, py_files: Iterable[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in py_files:
        mod = _module_name_for_file(root, path)
        index[mod] = path
    return index


def _read_ipynb_imports(path: Path) -> str:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    cells = nb.get("cells", [])
    chunks: List[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            chunks.append("".join(src))
        elif isinstance(src, str):
            chunks.append(src)
    return "\n".join(chunks)


def _parse_imports(
    code: str,
    *,
    current_module: Optional[str],
    module_index: Dict[str, Path],
) -> Set[Path]:
    try:
        tree = ast.parse(code)
    except Exception:
        return set()

    current_pkg = None
    if current_module:
        if current_module.endswith(".__init__"):
            current_pkg = current_module[: -len(".__init__")]
        else:
            current_pkg = current_module.rsplit(".", 1)[0] if "." in current_module else ""

    imported: Set[Path] = set()

    def _add_module(mod: str) -> None:
        if mod in module_index:
            imported.add(module_index[mod])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _add_module(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level and current_pkg is not None:
                parts = current_pkg.split(".") if current_pkg else []
                up = max(0, int(node.level) - 1)
                if up > 0:
                    parts = parts[: -up] if up <= len(parts) else []
                prefix = ".".join(parts)
                base = f"{prefix}.{base}".strip(".")

            if base:
                _add_module(base)
            for alias in node.names:
                # from pkg import submodule  -> treat as pkg.submodule when it exists
                if base and alias.name != "*":
                    _add_module(f"{base}.{alias.name}")

    return imported


def build_import_graph(root: Path, py_files: List[Path]) -> Graph:
    module_index = _build_module_index(root, py_files)
    edges: Dict[Path, Set[Path]] = {p: set() for p in py_files}
    for path in py_files:
        code = path.read_text(encoding="utf-8", errors="ignore")
        current_mod = _module_name_for_file(root, path)
        deps = _parse_imports(code, current_module=current_mod, module_index=module_index)
        edges[path] = {d for d in deps if d in edges}
    return Graph(edges=edges)


def reachable_from(entry_files: List[Path], graph: Graph) -> Set[Path]:
    reachable: Set[Path] = set()
    stack: List[Path] = []
    for p in entry_files:
        if p in graph.edges:
            reachable.add(p)
            stack.append(p)
    while stack:
        cur = stack.pop()
        for nxt in graph.edges.get(cur, set()):
            if nxt not in reachable:
                reachable.add(nxt)
                stack.append(nxt)
    return reachable


def _resolve_entrypoints(root: Path, raw: List[str]) -> Tuple[List[Path], str]:
    files: List[Path] = []
    nb_code = ""
    for item in raw:
        path = (root / item).resolve()
        if not path.exists():
            continue
        if path.suffix == ".ipynb":
            nb_code += "\n" + _read_ipynb_imports(path)
            continue
        files.append(path)
    return files, nb_code


def write_report(
    *,
    root: Path,
    graph: Graph,
    pipeline_reachable: Set[Path],
    tests_reachable: Set[Path],
    out_path: Path,
) -> None:
    all_files = sorted(graph.edges.keys())
    pipeline_only = sorted([p for p in all_files if p in pipeline_reachable])
    tests_only = sorted([p for p in all_files if (p in tests_reachable and p not in pipeline_reachable)])
    unreachable = sorted([p for p in all_files if (p not in pipeline_reachable and p not in tests_reachable)])

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(root)).replace("\\", "/")
        except Exception:
            return str(p)

    lines: List[str] = []
    lines.append("# UNUSED_CODE_REPORT")
    lines.append("")
    lines.append("This report is generated by `tools/audit_unused.py`.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total python files scanned: {len(all_files)}")
    lines.append(f"- Reachable from pipeline entrypoints: {len(pipeline_only)}")
    lines.append(f"- Reachable only from tests: {len(tests_only)}")
    lines.append(f"- Unreachable (likely unused): {len(unreachable)}")
    lines.append("")
    lines.append("## Pipeline Entrypoints Reachability")
    lines.append("")
    lines.append("These files are not imported (directly or transitively) by the pipeline entrypoints:")
    lines.append("")
    for p in unreachable:
        lines.append(f"- `{rel(p)}`")
    lines.append("")
    lines.append("## Tests-Only Reachability")
    lines.append("")
    lines.append("These files are reachable from tests but not from the pipeline entrypoints:")
    lines.append("")
    for p in tests_only:
        lines.append(f"- `{rel(p)}`")
    lines.append("")
    lines.append("## Notes / Limitations")
    lines.append("")
    lines.append("- Import graph is static and best-effort (dynamic imports may be missed).")
    lines.append("- `from pkg import name` is treated as importing `pkg.name` only when that module exists in-repo.")
    lines.append("- Notebook parsing is best-effort (imports are extracted from code cells).")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static unused-code audit (import reachability).")
    p.add_argument(
        "--entrypoints",
        nargs="*",
        default=DEFAULT_PIPELINE_ENTRYPOINTS,
        help="Pipeline entrypoints to seed reachability (paths relative to repo root).",
    )
    p.add_argument(
        "--out",
        default="docs/UNUSED_CODE_REPORT.md",
        help="Output report path (relative to repo root).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    py_files = _iter_python_files(REPO_ROOT)
    graph = build_import_graph(REPO_ROOT, py_files)

    pipeline_entry_files, nb_code = _resolve_entrypoints(REPO_ROOT, list(args.entrypoints))
    module_index = _build_module_index(REPO_ROOT, py_files)
    nb_imports = _parse_imports(nb_code, current_module=None, module_index=module_index) if nb_code else set()

    pipeline_reachable = reachable_from(pipeline_entry_files + list(nb_imports), graph)

    test_files = [p for p in py_files if "tests" in p.relative_to(REPO_ROOT).parts]
    tests_reachable = reachable_from(test_files, graph)

    out_path = (REPO_ROOT / str(args.out)).resolve()
    write_report(
        root=REPO_ROOT,
        graph=graph,
        pipeline_reachable=pipeline_reachable,
        tests_reachable=tests_reachable,
        out_path=out_path,
    )
    print(f"[audit_unused] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
