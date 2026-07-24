"""Fail-closed local runtime dependency authority for pipeline stages."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

CLASSIFICATION_PACKAGE = "pig_behavior.classification_v2"
CLASSIFICATION_SOURCE_ROOT = PurePosixPath(
    "src/pig_behavior/classification_v2"
)
RUNTIME_DEPENDENCY_CLOSURE_VERSION = (
    "classification_v2.runtime_dependency_closure.v1"
)


def _relative_path(repo_root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"production path escapes repository: {path}") from exc
    return relative.as_posix()


def _module_name(repo_root: Path, source_path: Path) -> str | None:
    package_root = repo_root / CLASSIFICATION_SOURCE_ROOT
    try:
        relative = source_path.resolve().relative_to(package_root.resolve())
    except ValueError:
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    suffix = ".".join(parts)
    return CLASSIFICATION_PACKAGE + (f".{suffix}" if suffix else "")


def _module_file(repo_root: Path, module_name: str) -> Path | None:
    if module_name == CLASSIFICATION_PACKAGE:
        candidate = repo_root / CLASSIFICATION_SOURCE_ROOT / "__init__.py"
        return candidate if candidate.is_file() else None
    prefix = CLASSIFICATION_PACKAGE + "."
    if not module_name.startswith(prefix):
        return None
    suffix = module_name[len(prefix) :]
    base = repo_root / CLASSIFICATION_SOURCE_ROOT / PurePosixPath(
        suffix.replace(".", "/")
    )
    file_candidate = base.with_suffix(".py")
    if file_candidate.is_file():
        return file_candidate
    package_candidate = base / "__init__.py"
    if package_candidate.is_file():
        return package_candidate
    return None


def _package_initializers(repo_root: Path, source_path: Path) -> list[Path]:
    package_root = (repo_root / CLASSIFICATION_SOURCE_ROOT).resolve()
    try:
        relative = source_path.resolve().relative_to(package_root)
    except ValueError:
        return []
    directory = relative.parent
    initializers: list[Path] = []
    current = package_root
    root_initializer = current / "__init__.py"
    if root_initializer.is_file():
        initializers.append(root_initializer)
    for part in directory.parts:
        current = current / part
        initializer = current / "__init__.py"
        if initializer.is_file():
            initializers.append(initializer)
    return initializers


def _is_type_checking_test(
    node: ast.expr,
    type_checking_names: set[str],
    typing_module_names: set[str],
) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id in type_checking_names
        or isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in typing_module_names
        and node.attr == "TYPE_CHECKING"
    )


def _static_truth(node: ast.expr) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (bool, int)):
        return bool(node.value)
    return None


class _RuntimeImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: list[ast.Import | ast.ImportFrom] = []
        self.dynamic_imports: list[ast.Call] = []
        self.dynamic_function_names = {"__import__", "import_module"}
        self.importlib_module_names = {"importlib"}
        self.type_checking_names = {"TYPE_CHECKING"}
        self.typing_module_names = {"typing"}

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_test(
            node.test,
            self.type_checking_names,
            self.typing_module_names,
        ):
            for child in node.orelse:
                self.visit(child)
            return
        truth = _static_truth(node.test)
        if truth is False:
            for child in node.orelse:
                self.visit(child)
            return
        if truth is True:
            for child in node.body:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.append(node)
        for alias in node.names:
            if alias.name == "importlib":
                self.importlib_module_names.add(alias.asname or alias.name)
            if alias.name == "typing":
                self.typing_module_names.add(alias.asname or alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.append(node)
        if node.level == 0 and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    self.dynamic_function_names.add(
                        alias.asname or alias.name
                    )
        if node.level == 0 and node.module == "typing":
            for alias in node.names:
                if alias.name == "TYPE_CHECKING":
                    self.type_checking_names.add(
                        alias.asname or alias.name
                    )

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        is_dynamic = (
            isinstance(function, ast.Name)
            and function.id in self.dynamic_function_names
            or isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id in self.importlib_module_names
            and function.attr == "import_module"
        )
        if is_dynamic:
            self.dynamic_imports.append(node)
        self.generic_visit(node)


def _current_package(repo_root: Path, source_path: Path) -> str | None:
    module = _module_name(repo_root, source_path)
    if module is None:
        return None
    if source_path.name == "__init__.py":
        return module
    return module.rpartition(".")[0]


def _resolve_from_module(
    repo_root: Path,
    source_path: Path,
    node: ast.ImportFrom,
) -> str | None:
    if node.level == 0:
        return node.module
    package = _current_package(repo_root, source_path)
    if package is None:
        return None
    parts = package.split(".")
    remove_count = node.level - 1
    if remove_count > len(parts):
        return None
    if remove_count:
        parts = parts[:-remove_count]
    if node.module:
        parts.extend(node.module.split("."))
    return ".".join(parts)


def _resolve_import_modules(
    repo_root: Path,
    source_path: Path,
    node: ast.Import | ast.ImportFrom,
) -> tuple[list[tuple[str, bool]], list[str]]:
    modules: list[tuple[str, bool]] = []
    unresolved: list[str] = []
    if isinstance(node, ast.Import):
        modules.extend((alias.name, True) for alias in node.names)
        return modules, unresolved
    base = _resolve_from_module(repo_root, source_path, node)
    if base is None:
        if node.level:
            unresolved.append(
                f"{_relative_path(repo_root, source_path)}:{node.lineno}:"
                "UNRESOLVED_RELATIVE_IMPORT"
            )
        return modules, unresolved
    modules.append((base, True))
    for alias in node.names:
        if alias.name != "*":
            modules.append((f"{base}.{alias.name}", False))
    return modules, unresolved


def _dynamic_import_module(node: ast.Call) -> str | None:
    if not node.args:
        return None
    argument = node.args[0]
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return argument.value
    return None


def resolve_runtime_dependency_closure(
    repo_root: Path,
    entry_points: Sequence[str],
) -> dict[str, Any]:
    """Resolve recursive local runtime imports for production entry points."""

    queue: list[Path] = []
    missing_files: list[str] = []
    for relative in entry_points:
        path = repo_root / PurePosixPath(relative)
        if not path.is_file():
            missing_files.append(relative)
        else:
            queue.append(path)
    visited: set[str] = set()
    unresolved_dynamic: set[str] = set()
    unresolved_local: set[str] = set()
    while queue:
        source_path = queue.pop()
        relative = _relative_path(repo_root, source_path)
        if relative in visited:
            continue
        visited.add(relative)
        try:
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                filename=relative,
            )
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"cannot parse production file {relative}") from exc
        visitor = _RuntimeImportVisitor()
        visitor.visit(tree)
        candidate_paths: set[Path] = set(
            _package_initializers(repo_root, source_path)
        )
        for import_node in visitor.imports:
            modules, unresolved = _resolve_import_modules(
                repo_root,
                source_path,
                import_node,
            )
            unresolved_local.update(unresolved)
            for module, required in modules:
                if (
                    module != CLASSIFICATION_PACKAGE
                    and not module.startswith(CLASSIFICATION_PACKAGE + ".")
                ):
                    continue
                dependency = _module_file(repo_root, module)
                if dependency is None:
                    if required and module.startswith(
                        CLASSIFICATION_PACKAGE + "."
                    ):
                        unresolved_local.add(
                            f"{relative}:{import_node.lineno}:"
                            f"UNRESOLVED_LOCAL_IMPORT:{module}"
                        )
                    continue
                candidate_paths.add(dependency)
                candidate_paths.update(
                    _package_initializers(repo_root, dependency)
                )
        for dynamic_node in visitor.dynamic_imports:
            module = _dynamic_import_module(dynamic_node)
            if module is None:
                unresolved_dynamic.add(
                    f"{relative}:{dynamic_node.lineno}:"
                    "UNRESOLVED_DYNAMIC_IMPORT"
                )
                continue
            if module.startswith("."):
                unresolved_dynamic.add(
                    f"{relative}:{dynamic_node.lineno}:"
                    f"UNRESOLVED_DYNAMIC_IMPORT:{module}"
                )
                continue
            if (
                module != CLASSIFICATION_PACKAGE
                and not module.startswith(CLASSIFICATION_PACKAGE + ".")
            ):
                continue
            dependency = _module_file(repo_root, module)
            if dependency is None:
                unresolved_dynamic.add(
                    f"{relative}:{dynamic_node.lineno}:"
                    f"UNRESOLVED_DYNAMIC_IMPORT:{module}"
                )
                continue
            candidate_paths.add(dependency)
            candidate_paths.update(
                _package_initializers(repo_root, dependency)
            )
        for candidate in candidate_paths:
            candidate_relative = _relative_path(repo_root, candidate)
            if candidate_relative not in visited:
                queue.append(candidate)
    return {
        "runtime_dependency_closure_version": (
            RUNTIME_DEPENDENCY_CLOSURE_VERSION
        ),
        "entry_points": sorted(set(entry_points)),
        "runtime_dependency_closure": sorted(visited),
        "missing_production_files": sorted(set(missing_files)),
        "unresolved_local_imports": sorted(unresolved_local),
        "unresolved_dynamic_imports": sorted(unresolved_dynamic),
    }


def stage_runtime_dependency_audit(
    repo_root: Path,
    stage_id: str,
    mapping_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Compare one stage's runtime closure with its hashed code authority."""

    stage_rows = [
        row
        for row in mapping_rows
        if row.get("contract_item_type") == "stage"
        and row.get("contract_item_id") == stage_id
    ]
    hashed_files = sorted(
        {
            str(row.get("source_file", "")).replace("\\", "/")
            for row in stage_rows
            if str(row.get("source_file", "")).strip()
        }
    )
    entry_points = sorted(
        {
            str(row.get("source_file", "")).replace("\\", "/")
            for row in stage_rows
            if str(row.get("source_file", "")).strip()
            and str(row.get("symbol", "")).strip()
        }
    )
    if not hashed_files:
        return {
            "stage_id": stage_id,
            "entry_points": [],
            "runtime_dependency_closure": [],
            "hashed_code_files": [],
            "missing_dependencies": [],
            "missing_production_files": [],
            "unresolved_local_imports": [],
            "unresolved_dynamic_imports": [],
            "status": "FAIL_NO_CODE_AUTHORITY",
        }
    if not entry_points:
        return {
            "stage_id": stage_id,
            "entry_points": [],
            "runtime_dependency_closure": [],
            "hashed_code_files": hashed_files,
            "missing_dependencies": [],
            "missing_production_files": [],
            "unresolved_local_imports": [],
            "unresolved_dynamic_imports": [],
            "status": "FAIL_NO_ENTRY_POINT",
        }
    closure = resolve_runtime_dependency_closure(repo_root, entry_points)
    mapped_missing_files = [
        relative
        for relative in hashed_files
        if not (repo_root / PurePosixPath(relative)).is_file()
    ]
    missing_production_files = sorted(
        set(closure["missing_production_files"]) | set(mapped_missing_files)
    )
    missing_dependencies = sorted(
        set(closure["runtime_dependency_closure"]) - set(hashed_files)
    )
    status = "PASS"
    if (
        missing_dependencies
        or missing_production_files
        or closure["unresolved_local_imports"]
        or closure["unresolved_dynamic_imports"]
    ):
        status = "FAIL"
    return {
        "stage_id": stage_id,
        "entry_points": entry_points,
        "runtime_dependency_closure": closure[
            "runtime_dependency_closure"
        ],
        "hashed_code_files": hashed_files,
        "missing_dependencies": missing_dependencies,
        "missing_production_files": missing_production_files,
        "unresolved_local_imports": closure["unresolved_local_imports"],
        "unresolved_dynamic_imports": closure[
            "unresolved_dynamic_imports"
        ],
        "status": status,
    }


def assert_stage_runtime_dependencies_complete(
    repo_root: Path,
    stage_id: str,
    mapping_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Return a passing audit or refuse incomplete stage code authority."""

    audit = stage_runtime_dependency_audit(
        repo_root,
        stage_id,
        mapping_rows,
    )
    if audit["status"] != "PASS":
        raise ValueError(
            "stage runtime dependency authority incomplete: "
            f"stage_id={stage_id},"
            f"missing_dependencies={audit['missing_dependencies']},"
            f"missing_production_files={audit['missing_production_files']},"
            f"unresolved_local_imports={audit['unresolved_local_imports']},"
            f"unresolved_dynamic_imports={audit['unresolved_dynamic_imports']},"
            f"status={audit['status']}"
        )
    return audit


def audit_all_stage_runtime_dependencies(
    repo_root: Path,
    stage_ids: Sequence[str],
    mapping_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Audit every exact contract stage and fail the aggregate closed."""

    stages = [
        stage_runtime_dependency_audit(
            repo_root,
            stage_id,
            mapping_rows,
        )
        for stage_id in stage_ids
    ]
    missing_count = sum(
        len(stage["missing_dependencies"]) for stage in stages
    )
    unresolved_dynamic_count = sum(
        len(stage["unresolved_dynamic_imports"]) for stage in stages
    )
    passing = all(stage["status"] == "PASS" for stage in stages)
    return {
        "runtime_dependency_closure_version": (
            RUNTIME_DEPENDENCY_CLOSURE_VERSION
        ),
        "stage_count": len(stages),
        "unmapped_production_dependencies": missing_count,
        "unresolved_dynamic_import_count": unresolved_dynamic_count,
        "stages": stages,
        "status": "PASS" if passing else "FAIL",
    }
