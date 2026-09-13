# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Main entry point for static-analysis orchestration."""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from dataclasses import replace
from pathlib import Path

from docmethis_extract_python.static_extraction.ast_analysis import extract_module_record
from docmethis_extract_python.static_extraction.ast_inheritance import analyze_inheritance_project
from docmethis_extract_python.static_extraction.ast_metrics import compute_project_percentiles
from docmethis_extract_python.static_extraction.cache import (
    compute_file_hash,
    is_cache_valid,
    load_cache,
    restore_module_from_cache,
    save_cache,
)
from docmethis_extract_python.static_extraction.callgraph_ast import build_callgraph
from docmethis_extract_python.static_extraction.configuration import (
    ConfigurationDocmethis,
    load_configuration,
)
from docmethis_extract_python.static_extraction.dynamic_analysis.coverage import link_call_functions, map_coverage
from docmethis_extract_python.static_extraction.dynamic_analysis.snippets import generate_examples_from_calls
from docmethis_extract_python.static_extraction.dynamic_analysis.test_analysis import link_test_functions
from docmethis_extract_python.static_extraction.dynamic_analysis.test_runner import execute_tests
from docmethis_extract_python.static_extraction.models import CallgraphEdge, ModuleRecord, ProjectRecord
from docmethis_extract_python.static_extraction.project_metadata import extract_metadata
from docmethis_extract_python.static_extraction.quality_report import QualityReport
from docmethis_extract_python.static_extraction.traversal import (
    classify_imports,
    discover_files,
    is_stub_file,
    is_test_file,
    resolve_module_name,
)
from docmethis_extract_python.static_extraction.type_resolution import propagate_exceptions_via_callgraph
from docmethis_extract_python.time import utc_now_iso

logger = logging.getLogger(__name__)


def analyze_project(
    root: Path,
    configuration: ConfigurationDocmethis | None = None,
    *,
    skip_dynamic: bool = False,
    write_cache: bool = True,
    raw_records: list[ModuleRecord] | None = None,
) -> ProjectRecord:
    """Perform a complete analysis of a Python project.

    1. Discover Python files.
    2. Check the cache for each file.
    3. Parse uncached files.
    4. Save the cache.
    5. Build and return the ProjectRecord.

    When ``skip_dynamic`` is true, skip test execution, coverage, dynamic calls, and example generation. This saves time for tools
    that need only static analysis, such as the audit.

    When ``raw_records`` is provided, fill it with a **deep copy** of each module BEFORE project passes. Records in ``modules``
    remain owned by the returned ``ProjectRecord`` and are mutated in place by those passes. This rebuilds a symmetric base model
    (DIA, check module) without re-extracting unchanged files.

    Parameters
    ----------
    root : Path
        Root directory of the Python project to analyze. It is resolved to an absolute path before discovery and analysis begin.
    configuration : ConfigurationDocmethis | None = None
        Optional configuration object that controls the analysis behavior. When not provided, the configuration is loaded from the
        project root.
    skip_dynamic : bool = False
        When ``skip_dynamic`` is true, skip test execution, coverage, dynamic calls, and example generation. This saves time for
        tools that need only static analysis, such as the audit.
    write_cache : bool = True
        If True (default), save the updated cache to disk after parsing uncached files; if False, skip writing the cache.
    raw_records : list[ModuleRecord] | None = None
        If provided, this list is filled with a deep copy of each module record before project passes are applied. This allows
        callers to access the original extracted modules after the returned ProjectRecord's modules have been mutated in place by
        subsequent analysis passes, without re-extracting unchanged files.

    Returns
    -------
    ProjectRecord
        A ProjectRecord containing the completed project analysis, including the discovered modules, quality report, and analysis
        timestamp, after all static project passes have been applied.

    """
    root = root.resolve()

    if configuration is None:
        configuration = load_configuration(root)

    metadata = extract_metadata(root)
    files = discover_files(root, configuration)

    cache = load_cache(root, configuration) if not configuration.full_reanalysis else {}
    report = QualityReport()
    modules: list[ModuleRecord] = []

    for source_file in files:
        report.analyzed_files += 1

        if is_cache_valid(source_file, cache):
            mapping_key = source_file.as_posix()
            module = restore_module_from_cache(cache[mapping_key])
            report.files_from_cache += 1
        else:
            module_name = resolve_module_name(source_file, root)
            hash_file = compute_file_hash(source_file)
            module = extract_module_record(
                source_file,
                module_name,
                is_stub=is_stub_file(source_file),
                is_test=is_test_file(source_file, root),
                file_hash=hash_file,
                mixin_max_methods=configuration.mixin_max_methods,
            )
            if module is None:
                report.add_error(source_file, "parse_error", "Unable to parse the file.")
                continue

        report.found_functions += len(module.functions)
        report.found_classes += len(module.classes)
        for class_ in module.classes:
            report.found_functions += len(class_.methods)
        modules.append(module)

    if raw_records is not None:
        raw_records.extend(copy.deepcopy(module) for module in modules)

    if write_cache:
        save_cache(root, modules, configuration)

    timestamp = utc_now_iso()

    project_record = ProjectRecord(
        project_root=root,
        project_name=metadata.identifier_name,
        project_version=metadata.version,
        modules=modules,
        quality_report=report,
        analysis_timestamp=timestamp,
    )
    apply_project_passes(project_record)
    if not skip_dynamic:
        link_test_functions(modules)
        result_tests = execute_tests(root)
        map_coverage(result_tests, modules)
        link_call_functions(result_tests.raw_calls, modules)
        generate_examples_from_calls(modules, result_tests.failing_tests)
    return project_record


def apply_project_passes(project_record: ProjectRecord) -> dict[str, list[CallgraphEdge]]:
    """Apply the complete static enrichment pipeline and return its callgraph.

    Parameters
    ----------
    project_record : ProjectRecord
        The ProjectRecord instance to be processed by the static enrichment pipeline; the passes update it with derived data and
        use it to build the callgraph.

    Returns
    -------
    dict[str, list[CallgraphEdge]]
        A dictionary mapping callgraph node identifiers to lists of CallgraphEdge objects representing the outgoing edges from
        each node.

    """
    compute_project_percentiles(project_record)
    analyze_inheritance_project(project_record)
    callgraph = build_callgraph(project_record)
    propagate_exceptions_via_callgraph(project_record, callgraph)
    classify_imports(project_record)
    return callgraph


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Constructs and returns an argparse.ArgumentParser instance configured for the CLI, including a root path positional
        argument and options for full reanalysis and verbose logging.

    """
    parser = argparse.ArgumentParser(
        description="Static analysis of a Python project for docmethis.",
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Path to the project root to analyze.",
    )
    parser.add_argument(
        "--full-reanalysis",
        action="store_true",
        default=False,
        help="Force a full reanalysis and ignore the cache.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable detailed logging.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    root = args.root.resolve()

    if not root.is_dir():
        logger.error("Path %s is not a directory.", root)
        sys.exit(1)

    configuration = load_configuration(root)
    if args.full_reanalysis:
        # Use replace() instead of rebuilding fields: --full-reanalysis overrides
        # one setting while preserving every other [tool.docmethis] field,
        # including fields added after this line was written.
        configuration = replace(configuration, full_reanalysis=True)

    outcome = analyze_project(root, configuration)

    report = outcome.quality_report
    if report is not None:
        sys.stdout.write(report.generate_summary() + "\n")


if __name__ == "__main__":
    main()
