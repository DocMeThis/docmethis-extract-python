# Copyright (c) 2026 DocMeThis SAS. All rights reserved.

"""Quality report scaffold for iteration 9.

Records parsing errors and analysis counters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class AnalysisError:
    """Error raised while analyzing a file.

    Attributes
    ----------
    source_file : str
        Source file the error concerns.
    type_error : str
        Type of the analysis error.
    message : str
        Human-readable error message.

    """

    source_file: str
    type_error: str
    message: str


@dataclass
class QualityReport:
    """Analysis quality report (completed in iteration 9).

    Attributes
    ----------
    analyzed_files : int
        Number of files analyzed.
    files_with_errors : int
        Number of files with errors.
    files_from_cache : int
        Number of files loaded from cache.
    found_functions : int
        Number of functions found.
    found_classes : int
        Number of classes found.
    errors : list[AnalysisError]
        List of analysis errors.

    """

    analyzed_files: int = 0
    files_with_errors: int = 0
    files_from_cache: int = 0
    found_functions: int = 0
    found_classes: int = 0
    errors: list[AnalysisError] = field(default_factory=list)

    def add_error(self, source_file: Path, type_error: str, message: str) -> None:
        """Record an analysis error.

        Parameters
        ----------
        source_file : Path
            The path to the source file associated with the analysis error.
        type_error : str
            The type or category of the analysis error.
        message : str
            A string containing the error message to record.

        """
        self.errors.append(
            AnalysisError(
                source_file=str(source_file),
                type_error=type_error,
                message=message,
            )
        )
        self.files_with_errors += 1

    def generate_summary(self) -> str:
        """Return a textual summary of the report.

        Returns
        -------
        str
            A string containing a summary of the report, including counts of analyzed files, functions, classes, and any errors.

        """
        lines = [
            f"Analyzed files: {self.analyzed_files}",
            f"  from cache: {self.files_from_cache}",
            f"  with errors: {self.files_with_errors}",
            f"Found functions: {self.found_functions}",
            f"Found classes: {self.found_classes}",
        ]
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  [{err.type_error}] {err.source_file}: {err.message}" for err in self.errors)
        return "\n".join(lines)
