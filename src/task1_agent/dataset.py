"""Read Task 1 records from local ZIP and JSONL inputs."""

import io
import json
from pathlib import Path, PurePosixPath
from typing import TextIO
from zipfile import ZipFile

from .models import Question, QuestionType

TASK_FILES: dict[str, QuestionType] = {
    "01_segment_questions.jsonl": "segment",
    "02_region_questions.jsonl": "region",
}


def _read_records(handle: TextIO, name: str, kind: QuestionType) -> list[Question]:
    rows = []
    for index, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            rows.append(Question.model_validate({**row, "kind": kind}))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid Task 1 record in {name}, line {index}") from exc
    return rows


def input_paths(source: Path) -> list[Path]:
    """Resolve a ZIP, one question file, or a directory containing both files."""
    if source.is_file():
        if source.suffix.lower() == ".zip" or source.name in TASK_FILES:
            return [source]
        raise ValueError("expected a data ZIP or a Task 1 question file")
    if source.is_dir():
        folder = source / "task1" if (source / "task1").is_dir() else source
        paths = [folder / name for name in TASK_FILES]
        if all(path.is_file() for path in paths):
            return paths
    raise ValueError("input must contain the Task 1 question files")


def read_questions(paths: list[Path]) -> list[Question]:
    """Read all requested records and reject duplicate sample IDs."""
    questions = []
    for path in paths:
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                names = [PurePosixPath(candidate) for candidate in archive.namelist()]
                for file, kind in TASK_FILES.items():
                    members = [str(name) for name in names if name.name == file and name.parent.name == "task1"]
                    if len(members) != 1:
                        raise ValueError(f"ZIP must contain exactly one task1/{file}")
                    with io.TextIOWrapper(archive.open(members[0]), encoding="utf-8-sig") as handle:
                        questions.extend(_read_records(handle, members[0], kind))
        else:
            with path.open(encoding="utf-8-sig") as handle:
                questions.extend(_read_records(handle, path.name, TASK_FILES[path.name]))
    ids = {question.sample_id for question in questions}
    if not ids or len(ids) != len(questions):
        raise ValueError("Task 1 input must contain records with unique sample IDs")
    return questions
