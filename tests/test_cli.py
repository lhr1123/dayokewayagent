"""CLI integration tests using synthetic inputs and a model double."""

from contextlib import nullcontext, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from task1_agent.__main__ import main
from task1_agent.dataset import TASK_FILES
from task1_agent.examples import demo_question
from task1_agent.llm import ModelCallError
from tests.support import ScriptedModel, checked_answer


class CliTests(unittest.TestCase):
    def test_offline_demos_use_built_in_cases_without_opening_a_client(self):
        for case, expected in (("segment", "B"), ("region", "C")):
            with self.subTest(case=case), patch("task1_agent.runner.ChatClient.from_env") as client, redirect_stdout(StringIO()) as out:
                self.assertEqual(main(["demo", "--case", case, "--offline"]), 0)
                client.assert_not_called()
                trace = json.loads(out.getvalue())
                self.assertEqual(trace["result"]["结果"], expected)
                self.assertTrue(trace["passed"])

    def test_batch_reads_zip_and_saves_success_separately_from_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, trace = root / "input.zip", root / "result.jsonl", root / "trace.jsonl"
            with ZipFile(source, "w") as archive:
                for filename, kind in TASK_FILES.items():
                    case, _ = demo_question(kind)
                    archive.writestr(f"stage2/task1/{filename}", case.model_dump_json(by_alias=True) + "\n")
            model = ScriptedModel({"queries": [{"tool": "window", "start": 120}]},
                                  checked_answer("B", "altitude.change", 0, op="lt"),
                                  ModelCallError("timeout"))
            argv = ["run", "--input", str(source), "--output", str(output), "--trace", str(trace)]
            with patch("task1_agent.runner.ChatClient.from_env", return_value=nullcontext(model)), redirect_stdout(StringIO()):
                self.assertEqual(main(argv), 1)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["结果"], "B")
            failure = output.with_suffix(".errors.jsonl")
            self.assertEqual(json.loads(failure.read_text(encoding="utf-8"))["sample_id"], "demo_region")
            self.assertEqual(len(trace.read_text(encoding="utf-8").splitlines()), 2)
