import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from duplexpo_portable.cli import main
from duplexpo_portable.io import read_jsonl


class CliTest(unittest.TestCase):
    def test_synthetic_end_to_end(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            windows = Path(directory) / "windows.jsonl"
            audit = Path(directory) / "audit.jsonl"
            records = Path(directory) / "records.jsonl"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(
                    [
                        "construct",
                        "--profile",
                        "fisher",
                        "--input",
                        str(root / "examples" / "conversations.jsonl"),
                        "--output",
                        str(windows),
                        "--audit-output",
                        str(audit),
                    ]
                )
                main(
                    [
                        "prepare",
                        "--windows",
                        str(windows),
                        "--rollouts",
                        str(root / "examples" / "rollouts.jsonl"),
                        "--output",
                        str(records),
                    ]
                )
                main(
                    [
                        "conformance",
                        "--windows",
                        str(windows),
                        "--rollouts",
                        str(root / "examples" / "rollouts.jsonl"),
                        "--records",
                        str(records),
                    ]
                )
            self.assertEqual(len(list(read_jsonl(windows))), 2)
            values = list(read_jsonl(records))
            self.assertEqual(len(values), 8)
            self.assertTrue(all(value["schema_version"] == "duplexpo.grpo.v1" for value in values))
            self.assertEqual(windows.read_text(), (root / "examples" / "expected" / "windows.jsonl").read_text())
            self.assertEqual(audit.read_text(), (root / "examples" / "expected" / "audit.jsonl").read_text())
            self.assertEqual(records.read_text(), (root / "examples" / "expected" / "grpo_records.jsonl").read_text())
            summaries = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(summaries[-2]["group_count"], 2)
            self.assertTrue(summaries[-1]["ok"])


if __name__ == "__main__":
    unittest.main()
