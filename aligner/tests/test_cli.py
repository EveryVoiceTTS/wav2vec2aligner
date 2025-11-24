"""
Run wav2vec2aligner unit tests.
How to run this test suite:
If you installed wav2vec2aligner:
    python -m unittest aligner.tests.test_cli
If you installed everyvoice:
    python -m unittest everyvoice.model.aligner.wav2vec2aligner.aligner.tests.test_cli
"""

import io
import os
import subprocess
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from unittest import SkipTest, TestCase
from urllib.error import URLError
from urllib.request import Request, urlopen

from typer.testing import CliRunner

from ..classes import Segment
from ..cli import app

VERBOSE_OVERRIDE = bool(os.environ.get("EVERYVOICE_VERBOSE_TESTS", False))


class CLITest(TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_main_help(self):
        for help in "-h", "--help":
            with self.subTest(help=help):
                result = self.runner.invoke(app, [help])
                self.assertEqual(result.exit_code, 0)
                self.assertIn("align", result.stdout)
                self.assertIn("extract", result.stdout)

    def test_sub_help(self):
        for cmd in "align", "extract":
            for help in "-h", "--help":
                with self.subTest(cmd=cmd, help=help):
                    result = self.runner.invoke(app, [cmd, help])
                    self.assertEqual(result.exit_code, 0)
                    self.assertIn("Usage:", result.stdout)
                    self.assertIn(cmd, result.stdout)

    def test_align_empty_file(self):
        with self.subTest("empty file"):
            result = self.runner.invoke(app, ["align", os.devnull, os.devnull])
            self.assertNotEqual(result.exit_code, 0)
            self.assertRegex(result.stdout, r"(?s)is.*empty")

        with self.subTest("file with only empty lines"):
            with tempfile.TemporaryDirectory() as tmpdir:
                textfile = os.path.join(tmpdir, "emptylines.txt")
                with open(textfile, "w", encoding="utf8") as f:
                    f.write("\n \n   \n")
                result = self.runner.invoke(app, ["align", textfile, os.devnull])
                self.assertNotEqual(result.exit_code, 0)
                self.assertRegex(result.stdout, r"(?s)is.*empty")

    def fetch_ras_test_file(self, filename, outputdir):
        repo, path = "https://github.com/ReadAlongs/Studio/", "/tests/data/"
        request = Request(repo + "raw/refs/heads/main" + path + filename)
        request.add_header("Referer", repo + "blob/main" + path + filename)
        response = urlopen(request, timeout=5)
        with open(os.path.join(outputdir, filename), "wb") as f:
            f.write(response.read())

    def test_align_something(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            try:
                with redirect_stderr(io.StringIO()):
                    self.fetch_ras_test_file("ej-fra.txt", tmpdir)
                    self.fetch_ras_test_file("ej-fra.m4a", tmpdir)
            except URLError as e:  # pragma: no cover
                raise SkipTest(
                    f"Can't fetch test data: {e}; skipping the test that depends on the Internet."
                )
            txt = tmppath / "ej-fra.txt"
            m4a = tmppath / "ej-fra.m4a"
            wav = tmppath / "ej-fra.wav"
            # Under most circumstances, align can take a .m4a input file, but not
            # in CI. It's not a hard requirement, so just convert to .wav.
            result = subprocess.run(["ffmpeg", "-i", m4a, wav], capture_output=True)
            if result.returncode != 0 or VERBOSE_OVERRIDE:
                print("ffmpeg output:", result.stdout, result.stderr)
                print("ffmpeg exit code:", result.returncode)
                print(tmpdir)
                os.system("ls -la " + tmpdir)
            textgrid = tmppath / "ej-fra-16000.TextGrid"
            wav_out = tmppath / "ej-fra-16000.wav"

            with self.subTest("ctc-segmenter align"):
                result = self.runner.invoke(app, ["align", str(txt), str(wav)])
                if result.exit_code != 0:
                    os.system("ls -la " + tmpdir)
                    print(result.stdout)
                self.assertEqual(result.exit_code, 0)
                self.assertTrue(textgrid.exists())
                self.assertTrue(wav_out.exists())

            with self.subTest("ctc-segmenter extract"):
                result = self.runner.invoke(
                    app, ["extract", str(textgrid), str(wav_out), str(tmppath / "out")]
                )
                if result.exit_code != 0:
                    print(result.stdout)
                self.assertEqual(result.exit_code, 0)
                self.assertTrue((tmppath / "out/metadata.psv").exists())
                with open(txt, encoding="utf8") as txt_f:
                    non_blank_line_count = sum(1 for line in txt_f if line.strip())
                for i in range(non_blank_line_count):
                    self.assertTrue((tmppath / f"out/wavs/segment{i}.wav"))


class MiscTests(TestCase):
    def test_segment(self):
        segment = Segment("text", 500, 700, 0.42)
        self.assertEqual(len(segment), 200)
        self.assertEqual(repr(segment), "text (0.42): [ 500,  700)")
