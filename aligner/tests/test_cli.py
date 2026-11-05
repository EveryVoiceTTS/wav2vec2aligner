"""
Run wav2vec2aligner unit tests.
How to run this test suite: pytest path/to/test_cli.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pytest
from typer.testing import CliRunner

from ..classes import Segment
from ..cli import app


@pytest.fixture(scope="class")
def runner(request) -> None:
    request.cls.runner = CliRunner()


@pytest.mark.usefixtures("runner")
class TestCLI:
    def test_main_help(self, subtests):
        for help in "-h", "--help":
            with subtests.test(help=help):
                result = self.runner.invoke(app, [help])
                assert result.exit_code == 0
                assert "align" in result.stdout
                assert "extract" in result.stdout

    def test_sub_help(self, subtests):
        for cmd in "align", "extract":
            for help in "-h", "--help":
                with subtests.test(cmd=cmd, help=help):
                    result = self.runner.invoke(app, [cmd, help])
                    assert result.exit_code == 0
                    assert "Usage:" in result.stdout
                    assert cmd in result.stdout

    def test_align_empty_file(self, subtests, tmp_path):
        with subtests.test("empty file"):
            result = self.runner.invoke(app, ["align", os.devnull, os.devnull])
            assert result.exit_code != 0
            assert re.search(r"(?s)is.*empty", result.output)

        with subtests.test("file with only empty lines"):
            textfile = tmp_path / "emptylines.txt"
            with open(textfile, "w", encoding="utf8") as f:
                f.write("\n \n   \n")
            result = self.runner.invoke(app, ["align", str(textfile), os.devnull])
            assert result.exit_code != 0
            assert re.search(r"(?s)is.*empty", result.output)

    def fetch_ras_test_file(self, filename, output_dir: Path):
        output_file = output_dir / filename
        if output_file.exists():
            print(f"Reusing {output_file} previously fetched", file=sys.stderr)
            return
        repo, path = "https://github.com/ReadAlongs/Studio/", "/tests/data/"
        request = Request(repo + "raw/refs/heads/main" + path + filename)
        print(request.full_url, file=sys.stderr)
        request.add_header("Referer", repo + "blob/main" + path + filename)
        response = urlopen(request, timeout=5)
        with open(output_file, "wb") as f:
            f.write(response.read())

    def test_align_something(self, subtests, tmp_path):
        data_path = Path(__file__).parent / "data"
        tmpdir = str(tmp_path)
        try:
            self.fetch_ras_test_file("ej-fra.txt", data_path)
            self.fetch_ras_test_file("ej-fra.m4a", data_path)
        except Exception as e:  # pragma: no cover
            print(e)
            raise pytest.skip(
                f"Can't fetch test data: {e}; skipping the test that depends on the Internet."
            )
        txt = data_path / "ej-fra.txt"
        m4a = data_path / "ej-fra.m4a"
        wav = tmp_path / "ej-fra.wav"
        # Under most circumstances, align can take a .m4a input file, but not
        # in CI. It's not a hard requirement, so just convert to .wav.
        result = subprocess.run(["ffmpeg", "-i", m4a, wav], capture_output=True)
        print("ffmpeg output:", result.stdout, result.stderr)
        print("ffmpeg exit code:", result.returncode)
        print(tmp_path)
        os.system("ls -la " + tmpdir)
        textgrid = tmp_path / "ej-fra-16000.TextGrid"
        wav_out = tmp_path / "ej-fra-16000.wav"

        with subtests.test("ctc-segmenter align"):
            result = self.runner.invoke(app, ["align", str(txt), str(wav)])
            if result.exit_code != 0:
                os.system("ls -la " + tmpdir)
                print(result.output)
            assert result.exit_code == 0
            assert textgrid.exists()
            assert wav_out.exists()

        with subtests.test("ctc-segmenter extract"):
            result = self.runner.invoke(
                app, ["extract", str(textgrid), str(wav_out), str(tmp_path / "out")]
            )
            if result.exit_code != 0:
                print(result.output)
            assert result.exit_code == 0
            assert (tmp_path / "out/metadata.psv").exists()
            with open(txt, encoding="utf8") as txt_f:
                non_blank_line_count = sum(1 for line in txt_f if line.strip())
            for i in range(non_blank_line_count):
                assert (tmp_path / f"out/wavs/segment{i}.wav").exists()


class TestMisc:
    def test_segment(self):
        segment = Segment("text", 500, 700, 0.42)
        assert len(segment) == 200
        assert repr(segment) == "text (0.42): [ 500,  700)"
