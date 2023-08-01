import os
from pathlib import Path

import click
import torchaudio

from .utils import (align_speech_file, create_text_grid_from_segments,
                    load_model, normalize_uroman)


@click.group()
@click.version_option(version="1.0", prog_name="aligner")
def cli():
    """Management script for aligner"""
    pass


@click.argument("audio_path")
@click.argument("text_path")
@click.option("--sample-rate", default=16000, help="The target sample rate.")
@cli.command()
def align_single(
    text_path: Path(exists=True, file_okay=True, dir_okay=False),
    audio_path: Path(exists=True, file_okay=True, dir_okay=False),
    sample_rate: int = 16000,
):
    print("loading model...")
    model = load_model()
    wav, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        print(f"resampling audio from {sr} to {sample_rate}")
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        fn, ext = os.path.splitext(audio_path)
        audio_path = fn + f"-{sample_rate}" + ext
        torchaudio.save(audio_path, wav, sample_rate)
    print("performing alignment")
    with open(text_path) as f:
        text = "| ".join([normalize_uroman(x) for x in f])
    _, ws, ss, num_frames = align_speech_file(wav, text, model)
    print("creating textgrid")
    waveform_to_frame_ratio = wav.size(1) / num_frames
    tg = create_text_grid_from_segments(
        ss, "sentence", waveform_to_frame_ratio, sample_rate=sample_rate
    )
    tg_path = Path(audio_path).with_suffix(".TextGrid")
    print(f"writing file to {tg_path}")
    tg.to_file(tg_path)


if __name__ == "__main__":
    align_single()
