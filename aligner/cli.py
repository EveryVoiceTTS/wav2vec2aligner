import os
from pathlib import Path

import click
import torchaudio

from .utils import (
    align_speech_file,
    create_text_grid_from_segments,
    create_transducer,
    load_model,
    read_text,
    TextHash
)


@click.group()
@click.version_option(version="1.0", prog_name="aligner")
def cli():
    """Management script for aligner"""
    pass


@click.argument("audio_path")
@click.argument("text_path")
@click.option("--sample-rate", default=16000, help="The target sample rate for the model.")
@click.option("--word-padding", default=0, help="How many frames to pad around words.")
@click.option("--sentence-padding", default=0, help="How many frames to pad around sentences (additive with word-padding).")
@cli.command()
def align_single(
    text_path: Path(exists=True, file_okay=True, dir_okay=False),
    audio_path: Path(exists=True, file_okay=True, dir_okay=False),
    sample_rate: int = 16000,
    word_padding: int = 0,
    sentence_padding: int = 0
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
    print("processing text")
    sentence_list = read_text(text_path)
    transducer = create_transducer(''.join(sentence_list))
    text_hash = TextHash(sentence_list, transducer)
    print("performing alignment")
    characters, words, sentences, num_frames = align_speech_file(wav, text_hash, model, word_padding, sentence_padding)
    print("creating textgrid")
    waveform_to_frame_ratio = wav.size(1) / num_frames
    tg = create_text_grid_from_segments(
        characters, "characters", waveform_to_frame_ratio, sample_rate=sample_rate
    )
    words_tg = create_text_grid_from_segments(
        words, "words", waveform_to_frame_ratio, sample_rate=sample_rate
    )
    sentences_tg = create_text_grid_from_segments(
        sentences, "sentences", waveform_to_frame_ratio, sample_rate=sample_rate
    )
    tg.tiers += words_tg.get_tiers()
    tg.tiers += sentences_tg.get_tiers()
    tg_path = Path(audio_path).with_suffix(".TextGrid")
    print(f"writing file to {tg_path}")
    tg.to_file(tg_path)


if __name__ == "__main__":
    align_single()
