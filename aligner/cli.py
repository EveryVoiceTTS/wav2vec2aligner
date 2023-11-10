import os
from pathlib import Path

import typer
import torchaudio

from .utils import (
    align_speech_file,
    create_text_grid_from_segments,
    create_transducer,
    load_model,
    read_text,
    TextHash
)


app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="An alignment tool based on CTC segmentation to split long audio into utterances",
)

@app.command()
def align_single(
    text_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    audio_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False),
    sample_rate: int = typer.Option(16000, help="The target sample rate for the model."),
    word_padding: int = typer.Option(0, help="How many frames to pad around words."),
    sentence_padding: int = typer.Option(0, help="How many frames to pad around sentences (additive with word-padding).")
):
    print("loading model...")
    model, labels = load_model()
    wav, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        print(f"resampling audio from {sr} to {sample_rate}")
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        fn, ext = os.path.splitext(audio_path)
        audio_path = fn + f"-{sample_rate}" + ext
        torchaudio.save(audio_path, wav, sample_rate)
    print("processing text")
    sentence_list = read_text(text_path)
    transducer = create_transducer(''.join(sentence_list), labels)
    text_hash = TextHash(sentence_list, transducer)
    print("performing alignment")
    characters, words, sentences, num_frames = align_speech_file(wav, text_hash, model, labels, word_padding, sentence_padding)
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
