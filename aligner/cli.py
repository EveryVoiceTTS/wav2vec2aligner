import os
from pathlib import Path

import typer

from .utils import (
    TextHash,
    create_text_grid_from_segments,
    create_transducer,
    read_text,
)

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="An alignment tool based on CTC segmentation to split long audio into utterances",
)


def complete_path():
    return []


@app.command()
def align_single(
    text_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, autocompletion=complete_path
    ),
    audio_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, autocompletion=complete_path
    ),
    sample_rate: int = typer.Option(
        16000, help="The target sample rate for the model."
    ),
    word_padding: int = typer.Option(0, help="How many frames to pad around words."),
    sentence_padding: int = typer.Option(
        0, help="How many frames to pad around sentences (additive with word-padding)."
    ),
    debug: bool = typer.Option(False, help="Print debug statements"),
):
    print("loading model...")
    import torch
    import torchaudio

    from .heavy import load_model

    model, labels = load_model()
    wav, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        print(f"resampling audio from {sr} to {sample_rate}")
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        fn, ext = os.path.splitext(audio_path)
        audio_path = Path(fn + f"-{sample_rate}" + ext)
        torchaudio.save(audio_path, wav, sample_rate)
    if wav.size(0) != 1:
        print(f"converting audio from {wav.size(0)} channels to mono")
        wav = torch.mean(wav, dim=0).unsqueeze(0)
        fn, ext = os.path.splitext(audio_path)
        audio_path = Path(fn + f"-{sample_rate}-mono" + ext)
        torchaudio.save(audio_path, wav, sample_rate)
    print("processing text")
    sentence_list = read_text(text_path)
    transducer = create_transducer("".join(sentence_list), labels, debug)
    text_hash = TextHash(sentence_list, transducer)
    print("performing alignment")
    from .heavy import align_speech_file

    characters, words, sentences, num_frames = align_speech_file(
        wav, text_hash, model, labels, word_padding, sentence_padding
    )
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
    tg_path = audio_path.with_suffix(".TextGrid")
    print(f"writing file to {tg_path}")
    tg.to_file(tg_path)


if __name__ == "__main__":
    align_single()
