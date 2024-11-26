import os
from pathlib import Path

import typer

from .utils import (
    TextHash,
    create_text_grid_from_segments,
    create_transducer,
    read_text,
)

CLI_LONG_HELP = """
    # Segment Help

        - **align** --- This command will align a long audio file with some text into words and sentences.

        - **extract** --- This command will take the alignment from the `align` command and extract it into multiple utterances in the format required for training a TTS system
    """

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    rich_markup_mode="markdown",
    help=CLI_LONG_HELP,
)


def complete_path(ctx, param, incomplete):
    return []


# We put this here for easy import into other modules that consume
# the aligner, like EveryVoice
EXTRACT_SEGMENTS_LONG_HELP = """
    # Segmentation help

    This command will take the alignment from the `align` command and extract it into multiple utterances in the format required for training a TTS system.
    """

EXTRACT_SEGMENTS_SHORT_HELP = "Extract the intervals from a TextGrid"


@app.command(
    name="extract",
    help=EXTRACT_SEGMENTS_LONG_HELP,
    short_help=EXTRACT_SEGMENTS_SHORT_HELP,
)
def extract_segments_from_textgrid(
    text_grid_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, shell_complete=complete_path
    ),
    audio_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, shell_complete=complete_path
    ),
    outdir: Path = typer.Argument(
        ..., exists=False, file_okay=False, dir_okay=True, shell_complete=complete_path
    ),
    tier_number: int = typer.Option(
        4, help="The index of the tier to extract intervals from."
    ),
    prefix: str = typer.Option(
        "segment", help="The basename prefix used to label files."
    ),
):
    import csv

    from pydub import AudioSegment
    from pympi.Praat import TextGrid
    from tqdm import tqdm

    audio = AudioSegment.from_file(audio_path)
    tg = TextGrid(text_grid_path)
    tier = tg.tiers[tier_number]
    intervals = [x for x in tier.get_all_intervals() if x[2]]
    segments = []
    n_fill = len(str(len(intervals)))
    for i, interval in enumerate(intervals):
        start = interval[0] * 1000
        end = interval[1] * 1000
        segments.append(
            {
                "audio": audio[start:end],
                "text": interval[2],
                "basename": f"{prefix}{str(i).zfill(n_fill)}",
            }
        )

    wavs_dir = outdir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    for seg in tqdm(segments, desc="Writing audio to files"):
        seg["audio"].export(wavs_dir / f'{seg["basename"]}.wav', format="wav")

    with open(outdir / "metadata.psv", "w", encoding="utf8") as f:
        writer = csv.DictWriter(f, delimiter="|", fieldnames=["basename", "text"])
        writer.writeheader()
        for seg in segments:
            writer.writerow({"basename": seg["basename"], "text": seg["text"]})

    print(
        f"Success! Your audio is available in {wavs_dir.absolute()} and your corresponding metadata file is available in {(outdir / 'metadata.psv').absolute()}"
    )


# We put this here for easy import into other modules that consume
# the aligner, like EveryVoice
ALIGN_SINGLE_LONG_HELP = """
    # Segmentation help

    This command will align a long audio file with some text. Your text should be separated so that each sentence/utterance is on a new line in the text file.
    This command should work on most languages and you should run it before running the new project or preprocessing steps.
    This command will create a Praat TextGrid file. You must install Praat (https://www.fon.hum.uva.nl/praat/) if you want to inspect the alignments.
    """
ALIGN_SINGLE_SHORT_HELP = "Align a long audio file with some text"


@app.command(
    name="align", help=ALIGN_SINGLE_LONG_HELP, short_help=ALIGN_SINGLE_SHORT_HELP
)
def align_single(
    text_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, shell_complete=complete_path
    ),
    audio_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, shell_complete=complete_path
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
    # Do fast error checking before loading expensive dependencies
    sentence_list = read_text(text_path)
    if not sentence_list or not any(sentence_list):
        raise typer.BadParameter(
            f"TEXT_PATH file '{text_path}' is empty; it should contain sentences to align.",
        )

    print("loading pytorch...")
    import torch
    import torchaudio

    from .heavy import load_model

    model, labels = load_model()
    wav, sr = torchaudio.load(str(audio_path))
    if sr != sample_rate:
        print(f"resampling audio from {sr} to {sample_rate}")
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        fn, ext = os.path.splitext(audio_path)
        audio_path = Path(fn + f"-{sample_rate}" + ext)
        torchaudio.save(str(audio_path), wav, sample_rate)
    if wav.size(0) != 1:
        print(f"converting audio from {wav.size(0)} channels to mono")
        wav = torch.mean(wav, dim=0).unsqueeze(0)
        fn, ext = os.path.splitext(audio_path)
        audio_path = Path(fn + f"-{sample_rate}-mono" + ext)
        torchaudio.save(str(audio_path), wav, sample_rate)
    print("processing text")
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
