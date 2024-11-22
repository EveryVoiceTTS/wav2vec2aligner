"""
Utility functions that require expensive imports. Don't import from cli until needed.
"""

import re

import torch

# from torchaudio.models import wav2vec2_model
import torchaudio
from torchaudio.functional import forced_align

from .classes import Frame, Segment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
    model = bundle.get_model()
    labels = {l.lower(): i for i, l in enumerate(bundle.get_labels())}
    del labels["-"]  # Remove blank token
    del labels["|"]  # Remove sentence token
    model.to(DEVICE)
    return model, labels


def align_speech_file(
    audio, text_hash, model, labels_dictionary, word_padding, sentence_padding
):
    emission = get_emission(model, audio.to(DEVICE))
    segments, words, sentences = compute_alignments(
        text_hash,
        labels_dictionary,
        emission,
        word_padding=word_padding,
        sentence_padding=sentence_padding,
    )
    return segments, words, sentences, emission.size(1)


def get_emission(model, waveform):
    with torch.inference_mode():
        # NOTE: this step is essential
        waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)
        emission, _ = model(waveform)
        return torch.log_softmax(emission, dim=-1)


def compute_alignments(
    transcript_hash, dictionary, emission, word_padding=0, sentence_padding=0
):
    all_words = [
        v["text"].output_string for k, v in transcript_hash.items() if "w" in k
    ]
    transcript = "".join(all_words)
    tokens = [dictionary[c] for c in transcript]
    targets = torch.tensor([tokens], dtype=torch.int32, device=emission.device)
    input_lengths = torch.tensor([emission.shape[1]], device=emission.device)
    target_lengths = torch.tensor([targets.shape[1]], device=emission.device)

    alignment, scores = forced_align(
        emission, targets, input_lengths, target_lengths, 0
    )

    scores = scores.exp()  # convert back to probability
    alignment, scores = alignment[0].tolist(), scores[0].tolist()

    assert len(alignment) == len(scores) == emission.size(1)
    token_index = -1
    prev_hyp = 0
    frames = []
    for i, (ali, score) in enumerate(zip(alignment, scores)):
        if ali == 0:
            prev_hyp = 0
            continue

        if ali != prev_hyp:
            token_index += 1
        frames.append(Frame(token_index, i, score))
        prev_hyp = ali
    words_to_match = [{**v, "key": k} for k, v in transcript_hash.items() if "w" in k]
    i1, i2 = 0, 0
    segments = []
    while i1 < len(frames):
        while i2 < len(frames) and frames[i1].token_index == frames[i2].token_index:
            i2 += 1
        score = sum(frames[k].score for k in range(i1, i2)) / (i2 - i1)

        segments.append(
            Segment(
                transcript[frames[i1].token_index],
                frames[i1].time_index,
                frames[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    segments_to_match = segments.copy()
    while len(words_to_match) > 0:
        current_word = words_to_match.pop(0)
        current_segment_sequence = ""
        scores = []

        if segments_to_match:
            start = segments_to_match[0].start
        end = None

        while len(segments_to_match) > 0:
            current_segment = segments_to_match.pop(0)
            scores.append(current_segment.score)
            end = current_segment.end

            current_segment_sequence += current_segment.label
            if (
                current_segment_sequence
                == current_word[
                    "text"
                ].output_string  # break the loop if the word is equal to the output string
            ):
                break
        if end is not None:
            transcript_hash[current_word["key"]] = Segment(
                current_word["text"].input_string,
                start - word_padding,
                end + word_padding,
                sum(scores) / len(scores),
            )

    key_pattern = re.compile(
        r"""
            (s\d+)                   # sentence
            (w\d+)                   # word
             """,
        re.VERBOSE | re.IGNORECASE,
    )
    word_hash = {k: v for k, v in transcript_hash.items() if "w" in k}
    for sentence in [k for k in transcript_hash.keys() if "w" not in k]:
        scores = []
        start = None
        end = None
        for w_k, w_v in word_hash.items():
            if sentence == re.match(key_pattern, w_k).group(1):
                scores.append(w_v.score)
                if start is None:
                    start = w_v.start
                end = w_v.end
        if end is not None:
            transcript_hash[sentence] = Segment(
                transcript_hash[sentence]["text"].input_string,
                start - sentence_padding,
                end + sentence_padding,
                sum(scores) / len(scores),
            )
    words = [v for k, v in transcript_hash.items() if "w" in k]
    sentences = [v for k, v in transcript_hash.items() if "w" not in k]
    return segments, words, sentences
