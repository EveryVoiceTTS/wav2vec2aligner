import re

import matplotlib.pyplot as plt
import torch
from g2p import make_g2p
from g2p.mappings import Mapping
from g2p.transducer import CompositeTransducer, Transducer
from pympi.Praat import TextGrid
from torchaudio.functional import forced_align
from torchaudio.models import wav2vec2_model

from .classes import Frame, Segment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DICTIONARY = {
        "<blank>": 0,
        "<pad>": 1,
        "</s>": 2,
        "@": 3,
        "a": 4,
        "i": 5,
        "e": 6,
        "n": 7,
        "o": 8,
        "u": 9,
        "t": 10,
        "s": 11,
        "r": 12,
        "m": 13,
        "k": 14,
        "l": 15,
        "d": 16,
        "g": 17,
        "h": 18,
        "y": 19,
        "b": 20,
        "p": 21,
        "w": 22,
        "c": 23,
        "v": 24,
        "j": 25,
        "z": 26,
        "f": 27,
        "'": 28,
        "q": 29,
        "x": 30,
    }

class TextHash(dict):
    def __init__(self, sentence_list, transducer):
        data = {}
        for i, sentence in enumerate(sentence_list):
            if sentence:
                data[f"s{i}"] = {"text": transducer(sentence)}
            else:
                continue
            words = sentence.split()
            for j, word in enumerate(words):
                data[f"s{i}w{j}"] = {"text": transducer(word)}
        super().__init__(data)


def create_transducer(text):
    text = text.lower()
    allowable_chars = DICTIONARY.keys()
    fallback_mapping = {}
    und_transducer = make_g2p("und", "und-ascii")
    text = und_transducer(text).output_string
    for char in text:
        if char not in allowable_chars and char not in fallback_mapping:
            fallback_mapping[char] = ""
    for k in fallback_mapping.keys():
        print(f"Found {k} which is not modelled by Wav2Vec2; skipping for alignment")
    punctuation_transducer = Transducer(
    Mapping(
        [{"in": k, "out": v} for k, v in fallback_mapping.items()],
        in_lang="und-ascii",
        out_lang="uroman",
        case_sensitive=False,
    )
    )
    und_transducer.__setattr__("norm_form", "NFC")
    return CompositeTransducer([und_transducer, punctuation_transducer])

def read_text(text_path):
    with open(text_path) as f:
        return [x.strip() for x in f]


def normalize_uroman(text):
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()


# utility function for playing audio segments.
def display_segment(i, waveform, word_segments, num_frames, sample_rate=16000):
    ratio = waveform.size(1) / num_frames
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    print(
        f"{word.label} ({word.score:.2f}): {x0 / sample_rate:.3f} - {x1 / sample_rate:.3f} sec"
    )
    segment = waveform[:, x0:x1]
    return segment.numpy(), sample_rate


# utility function for plotting word alignments
def plot_alignments(waveform, emission, segments, word_segments, sample_rate=16000):
    fig, ax = plt.subplots()
    ax.specgram(waveform[0], Fs=sample_rate)
    xlim = ax.get_xlim()

    ratio = waveform.size(1) / sample_rate / emission.size(1)
    for word in word_segments:
        t0, t1 = word.start * ratio, word.end * ratio
        ax.axvspan(t0, t1, facecolor="None", hatch="/", edgecolor="white")
        ax.annotate(
            f"{word.score:.2f}", (t0, sample_rate * 0.51), annotation_clip=False
        )

    for seg in segments:
        if seg.label != "|":
            ax.annotate(
                seg.label,
                (seg.start * ratio, sample_rate * 0.53),
                annotation_clip=False,
            )

    ax.set_xlabel("time [second]")
    ax.set_xlim(xlim)
    fig.tight_layout()

    return waveform, sample_rate


def plot_emission(emission):
    fig, ax = plt.subplots()
    ax.imshow(emission.T, aspect="auto")
    ax.set_title("Emission")
    fig.tight_layout()


def load_model():
    model = wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=[
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        extractor_conv_bias=True,
        encoder_embed_dim=1024,
        encoder_projection_dropout=0.0,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=0.0,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=0.1,
        encoder_dropout=0.0,
        encoder_layer_norm_first=True,
        encoder_layer_drop=0.1,
        aux_num_out=31,
    )
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt"
        )
    )
    model.eval()
    model.to(DEVICE)
    return model


def align_speech_file(audio, text_hash, model):
    # Construct the dictionary
    # '@' represents the OOV token
    # <pad> and </s> are fairseq's legacy tokens, which're not used.
    # <star> token is omitted as we do not use it in this tutorial
    

    emission = get_emission(model, audio.to(DEVICE))
    segments, words, sentences = compute_alignments(text_hash, DICTIONARY, emission)
    return segments, words, sentences, emission.size(1)


def get_emission(model, waveform):
    with torch.inference_mode():
        # NOTE: this step is essential
        waveform = torch.nn.functional.layer_norm(waveform, waveform.shape)
        emission, _ = model(waveform)
        return torch.log_softmax(emission, dim=-1)


def compute_alignments(transcript_hash, dictionary, emission):
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
    words_to_match = [v | {"key": k} for k, v in transcript_hash.items() if "w" in k]
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
                current_word["text"].input_string, start, end, sum(scores) / len(scores)
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
                start,
                end,
                sum(scores) / len(scores),
            )
    words = [v for k, v in transcript_hash.items() if "w" in k]
    sentences = [v for k, v in transcript_hash.items() if "w" not in k]
    return segments, words, sentences


def create_text_grid_from_segments(segments, seg_name, frame_ratio, sample_rate=16000):
    xmax = (frame_ratio * segments[-1].end) / sample_rate
    tg = TextGrid(xmax=xmax)
    value_tier = tg.add_tier(seg_name)
    score_tier = tg.add_tier(f"{seg_name}-score")
    for segment in segments:
        start = (frame_ratio * segment.start) / sample_rate
        end = (frame_ratio * segment.end) / sample_rate
        value_tier.add_interval(start, end, segment.label)
        score_tier.add_interval(start, end, "{:.2f}".format(segment.score))
    return tg
