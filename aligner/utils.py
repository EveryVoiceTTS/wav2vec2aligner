import re

from .classes import Segment


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
                transduced = transducer(word)
                # Words made only of unsupported characters (e.g. a lone "-")
                # have nothing to align
                if transduced.output_string:
                    data[f"s{i}w{j}"] = {"text": transduced}
        super().__init__(data)


def create_transducer(text, labels_dictionary, debug=False):
    # deferred expensive imports
    from g2p import make_g2p
    from g2p.mappings import Mapping
    from g2p.transducer import CompositeTransducer, Transducer

    text = text.lower()
    allowable_chars = labels_dictionary.keys()
    fallback_mapping = {}
    und_transducer = make_g2p("und", "und-ascii")
    text = und_transducer(text).output_string
    for char in text:
        if char not in allowable_chars and char not in fallback_mapping:
            fallback_mapping[char] = ""
    for k in fallback_mapping.keys():
        if debug:  # pragma: no cover
            print(
                f"Found {k} which is not modelled by Wav2Vec2; skipping for alignment"
            )
    # A single rule matching runs of unsupported characters: one rule per
    # character breaks g2p's index tracking on runs like "..." and leaves
    # characters behind that the model has no label for.
    rules = []
    if fallback_mapping:
        char_class = "".join(re.escape(k) for k in fallback_mapping)
        rules.append({"in": f"[{char_class}]+", "out": ""})
    punctuation_transducer = Transducer(
        Mapping(
            rules=rules,
            in_lang="und-ascii",
            out_lang="uroman",
            case_sensitive=False,
        )
    )
    und_transducer.__setattr__("norm_form", "NFC")
    return CompositeTransducer([und_transducer, punctuation_transducer])


def read_text(text_path):
    with open(text_path, encoding="utf8") as f:
        return [x.strip() for x in f]


def create_text_grid_from_segments(segments, seg_name, frame_ratio, sample_rate=16000):
    from pympi.Praat import TextGrid

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


def pad_sentences_with_silence(sentences, num_frames, max_pad_frames):
    """Extend each sentence into the unaligned audio around it.

    Each side gets up to max_pad_frames of padding. A gap between two sentences
    is shared equally so they never overlap, and the first and last sentences
    can use the whole gap to the start and end of the audio.
    """
    padded = []
    for i, sentence in enumerate(sentences):
        if i == 0:
            room_before = sentence.start
        else:
            room_before = (sentence.start - sentences[i - 1].end) // 2
        if i == len(sentences) - 1:
            room_after = num_frames - sentence.end
        else:
            room_after = (sentences[i + 1].start - sentence.end) // 2
        pad_before = min(max_pad_frames, max(0, room_before))
        pad_after = min(max_pad_frames, max(0, room_after))
        padded.append(
            Segment(
                sentence.label,
                sentence.start - pad_before,
                sentence.end + pad_after,
                sentence.score,
            )
        )
    return padded
