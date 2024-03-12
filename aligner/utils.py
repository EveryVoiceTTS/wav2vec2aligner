import re

from pympi.Praat import TextGrid


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
        if debug:
            print(
                f"Found {k} which is not modelled by Wav2Vec2; skipping for alignment"
            )
    punctuation_transducer = Transducer(
        Mapping(
            rules=[{"in": re.escape(k), "out": v} for k, v in fallback_mapping.items()],
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
