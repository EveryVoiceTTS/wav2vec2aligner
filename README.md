# Language-level Zero-Shot Wav2Vec2 Aligner with input preservation

An aligner based on Wav2Vec2 and [ctc segmentation](https://github.com/lumaku/ctc-segmentation). Most of the code was created by following [this tutorial](https://pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html) but uses [g2p](https://github.com/roedoejet/g2p) for preserving the input, turned into a package with a command line interface with a method for exporting to TextGrid.

## Install

Create a conda env, then `pip install -e .`

## Usage

`ctc-segmenter align-single sample.txt sample.wav` which will output a [Praat TextGrid](https://www.fon.hum.uva.nl/praat/) with the word, and sentence level alignments.

You can then adjust the Praat TextGrid as necessary and run `ctc-segmenter extract-segments-from-textgrid sample.TextGrid sample.wav outdir` which will extract the segments and write them to the `outdir` directory along with a metadata file.