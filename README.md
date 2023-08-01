# Wav2Vec2 Aligner

An aligner based on Wav2Vec2. Most of the code is from [this tutorial](https://pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html) but turned into a package with a command line interface and a method for exporting to TextGrid.

## Install

`torch==2.1.0.dev20230731` and `torchaudio==2.1.0.dev20230731` currently.
Create a conda env with those nightly installs, then `pip install -e .`

## Usage

`CUDA_VISIBLE_DEVICES=0 align align-single sample.txt sample.wav` which will output a [Praat TextGrid](https://www.fon.hum.uva.nl/praat/) with the segment, word, and sentence level alignments.
