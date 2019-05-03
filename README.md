# Mozilla Speech Recognition 

Test the latest mozilla deepspeech recongition and generate statistical comparison

## Installation

First, you need to go to https://github.com/mozilla/DeepSpeech to obtain the models for deepspeech recongition. It is impossible to upload them here since the files are too large.

## Usage

You can run the deepspeech by:

deepspeech --model models/output_graph.pbmm --alphabet models/alphabet.txt --lm models/lm.binary --trie models/trie --audio audio/youraudiofile.wav

In the directory, generate_statisitcs, is where I test the performance of Mozilla deepspeech recogntion.

