# TLDST: Tamper Localisation and Detection with Source Tracking TLDST

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

# :rocket: Quick Links:

[[`arXiv`](https://arxiv.org/abs/2401.17264)]
[[🤗`Hugging Face`](https://huggingface.co/facebook/audioseal)]
[[`Colab Notebook`](https://colab.research.google.com/github/facebookresearch/audioseal/blob/master/examples/colab.ipynb)]
[[`Webpage`](https://pierrefdz.github.io/publications/audioseal/)]

# Abstract

- **Key Features:**

# Installation

### Requirements:

- Python >= 3.8
- Pytorch >= 1.13.0
- [Omegaconf](https://omegaconf.readthedocs.io/)
- [Julius](https://pypi.org/project/julius/)
- [Numpy](https://pypi.org/project/numpy/)

### Install from PyPI:

To install from source: Clone this repo and install in editable mode:

# Models

You can find all the model checkpoints on the [Hugging Face Hub](https://huggingface.co/). We provide the checkpoints for the following models:

- [TDLST Generator] :
  Takes an audio signal (as a waveform) and outputs a watermark of the same size as the input, which can be added to the input to watermark it. Optionally, it can also take a secret 16-bit message to embed in the watermark.
- [TDLST Detector]
  Takes an audio signal (as a waveform) and outputs the probability that the input contains a watermark at each sample (every 1/16k second). Optionally, it may also output the secret message encoded in the watermark.

Note that the message is a method of identifiaction of the source by source tracking by using IP address (up to $2**16=65536$ possible choices).

# Usage

Here’s a quick example of how you can use AudioSeal’s API to embed and detect watermarks:

Updates soon....
