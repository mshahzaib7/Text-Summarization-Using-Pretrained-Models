# Text Summarization Using Pretrained Models

A practical repository demonstrating how to build extractive and abstractive text summarization systems using pretrained language models.

## Overview

This project shows example scripts for summarizing long documents into concise summaries using transformer-based models. It is intended for prototyping, learning, and extending with production-ready components.

## Contents

1. `Text Summarization Using Pretrained Models.py` - main script demonstrating summarization
2. `README.md` - this file with instructions

## Requirements

Python 3.7 or newer

Suggested packages

- transformers
- torch
- nltk
- numpy
- pandas

## Quick start

Clone the repository and install dependencies:

```bash
git clone https://github.com/mshahzaib7/Text-Summarization-Using-Pretrained-Models.git
cd Text-Summarization-Using-Pretrained-Models
pip install transformers torch nltk numpy pandas
```

Run the example script:

```bash
python "Text Summarization Using Pretrained Models.py"
```

The script loads a pretrained model, processes sample text, and prints a short summary.

## How to extend

You can replace the model with any transformer checkpoint, add batching, or expose a simple API endpoint.

## Author

mshahzaib7
