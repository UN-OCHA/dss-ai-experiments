# ReliefWeb Theme Classifier

This script uses a pre-trained TinyBERT model to classify humanitarian questions into relevant ReliefWeb themes. It demonstrates how to use the Hugging Face Transformers library to perform multi-label text classification.

## Features

- Utilizes the `cross-encoder/ms-marco-TinyBERT-L-2-v2` model for efficient text classification
- Classifies questions into 18 predefined ReliefWeb themes
- Returns top 3 most relevant themes with confidence scores
- Measures and reports processing time for performance analysis

## Requirements

- Python 3.6+
- PyTorch
- Transformers library

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install torch transformers
   ```

## Usage

Run the script with:

```
python main.py
```

The script will process a set of predefined test questions, displaying the top 3 relevant themes for each question along with their confidence scores and processing time.

## Output

For each question, the script outputs:
- The question text
- Top 3 relevant themes with confidence scores
- Time taken to process the question

At the end, it provides:
- Average processing time per question
- Number of questions that can be processed per second

## Customization

To classify your own questions, modify the `questions` list in the script.

## Note

The model and tokenizer are downloaded and cached locally in a `./models` directory to avoid repeated downloads.
