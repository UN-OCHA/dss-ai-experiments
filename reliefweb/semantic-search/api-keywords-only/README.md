# Humanitarian Question Answering System

This Python script processes humanitarian questions, extracts keywords, searches for relevant documents, and ranks passages to provide the most relevant information.

## Features

- Keyword extraction from input questions
- Similar word finding using word embeddings
- Document search using the ReliefWeb API
- Passage ranking and selection
- Metadata addition to final results

## Requirements

- Python 3.7+
- spaCy
- gensim
- flashrank
- requests

## Installation

1. Clone this repository:
2. Install the required packages:
   ```
   pip install spacy gensim flashrank requests
   ```

3. Download the spaCy English model:
   ```
   python -m spacy download en_core_web_sm
   ```

## Usage

Run the script:

```
python main.py
```

Follow the prompts to enter your humanitarian question. The system will process the question and provide relevant passages with metadata.

## How it works

1. Extracts keywords from the input question
2. Finds similar words using GloVe word embeddings
3. Ranks keywords and similar words
4. Searches for relevant documents using the ReliefWeb API
5. Splits documents into passages and ranks them
6. Selects top passages and adds metadata
7. Outputs the results

## Configuration

- `MODEL_NAME`: The name of the word embedding model (default: 'glove-wiki-gigaword-50')
- `MODEL_DIR`: Directory to store the word embedding model
- `RANKER`: FlashRank Ranker configuration
