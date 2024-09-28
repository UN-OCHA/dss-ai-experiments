# ReliefWeb Job Tagger ColBERT Model

## Overview

This project trains a ColBERT model for tagging jobs on ReliefWeb based on career categories. The model is designed to improve the accuracy of job categorization by leveraging advanced natural language processing techniques.

## Model Details

- **Base Model**: Alibaba-NLP/gte-en-mlm-base
- **Architecture**: ColBERT (Contextualized Late Interaction over BERT)
- **Training Framework**: neural_cherche

## Dataset

The dataset is created from the ReliefWeb API, consisting of job listings and their associated career categories. The data is split into training, validation, and test sets, with each entry containing:

- **Query**: Job title and description
- **Positive**: Description of the correct career category
- **Negative**: Description of a randomly selected incorrect career category

## Training Process

The model was trained for 4 epochs with the following parameters:

- **Batch Size**: 5
- **Learning Rate**: 3e-5
- **Optimizer**: AdamW
- **Gradient Accumulation Steps**: 50

## Performance Improvement

The model showed significant improvement in accuracy over the course of training:

| Metric    | Epoch 0     | Epoch 1     | Epoch 2     | Epoch 3     |
|-----------|-------------|-------------|-------------|-------------|
| NDCG@10   | 0.7145      | 0.7586      | 0.7754      | 0.7787      |
| Hits@1    | 0.4778      | 0.5389      | 0.5667      | 0.5667      |
| Hits@5    | 0.8556      | 0.9000      | 0.9222      | 0.9111      |

The final model achieved an NDCG@10 score of 0.7787 and a Hits@1 score of 0.5667, showing a substantial improvement from the initial epoch.

## Helper Scripts

This project includes several helper scripts to facilitate various tasks:

1. **Model Check Script** (`check_model_support.py`): Checks if a specified model supports masked language modeling.
2. **Token Count Analysis Script** (`analyze_token_counts.py`): Analyzes token count distributions for queries and documents in the training dataset.

## Usage

### Running the Scripts

To run the different scripts, follow these instructions:

1. **Prepare Dataset Script**:
   - Run the dataset preparation script to fetch and process data from the ReliefWeb API.
   ```bash
   python prepare_dataset.py
   ```

2. **Train Model Script**:
   - Train the ColBERT model using the prepared dataset.
   ```bash
   python train_model.py
   ```

## Requirements

- Python 3.7+
- PyTorch
- neural_cherche
- transformers
- tqdm
- requests

## Future Work

- Fine-tune hyperparameters for potentially better performance.
- Experiment with different base models.
- Expand the dataset with more diverse job listings.
- Implement the model in a production environment for real-time job tagging.

## Acknowledgements

This project uses data from the ReliefWeb API and builds upon the work of the Alibaba-NLP team for the base model.
