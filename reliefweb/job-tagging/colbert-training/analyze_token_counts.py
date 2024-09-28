"""
Script to analyze token count distribution in training data for a job tagging model.
"""

import json
from collections import defaultdict
from typing import Dict, List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

def get_range(count: int) -> str:
    """
    Determine the token count range for a given count.

    Args:
        count: The token count to categorize.

    Returns:
        A string representing the range the count falls into.
    """
    ranges = [0, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 25116, float('inf')]
    for i in range(len(ranges) - 1):
        if ranges[i] <= count < ranges[i+1]:
            return f"{ranges[i]}-{ranges[i+1]-1}"
    return f"{ranges[-2]}+"

def load_training_data(file_path: str) -> List[Dict[str, str]]:
    """
    Load training data from a JSON file.

    Args:
        file_path: Path to the JSON file containing training data.

    Returns:
        A list of dictionaries containing training triples.
    """
    print("Loading training data...")
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_token_counts(
    train_triples: List[Dict[str, str]],
    tokenizer: AutoTokenizer
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Analyze token counts for queries and documents in the training data.

    Args:
        train_triples: List of training data triples.
        tokenizer: The tokenizer to use for encoding text.

    Returns:
        A tuple containing dictionaries of query and document token count distributions.
    """
    query_token_counts: Dict[str, int] = defaultdict(int)
    document_token_counts: Dict[str, int] = defaultdict(int)

    print("Analyzing token counts...")
    for item in tqdm(train_triples, desc="Processing items"):
        # Tokenize query.
        query_tokens = tokenizer.encode(item['query'], add_special_tokens=True)
        query_token_counts[get_range(len(query_tokens))] += 1

        # Tokenize positive document.
        pos_doc_tokens = tokenizer.encode(item['positive'], add_special_tokens=True)
        document_token_counts[get_range(len(pos_doc_tokens))] += 1

        # Tokenize negative document.
        neg_doc_tokens = tokenizer.encode(item['negative'], add_special_tokens=True)
        document_token_counts[get_range(len(neg_doc_tokens))] += 1

    return query_token_counts, document_token_counts

def print_distribution(
    distribution: Dict[str, int],
    total: int,
    title: str
) -> None:
    """
    Print the distribution of token counts.

    Args:
        distribution: Dictionary containing the distribution data.
        total: Total count for percentage calculation.
        title: Title for the distribution output.
    """
    print(f"\n{title}:")
    for range_, count in sorted(distribution.items()):
        print(f"{range_}: {count}")

    print(f"\n{title} (percentage):")
    for range_, count in sorted(distribution.items()):
        percentage = (count / total) * 100
        print(f"{range_}: {percentage:.2f}%")

def main() -> None:
    """Main function to run the token count analysis."""
    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-en-mlm-base")

    # Load training data.
    train_triples = load_training_data('job_tagger_training_train.json')

    # Analyze token counts.
    query_token_counts, document_token_counts = analyze_token_counts(train_triples, tokenizer)

    # Calculate totals.
    total_queries = sum(query_token_counts.values())
    total_documents = sum(document_token_counts.values())

    # Print distributions.
    print_distribution(query_token_counts, total_queries, "Query token count distribution")
    print_distribution(document_token_counts, total_documents, "Document token count distribution")

if __name__ == "__main__":
    main()
