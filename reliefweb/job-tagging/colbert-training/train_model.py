"""
Script to train a ColBERT model for job tagging using the ReliefWeb dataset.
"""

import json
from typing import List, Tuple, Dict, Any

import torch
from neural_cherche import models, rank, retrieve, train, utils

def get_test_candidates(documents: List[Dict[str, str]], queries: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve test candidates using BM25 retrieval.

    Args:
        documents: List of documents to search.
        queries: List of queries to search with.

    Returns:
        A dictionary of candidates for each query.
    """
    # Setting up the retriever.
    retriever = retrieve.BM25(
        key="id",
        on=["text"],
    )

    retriever_documents_embeddings = retriever.encode_documents(
        documents=documents,
    )

    retriever.add(
        documents_embeddings=retriever_documents_embeddings,
    )

    queries_embeddings = retriever.encode_queries(
        queries=queries,
    )

    candidates = retriever(
        queries_embeddings=queries_embeddings,
        k=100,
    )

    return candidates

def get_test_data(triples: List[Tuple[str, str, str]]) -> Tuple[List[Dict[str, str]], List[str], Dict[str, Dict[str, int]], Dict[str, List[Dict[str, Any]]]]:
    """
    Prepare test data from triples.

    Args:
        triples: List of (query, positive, negative) triples.

    Returns:
        A tuple containing documents, queries, qrels, and candidates.
    """
    unique_texts = set()
    documents = []
    queries = []
    qrels: Dict[str, Dict[str, int]] = {}
    text_to_id = {}

    for query, positive, negative in triples:
        # Add to queries list if not already present.
        if query not in queries:
            queries.append(query)

        # Process positive document.
        if positive not in unique_texts:
            unique_texts.add(positive)
            doc_id = f"doc_{len(documents)}"
            documents.append({"id": doc_id, "text": positive})
            text_to_id[positive] = doc_id
        else:
            doc_id = text_to_id[positive]

        # Update qrels.
        if query not in qrels:
            qrels[query] = {}
        qrels[query][doc_id] = 1  # Positive document is relevant.

        # Process negative document.
        if negative not in unique_texts:
            unique_texts.add(negative)
            doc_id = f"doc_{len(documents)}"
            documents.append({"id": doc_id, "text": negative})
            text_to_id[negative] = doc_id

    candidates = get_test_candidates(documents, queries)

    return documents, queries, qrels, candidates

def load_triples(filename: str) -> List[Tuple[str, str, str]]:
    """
    Load triples from a JSON file.

    Args:
        filename: Path to the JSON file.

    Returns:
        A list of (query, positive, negative) triples.
    """
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)
        return [(item["query"], item["positive"], item["negative"]) for item in data]

def main() -> None:
    """Main function to train the ColBERT model."""
    train_triples = load_triples("job_tagger_training_train.json")
    test_triples = load_triples("job_tagger_training_test.json")

    test_documents, test_queries, test_qrels, test_candidates = get_test_data(test_triples)

    model = models.ColBERT(
        model_name_or_path="Alibaba-NLP/gte-en-mlm-base",
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length_query=2048,
        max_length_document=64,
        trust_remote_code=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # Training loop.
    batch_size = 5
    epochs = 4

    for step, (anchor, positive, negative) in enumerate(
        utils.iter(train_triples, epochs=epochs, batch_size=batch_size, shuffle=True)
    ):
        train.train_colbert(
            model=model,
            optimizer=optimizer,
            anchor=anchor,
            positive=positive,
            negative=negative,
            step=step,
            gradient_accumulation_steps=50,
        )

        # Evaluate the model every 180 steps.
        if (step + 1) % 180 == 0:
            # Setting up the ranker.
            ranker = rank.ColBERT(key="id", on=["text"], model=model)

            ranker_queries_embeddings = ranker.encode_queries(
                queries=test_queries,
                batch_size=batch_size,
            )

            ranker_documents_embeddings = ranker.encode_candidates_documents(
                documents=test_documents,
                candidates=test_candidates,
                batch_size=batch_size,
            )

            scores = ranker(
                documents=test_candidates,
                queries_embeddings=ranker_queries_embeddings,
                documents_embeddings=ranker_documents_embeddings,
                k=10,
            )

            # Evaluate the pipeline.
            evaluation_scores = utils.evaluate(
                scores=scores,
                qrels=test_qrels,
                queries=test_queries,
                metrics=["ndcg@10"] + [f"hits@{k}" for k in range(1, 10)],
            )

            print(evaluation_scores)

    model.save_pretrained("colbert-reliefweb")

if __name__ == "__main__":
    main()
