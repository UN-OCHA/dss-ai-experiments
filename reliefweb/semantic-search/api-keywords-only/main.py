"""
This script processes a humanitarian question, extracts keywords, searches for relevant
documents, and ranks passages to provide the most relevant information.
"""

import os
import time
from functools import wraps
from typing import List, Dict, Any, Callable, Set
import requests
from flashrank import Ranker, RerankRequest
from gensim.models import KeyedVectors
from gensim import downloader as api
import spacy

# Constants
MODEL_NAME = 'glove-wiki-gigaword-50'
MODEL_DIR = './models'
MODEL_PATH = os.path.join(MODEL_DIR, f'{MODEL_NAME}.kv')
TEMP_MODEL_PATH = os.path.join(MODEL_DIR, f'{MODEL_NAME}_temp.kv')

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def timed_function(func: Callable) -> Callable:
    """Decorator to measure and print the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@timed_function
def download_and_prepare_model():
    """Download and prepare the word embedding model."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading and preparing {MODEL_NAME} model...")
        model = api.load(MODEL_NAME)
        model.save(TEMP_MODEL_PATH)
        temp_model = KeyedVectors.load(TEMP_MODEL_PATH, mmap='r')
        temp_model.save(MODEL_PATH)
        os.remove(TEMP_MODEL_PATH)
        print("Model prepared and saved.")
    else:
        print("Model already prepared.")

@timed_function
def load_model():
    """Load the prepared word embedding model."""
    print("Loading model...")
    return KeyedVectors.load(MODEL_PATH, mmap='r')

print("Loading spaCy model...")
NLP = spacy.load('en_core_web_sm', exclude=['tok2vec', 'tagger', 'parser',
                                            'attribute_ruler', 'lemmatizer', 'ner'])
NLP.enable_pipe('senter')

download_and_prepare_model()
GLOVE_MODEL = load_model()

print("Initializing FlashRank Ranker...")
RANKER = Ranker(max_length=256)

@timed_function
def extract_keywords(text: str) -> List[str]:
    """Extract keywords from the given text using spaCy."""
    doc = NLP(text)
    return [token.text.lower() for token in doc if not token.is_punct and not token.is_stop]

@timed_function
def get_similar_words(keywords: List[str], num_similar: int) -> Set[str]:
    """Get similar words for given keywords using the GloVe model."""
    similar_words = set()
    for keyword in keywords:
        cleaned_word = keyword.replace(',', '').replace(' ', '')
        if cleaned_word != keyword:
            continue

        try:
            similar = GLOVE_MODEL.most_similar(keyword, topn=num_similar)
            similar_words.update(w for w, _ in similar)
        except KeyError:
            continue
    return similar_words

@timed_function
def rank_passages(query: str, passages: List[dict], limit: int) -> List[str]:
    """Rank passages with metadata using the FlashRank Ranker."""
    if limit < 1:
        return []
    ranked = RANKER.rerank(RerankRequest(query=query, passages=passages))
    return ranked[:limit]

@timed_function
def rank_texts(query: str, texts: List[str], limit: int) -> List[str]:
    """Rank texts using the FlashRank Ranker."""
    if limit < 1:
        return []
    passages = [{'text': text} for text in texts]
    ranked = RANKER.rerank(RerankRequest(query=query, passages=passages))
    return [item['text'] for item in ranked[:limit]]

@timed_function
def search_reliefweb(query: str) -> List[Dict[str, Any]]:
    """Search the ReliefWeb API for relevant documents."""
    url = "https://api.reliefweb.int/v1/reports"
    params = {
        "appname": "rw-semantic-search-experiments",
        "query": {
            "value": query,
            "fields": ["title", "body"],
            "operator": "OR"
        },
        "fields": {
            "include": ["id", "title", "body", "source.name", "source.shortname", "url_alias"]
        },
        "sort": ["score:desc"],
        "limit": 10
    }

    try:
        response = requests.post(url, json=params, timeout=10)
        response.raise_for_status()
    except requests.Timeout:
        print("The request to the ReliefWeb API timed out")
        return []
    except requests.RequestException as exception:
        print(f"An error occurred while requesting the ReliefWeb API: {exception}")
        return []

    return response.json()['data']

@timed_function
def split_and_rank_passages(question: str, documents: List[Dict[str, Any]]) -> List[str]:
    """Split documents into passages, rank them, and return top passages."""
    all_passages = []
    for doc in documents:
        passages = split_into_passages(doc['fields']['body'])
        ranked_passages = rank_texts(question, passages, 2)

        for passage in ranked_passages:
            text = f"{doc['fields']['title']}\n\n{passage}"
            source = doc['fields']['source'][0]['shortname'] or doc['fields']['source'][0]['name']
            all_passages.append({
                'text': text,
                'meta': {
                    'id': doc['fields']['id'],
                    'url': doc['fields']['url_alias'],
                    'title': doc['fields']['title'],
                    'source': source
                }
            })
    return rank_passages(question, all_passages, 5)

def split_into_passages(text: str, max_chars: int = 500) -> List[str]:
    """Split a text into passages of maximum length."""
    doc = NLP(text)
    passages = []
    current_passage = ""
    for sent in doc.sents:
        if len(current_passage) + len(sent.text) <= max_chars:
            current_passage += sent.text + " "
        else:
            passages.append(current_passage.strip())
            current_passage = sent.text + " "
    if current_passage:
        passages.append(current_passage.strip())
    return passages

def main() -> None:
    """Main function to process the question and output relevant passages."""
    start_time = time.time()

    question = input("Please enter your humanitarian question: ")
    print(f"\nProcessing question: '{question}'\n")

    print("Step 1: Extracting keywords...")
    keywords = extract_keywords(question)
    print(f"Extracted keywords: {', '.join(keywords)}\n")

    print("Step 2-3: Finding similar words...")
    num_similar = 10
    similar_words = get_similar_words(keywords, num_similar)
    print(similar_words)
    print(f"Found {len(similar_words)} similar words\n")

    print("Step 4-5: Ranking keywords...")
    ranked_keywords = keywords + rank_texts(question, similar_words, 30 - len(keywords))
    print(ranked_keywords)
    print(f"Top 5 ranked keywords: {', '.join(ranked_keywords[:5])}\n")

    print("Step 6: Searching ReliefWeb...")
    search_query = " ".join(ranked_keywords)
    documents = search_reliefweb(search_query)
    print(f"Found {len(documents)} relevant documents\n")

    print("Step 7-8: Splitting documents into passages and ranking them...")
    final_passages = split_and_rank_passages(question, documents)
    print(f"Selected top {len(final_passages)} passages\n")

    print("Step 9: Adding metadata and outputting results...")
    for i, passage in enumerate(final_passages, 1):
        print(f"Passage {i}:")
        print(f"Text: {passage['text']}")
        print(f"Title: {passage['meta']['title']}")
        print(f"Source: {passage['meta']['source']}")
        print(f"URL: {passage['meta']['url']}")
        print()

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
