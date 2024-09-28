"""
Script to prepare datasets for training a ColBERT model for job tagging.
"""

import json
import logging
import random
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional

import markdown
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from transformers import AutoTokenizer

# Set up logging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the tokenizer.
tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-en-mlm-base")

# ReliefWeb API appname.
APPNAME = "rw-job-tagger-dataset"

def requests_retry_session(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: Tuple[int, ...] = (500, 502, 504),
    session: Optional[requests.Session] = None,
) -> requests.Session:
    """
    Set up a session with retries for API requests.

    Args:
        retries: Number of retries.
        backoff_factor: Backoff factor for retries.
        status_forcelist: HTTP status codes to retry on.
        session: Existing session to use, if any.

    Returns:
        A requests Session object with retry configuration.
    """
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_career_categories() -> Dict[str, Dict[str, str]]:
    """
    Fetch career categories from the ReliefWeb API.

    Returns:
        A dictionary of career categories with their IDs, names, and descriptions.
    """
    url = "https://api.reliefweb.int/v1/references/career-categories"
    params = {"appname": APPNAME}
    try:
        response = requests_retry_session().get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {item['fields']['id']: {
            'name': item['fields']['name'],
            'description': item['fields']['description']
        } for item in data['data']}
    except requests.exceptions.RequestException as e:
        logging.error("Error fetching career categories: %s", e)
        return {}

def get_jobs_for_category(category_id: str, total: int = 5000) -> List[Dict]:
    """
    Fetch jobs for a specific category from the ReliefWeb API.

    Args:
        category_id: ID of the career category.
        total: Total number of jobs to fetch.

    Returns:
        A list of job dictionaries.
    """
    url = "https://api.reliefweb.int/v1/jobs"
    jobs = []
    offset = 0
    limit = 1000
    with tqdm(total=total, desc=f"Fetching jobs for category {category_id}") as pbar:
        while len(jobs) < total:
            params = {
                "appname": APPNAME,
                "profile": "list",
                "preset": "analysis",
                "limit": min(limit, total - len(jobs)),
                "offset": offset,
                "sort": ["id:desc"],
                "filter": {
                    "field": "career_categories.id",
                    "value": category_id
                },
                "fields": {
                    "include": ["id", "title", "body", "career_categories"]
                }
            }
            try:
                response = requests_retry_session().post(url, json=params, timeout=300)
                response.raise_for_status()
                data = response.json()
                if 'data' not in data or not data['data']:
                    logging.info("No more data available for category %s", category_id)
                    break
                jobs.extend(data['data'])
                pbar.update(len(data['data']))
                offset += len(data['data'])
                if len(data['data']) < limit:
                    break
                time.sleep(1)  # Be nice to the API.
            except requests.exceptions.RequestException as e:
                logging.error("Error fetching jobs for category %s: %s", category_id, e)
                time.sleep(5)  # Wait a bit longer before retrying.
    return jobs[:total]

def markdown_to_text(md_string: str) -> str:
    """
    Convert markdown to plain text.

    Args:
        md_string: Markdown formatted string.

    Returns:
        Plain text version of the input markdown.
    """
    html = markdown.markdown(md_string)
    text = re.sub('<[^<]+?>', '', html)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def get_token_count(text: str) -> int:
    """
    Get the token count for a given text.

    Args:
        text: Input text.

    Returns:
        Number of tokens in the text.
    """
    return len(tokenizer.encode(text))

def process_job(job: Dict) -> Dict[str, Union[Dict, str, int]]:
    """
    Process a job by combining title and body, and counting tokens.

    Args:
        job: Job dictionary.

    Returns:
        Processed job dictionary with query and token count.
    """
    query = f"{job['fields']['title']} {markdown_to_text(job['fields'].get('body', ''))}"
    token_count = get_token_count(query)
    return {
        'job': job,
        'query': query,
        'token_count': token_count
    }

def distribute_texts(job_texts: List[Dict], group_sizes: Optional[List[int]] = None) -> List[List[Dict]]:
    """
    Distribute texts into groups based on token count ranges.

    Args:
        job_texts: List of processed job dictionaries.
        group_sizes: List of desired group sizes.

    Returns:
        List of groups of job texts.
    """
    group_sizes = group_sizes or [100, 20, 10]
    filtered_texts = [text for text in job_texts if text['token_count'] <= 2048]

    grouped_texts = defaultdict(list)
    for text in filtered_texts:
        token_range = (text['token_count'] - 1) // 512  # 0 for 0-511, 1 for 512-1023, etc.
        grouped_texts[token_range].append(text)

    total_texts = len(filtered_texts)
    range_proportions = {range_: len(texts) / total_texts for range_, texts in grouped_texts.items()}

    result_groups = []
    used_texts = set()
    for group_size in group_sizes:
        group = []
        for range_, proportion in range_proportions.items():
            target_count = int(group_size * proportion)
            available_texts = [text for text in grouped_texts[range_] if text['query'] not in used_texts]
            selected_texts = random.sample(available_texts, min(target_count, len(available_texts)))
            group.extend(selected_texts)
            used_texts.update(text['query'] for text in selected_texts)

        if len(group) < group_size:
            remaining_texts = [text for text in filtered_texts if text['query'] not in used_texts]
            additional_texts = random.sample(remaining_texts, min(group_size - len(group), len(remaining_texts)))
            group.extend(additional_texts)
            used_texts.update(text['query'] for text in additional_texts)

        result_groups.append(group)

    return result_groups

def generate_triples(jobs: List[Dict], categories: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Generate triples (query, positive, negative) for training.

    Args:
        jobs: List of processed job dictionaries.
        categories: Dictionary of career categories.

    Returns:
        List of triples for training.
    """
    triples = []
    for job in jobs:
        if 'career_categories' not in job['job']['fields'] or not job['job']['fields']['career_categories']:
            logging.warning("Job %s has no career categories, skipping", job['job']['id'])
            continue

        positive_category = job['job']['fields']['career_categories'][0]['id']
        negative_categories = [c for c in categories.keys() if c != positive_category]
        if not negative_categories:
            logging.warning("No negative categories available for job %s, skipping", job['job']['id'])
            continue

        negative_category = random.choice(negative_categories)
        triples.append({
            "query": job['query'],
            "positive": markdown_to_text(categories[positive_category]['description']),
            "negative": markdown_to_text(categories[negative_category]['description'])
        })

    return triples

def main() -> None:
    """Main function to prepare the dataset."""
    categories = get_career_categories()
    if not categories:
        logging.error("Failed to retrieve career categories. Exiting.")
        return

    all_data = {cat_id: [] for cat_id in categories.keys()}
    for category_id in categories.keys():
        jobs = get_jobs_for_category(category_id, total=5000)
        all_data[category_id] = [process_job(job) for job in jobs]
        logging.info("Retrieved and processed %d jobs for category %s", len(all_data[category_id]), category_id)

    train_data, val_data, test_data = {}, {}, {}
    for category_id, jobs in all_data.items():
        result = distribute_texts(jobs)
        train, val, test = result if result and len(result) == 3 else (None, None, None)
        if train and val and test:
            train_data[category_id] = train
            val_data[category_id] = val
            test_data[category_id] = test

    train_triples, val_triples, test_triples = [], [], []
    for category_id in categories.keys():
        train_triples.extend(generate_triples(train_data[category_id], categories))
        val_triples.extend(generate_triples(val_data[category_id], categories))
        test_triples.extend(generate_triples(test_data[category_id], categories))

    logging.info("Generated %d training triples", len(train_triples))
    logging.info("Generated %d validation triples", len(val_triples))
    logging.info("Generated %d test triples", len(test_triples))

    # Print token count distributions.
    for dataset_name, data in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
        token_counts = defaultdict(int)
        for category in data.values():
            for item in category:
                token_range = (item['token_count'] - 1) // 512
                token_counts[token_range] += 1
        total = sum(token_counts.values())
        print(f"\n{dataset_name} token count distribution:")
        for range_, count in sorted(token_counts.items()):
            lower = range_ * 512
            upper = (range_ + 1) * 512 - 1
            percentage = count / total * 100
            print(f"{lower}-{upper}: {count} ({percentage:.2f}%)")

    # Save the data.
    with open('job_tagger_training_train.json', 'w', encoding="utf-8") as f:
        json.dump(train_triples, f)
    with open('job_tagger_training_validate.json', 'w', encoding="utf-8") as f:
        json.dump(val_triples, f)
    with open('job_tagger_training_test.json', 'w', encoding="utf-8") as f:
        json.dump(test_triples, f)

    logging.info("Data saved to JSON files.")

if __name__ == "__main__":
    main()
