"""
This script analyzes the tagging accuracy for ReliefWeb training ads, comparing
automated tagging using AWS Bedrock Titan Premier against historical manual
tagging for both professional functions (career categories) and themes.
"""

# pylint: disable=broad-exception-caught,too-many-statements,too-few-public-methods,too-many-arguments,too-many-positional-arguments,unused-argument

import argparse
import os
import json
import time
import sys
import re
import csv
import textwrap
from queue import Queue, Empty
from typing import Dict, List, Tuple, Any

import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import requests
import boto3

def load_config() -> Dict[str, str]:
    """
    Load configuration from a JSON file.

    Returns:
        Dict[str, str]: Configuration dictionary.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in configuration file at {config_path}")
        sys.exit(1)

config = load_config()

# AWS Bedrock configuration
bedrock_east = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=config["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"]
)

APPNAME = "rw-training-tagging-experiments"

# Constants for rate limiting
MAX_WORKERS = 10
RATE_LIMIT_REQUESTS = 100 / 60  # 100 requests per minute
RATE_LIMIT_TOKENS = 300000 / 60  # 300,000 tokens per minute
TOKEN_BUFFER = 5000  # Average tokens per request

class RateLimiter:
    """Rate limiter for managing API request rates."""

    def __init__(self, rate_limit):
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self):
        """Attempt to acquire a token for making a request."""
        with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens += time_passed * self.rate_limit
            self.tokens = min(self.tokens, self.rate_limit)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

# Initialize rate limiters
request_limiter = RateLimiter(RATE_LIMIT_REQUESTS)
token_limiter = RateLimiter(RATE_LIMIT_TOKENS / TOKEN_BUFFER)

def load_training_data() -> Dict[str, Dict[str, str]]:
    """
    Load training data from a TSV file.

    Returns:
        Dict[str, Dict[str, str]]: Dictionary of training data.
    """
    training_data = {}
    with open("training-data.tsv", "r", newline="", encoding="utf-8") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter="\t")
        for row in reader:
            training_data[row["nid"]] = row
    return training_data

def fetch_references(reference_type: str) -> Dict[str, str]:
    """
    Fetch references (career categories or themes) from the ReliefWeb API.

    Args:
        reference_type (str): Type of reference to fetch ('professional-functions' or 'themes').

    Returns:
        Dict[str, str]: Dictionary of references.
    """
    url = f"https://api.reliefweb.int/v1/references/{reference_type}"
    params = {"appname": APPNAME}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {item["fields"]["name"]: item["fields"]["description"] for item in data["data"]}
    except requests.RequestException as e:
        print(f"Error fetching {reference_type}: {str(e)}")
        return {}

def fetch_training_ads(training_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch training details from the ReliefWeb API.

    Args:
        training_ids (List[str]): List of training IDs to fetch.

    Returns:
        List[Dict[str, Any]]: List of training details.
    """
    url = "https://api.reliefweb.int/v1/training"
    all_training_ads = []
    batch_size = 500
    for i in range(0, len(training_ids), batch_size):
        batch_ids = training_ids[i:i+batch_size]
        payload = {
            "appname": APPNAME,
            "profile": "full",
            "preset": "analysis",
            "limit": batch_size,
            "fields": {
                "include": ["id", "url", "title", "body", "career_categories", "theme"]
            },
            "filter": {
                "field": "id",
                "value": batch_ids
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            batch_training_ads = response.json().get("data", [])
            all_training_ads.extend(batch_training_ads)
        except requests.RequestException as e:
            print(f"Error fetching training ads batch {i//batch_size + 1}: {str(e)}")
    return all_training_ads

def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from a string.

    Args:
        text (str): Input text with HTML tags.

    Returns:
        str: Text with HTML tags removed.
    """
    return re.sub("<[^<]+?>", "", text)

def generate_prompt(training: Dict[str, Any], functions: Dict[str, str], themes: Dict[str, str]) -> str:
    """
    Generate an improved prompt for training classification and theme tagging.

    Args:
        training (Dict[str, Any]): Training details.
        functions (Dict[str, str]): Professional functions.
        themes (Dict[str, str]): Themes.

    Returns:
        str: Generated prompt.
    """
    training_title = strip_html_tags(training['fields']['title'])
    training_body = strip_html_tags(training['fields']['body'])

    return textwrap.dedent(f"""
You are tasked with analyzing humanitarian training ads and classifying them according to specifically relevant professional functions and focus areas. Your analysis will be conducted in three steps and should be presented in a clear XML format.

Please analyze the following training ad:

<training>
Title: {training_title}
Description:
{training_body}
</training>

Before you begin, understand these key distinctions:

Professional Functions must represent CORE TECHNICAL COMPETENCIES that form the PRIMARY LEARNING OBJECTIVES of this training. Only select functions that:
- Are explicitly taught as skills to be acquired
- Constitute at least 25% of the curriculum
- Appear in multiple contextually different parts of the description

Humanitarian Focus Areas must reflect the ACTUAL OPERATIONAL ENVIRONMENT where the trained skills would be applied. Only select areas that:
- Are explicitly mentioned as application contexts
- Form the primary operational setting
- Appear in concrete examples within the description

For your analysis, follow these steps:

1. Provide a concise summary identifying the ESSENTIAL TECHNICAL SKILLS and PRIMARY OPERATIONAL CONTEXT
2. Select professional functions (0-3) ONLY if they represent CORE TECHNICAL COMPETENCIES being taught. Use item numbers (A1-A9).
3. Select focus areas (0-3) ONLY if they constitute the PRIMARY OPERATIONAL ENVIRONMENT. Use item numbers (B1-B20).

Professional Functions:
A1) Human Resources Management and Capacity Building
A2) Logistics, Procurement, Supply Chain, Asset Maintenance, and Operational Security
A3) Project Monitoring, Evaluation, Best Practices, and Lessons Learned
A4) Donor Relations, Fundraising, and Grants Management
A5) Information Management, Data Analysis, and Mapping/Visualization
A6) Advocacy, Communications, Public Relations, Social Media, and Translation Services
A7) Program/Project Management, Implementation, and Quality Assurance
A8) Information and Communications Technology (ICT) Infrastructure and Systems Management
A9) Administration, Financial Management, and Auditing

Humanitarian Focus Areas:
B1) Safety and Security: Aid worker safety policies, field security measures.
B2) Recovery and Reconstruction: Asset/infrastructure restoration, early recovery, relief-to-development transition.
B3) Health: Emergency medical services, disease control, reproductive health, psychosocial support.
B4) Education: Temporary learning spaces, school supplies, teacher support, infrastructure rehabilitation.
B5) Disaster Management: Early warning, preparedness, prevention, risk reduction, mitigation.
B6) Coordination: Inter-cluster, civil-military, and private sector partnerships.
B7) Contributions: Financial/in-kind aid reporting and announcements.
B8) Climate Change and Environment: Climate-induced humanitarian impacts, vulnerability, displacement.
B9) HIV/AIDS: Emergency services, high prevalence consequences.
B10) Agriculture: Fisheries, animal husbandry, food security, agricultural training.
B11) Camp Coordination/Management: Displaced persons services, life quality, post-displacement preparation.
B12) Mine Action: Landmine/UXO clearance, education, victim assistance.
B13) Water Sanitation Hygiene: Emergency water provision, sanitation, hygiene promotion.
B14) Shelter and Non-Food Items: Shelter materials, household items, camp management.
B15) Protection and Human Rights: Rights violations, gender-based violence, humanitarian law, access.
B16) Peacekeeping/Peacebuilding: Conflict resolution, social/political restoration, disarmament, electoral support.
B17) Logistics and Telecommunications: Aid supply chain, transportation, ICT services.
B18) Humanitarian Financing: Donorship, funding mechanisms, accountability, partnerships.
B19) Gender: Gender-specific emergency issues, women as change agents.
B20) Food and Nutrition: Food security, aid distribution, feeding programs.

Present your analysis STRICTLY in this XML format:

<training_analysis>
  <summary>Essential technical skills and primary operational context</summary>
  <professional_functions>Comma-separated item numbers (A1-A9) or empty</professional_functions>
  <focus_areas>Comma-separated item numbers (B1-B20) or empty</focus_areas>
</training_analysis>

Example of appropriate omission for technical field training:
<training_analysis>
  <summary>Field explosives safety training for mine clearance technicians</summary>
  <professional_functions></professional_functions>
  <focus_areas>B12</focus_areas>
</training_analysis>

IMPORTANT REJECTION CRITERIA:
- Never select a professional function if:
  - Management terms appear only in organizational context
  - Planning references are about field operations rather than program management
  - Security mentions relate to operational safety rather than security systems management
- Always prefer empty tags over questionable selections
- Require explicit evidence of taught management/administrative skills for A-series tags

    """).strip()

def extract_thinking_and_answer(
    response: Dict[str, Any],
    model_type: str,
    functions: Dict[str, str],
    themes: Dict[str, str]
) -> Tuple[str, str, str]:
    """
    Extract thinking, professional functions, and themes from the model response.

    Args:
        response: The raw response from the model.
        model_type: The type of model used (e.g., "titan").
        functions: Dictionary of valid professional functions.
        themes: Dictionary of valid themes.

    Returns:
        A tuple containing:
            - thinking: The model's reasoning for its classification.
            - functions: A pipe-separated string of valid professional functions.
            - themes: A pipe-separated string of valid themes (or "-" if none valid).

    Raises:
        ValueError: If an unknown model type is provided.
    """
    try:
        if model_type != "titan":
            raise ValueError(f"Unknown model type: {model_type}")

        output_text = response["results"][0]["outputText"].strip()

        extracted_thinking = extract_thinking(output_text)
        extracted_functions = extract_terms(output_text, 'professional_functions', functions, 3, 'A')
        extracted_themes = extract_terms(output_text, 'focus_areas', themes, 3, 'B')

        formatted_functions = format_terms(extracted_functions)
        formatted_themes = format_terms(extracted_themes)

        return extracted_thinking, formatted_functions, formatted_themes

    except Exception as e:
        print(f"Error extracting thinking and answer for {model_type}: {str(e)}")
        return "-", "-", "-"

def extract_thinking(output_text: str) -> str:
    """
    Extract and format the thinking from the output text.

    Args:
        output_text: The full output text to search in.

    Returns:
        A formatted string containing the extracted thinking.
    """
    parts = ['summary', 'functions_reasoning', 'themes_reasoning']
    extracted = [extract_tagged_content(output_text, part) for part in parts]
    thinking = " ".join(filter(None, extracted))
    return " ".join(thinking.split())  # Normalize whitespace

def extract_terms(
    output_text: str,
    tag: str,
    term_dict: Dict[str, str],
    max_terms: int,
    prefix: str
) -> List[str]:
    """
    Extract and validate terms (functions or themes) from the output text.

    Args:
        output_text: The full output text to search in.
        tag: The XML-like tag to look for (e.g., 'professional_functions' or 'focus_areas').
        term_dict: Dictionary of valid terms (functions or themes).
        max_terms: Maximum number of terms to return.
        prefix: The prefix used in the list (e.g., 'A' or 'B').

    Returns:
        List of valid terms, up to max_terms in number.
    """
    content = extract_tagged_content(output_text, tag)
    if not content:
        return []

    term_list = list(term_dict.keys())
    valid_terms = []

    # Regular expression to match the format 'A1', 'B2', etc.
    pattern = rf'{prefix}(\d+)'
    matches = re.findall(pattern, content)

    for match in matches:
        index = int(match) - 1  # Convert to 0-based index
        if 0 <= index < len(term_list):
            valid_terms.append(term_list[index])
            if len(valid_terms) == max_terms:
                break

    return valid_terms

def extract_tagged_content(text: str, tag: str) -> str:
    """
    Extract content between XML-like tags.

    Args:
        text: The text to search in.
        tag: The tag to look for.

    Returns:
        The content between the specified tags, or an empty string if not found.
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def format_terms(terms: List[str]) -> str:
    """
    Format a list of terms into a sorted, pipe-separated string or '-' if empty.

    Args:
        terms: List of terms to format.

    Returns:
        A pipe-separated string of sorted terms, or "-" if the list is empty.
    """
    return ",".join(sorted(terms)) if terms else "-"

def query_bedrock_titan(client: Any, prompt: str) -> Tuple[Dict[str, Any], int, int]:
    """
    Query AWS Bedrock Titan model with rate limiting.

    Args:
        client (Any): Bedrock client.
        prompt (str): Input prompt.

    Returns:
        Tuple[Dict[str, Any], int, int]: Tuple containing response body, input tokens, and output tokens.
    """
    while not request_limiter.acquire() or not token_limiter.acquire():
        time.sleep(0.1)

    try:
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 1024,
                "stopSequences": [],
                "temperature": 0.0,
                "topP": 0.1
            }
        })
        response = client.invoke_model(
            body=body,
            modelId="amazon.titan-text-premier-v1:0",
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        input_tokens = response_body["inputTextTokenCount"]
        output_tokens = response_body["results"][0]["tokenCount"]
        return response_body, input_tokens, output_tokens
    except Exception as e:
        print(f"Error querying AWS Bedrock Titan: {str(e)}")
        return None, 0, 0

def get_processed_training_ids(output_file: str) -> set:
    """
    Get the set of already processed training IDs.

    Args:
        output_file (str): Path to output file.

    Returns:
        set: Processed training IDs.
    """
    processed_training_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", newline="", encoding="utf-8") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter="\t")
            for row in reader:
                processed_training_ids.add(row["Training ID"])
    return processed_training_ids

def process_training(
    training: Dict[str, Any],
    training_data: Dict[str, Dict[str, str]],
    functions: Dict[str, str],
    themes: Dict[str, str],
    models: List[Tuple[str, Any, str, str]],
    training_index: int,
    total_training_ads: int
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Process a single training ad using multiple AI models for category and theme prediction.

    Args:
        training (Dict[str, Any]): Training data from API.
        training_data (Dict[str, Dict[str, str]]): Additional metadata.
        functions (Dict[str, str]): Valid professional functions.
        themes (Dict[str, str]): Valid themes.
        models (List[Tuple[str, Any, str, str]]): List of models to use.
            Each tuple contains (model_name, query_function, model_type, region).
        training_index (int): Current training index.
        total_training_ads (int): Total number of training ads.

    Returns:
        Tuple[Dict[str, Any], List[str]]: A tuple containing:
            - training_result (Dict[str, Any]): Dictionary with training details and model predictions.
            - output_lines (List[str]): List of formatted strings for logging or display.
    """
    prompt = generate_prompt(training, functions, themes)

    # Extract actual classifications.
    actual_functions = ",".join(sorted([
        func["name"] for func in training["fields"].get("career_categories", [])
    ])) or "Not specified"

    actual_themes = ",".join(sorted([
        theme["name"] for theme in training["fields"].get("theme", [])
    ])) or "Not specified"

    training_result = {
        "Training ID": training["id"],
        "Training URL": training["fields"]["url"],
        "Training Title": training["fields"]["title"],
        "Posted": training_data[training["id"]]["posted"],
        "Editor": training_data[training["id"]]["editor"],
        "Trusted": training_data[training["id"]]["trusted"],
        "Reviewed": training_data[training["id"]]["reviewed"],
        "Actual Professional Functions": actual_functions,
        "Actual Themes": actual_themes
    }

    output_lines = [
        f"Processing Training {training_index}/{total_training_ads}: "
        f"{training['fields']['title']} - "
        f"Actual Functions: {actual_functions}, "
        f"Actual Themes: {actual_themes}"
    ]

    # Process through each model.
    for model_name, query_func, model_type, region in models:
        model_start_time = time.time()
        response, input_tokens, output_tokens = query_func(prompt)
        model_end_time = time.time()
        response_time = model_end_time - model_start_time

        if response is None:
            thinking, llm_functions, llm_themes = "-", "-", "-"
        else:
            thinking, llm_functions, llm_themes = extract_thinking_and_answer(response, model_type, functions, themes)

        # Store model results.
        training_result.update({
            f"{model_name} - Professional Functions": llm_functions,
            f"{model_name} - Themes": llm_themes,
            f"{model_name} - Reason": thinking,
            f"{model_name} - Region": region,
            f"{model_name} - Time": f"{response_time:.2f}",
            f"{model_name} - Input Tokens": input_tokens,
            f"{model_name} - Output Tokens": output_tokens
        })

        output_lines.append(
            f"{model_name} - Functions: {llm_functions}, "
            f"Themes: {llm_themes}, Time: {response_time:.2f}s, "
            f"Tokens: {input_tokens}/{output_tokens}"
        )

    return training_result, output_lines

def process_output_queue(queue: Queue, total: int, start_time: float) -> None:
    """
    Process the output queue in order and print results to console.

    Args:
        queue (Queue): Output queue.
        total (int): Total number of training ads to process.
        start_time (float): Processing start time.
    """
    next_index = 1
    buffer = {}

    while next_index <= total:
        if next_index in buffer:
            result, lines, index = buffer.pop(next_index)
            print_training_result(result, lines, index, total, start_time)
            next_index += 1
        else:
            try:
                result, lines, index = queue.get(timeout=1)
                if index == next_index:
                    print_training_result(result, lines, index, total, start_time)
                    next_index += 1
                else:
                    buffer[index] = (result, lines, index)
            except Empty:
                continue

def print_training_result(
    result: Dict[str, Any],
    lines: List[str],
    index: int,
    total: int,
    start_time: float
) -> None:
    """
    Print the result of a single training ad processing.

    Args:
        result (Dict[str, Any]): Dictionary containing training processing results.
        lines (List[str]): List of output lines to be printed.
        index (int): Index of the current training.
        total (int): Total number of training ads to be processed.
        start_time (float): Start time of the training ad processing.
    """
    for line in lines:
        print(line)

    elapsed = time.time() - start_time
    avg_time = elapsed / index
    remaining = avg_time * (total - index)

    print(f"Training {index}/{total} completed")
    print(f"Elapsed: {elapsed:.2f}s | Remaining: {remaining:.2f}s")
    print("=" * 50)

def reset_output_file(file_path: str) -> None:
    """
    Reset the output file by removing it or emptying its contents.

    Args:
        file_path (str): Path to the output file.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
    print(f"Output file {file_path} has been reset.")

def main() -> None:
    """
    Main function to process training ads and classify them using parallel processing.
    """
    parser = argparse.ArgumentParser(description="Process and classify ReliefWeb training ads.")
    parser.add_argument("--reset", action="store_true", help="Reset the output file before processing")
    args = parser.parse_args()

    output_file = "training_classification_results.tsv"

    if args.reset:
        reset_output_file(output_file)

    processed_training_ids = get_processed_training_ids(output_file)
    training_data = load_training_data()

    # Fetch reference data.
    functions = fetch_references("career-categories")
    themes = fetch_references("themes")

    if not functions or not themes:
        print("Failed to fetch reference data")
        return

    # Get unprocessed training ads.
    training_ids = [training_id for training_id in training_data if training_id not in processed_training_ids]
    training_ads = fetch_training_ads(training_ids)

    if not training_ads:
        print("No training ads to process")
        return

    # Configure models.
    models = [
        ("AWS Bedrock Titan",
         lambda p: query_bedrock_titan(bedrock_east, p),
         "titan",
         "us-east-1")
    ]

    # Configure output fields.
    fieldnames = [
        "Training ID", "Training URL", "Training Title", "Posted",
        "Editor", "Trusted", "Reviewed", "Actual Professional Functions",
        "Actual Themes"
    ]

    for model in models:
        fieldnames.extend([
            f"{model[0]} - Professional Functions",
            f"{model[0]} - Themes",
            f"{model[0]} - Reason",
            f"{model[0]} - Region",
            f"{model[0]} - Time",
            f"{model[0]} - Input Tokens",
            f"{model[0]} - Output Tokens"
        ])

    # Process training ads.
    file_exists = os.path.exists(output_file)
    with open(output_file, "a", newline="", encoding="utf-8") as tsvfile:
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            writer.writeheader()

        total = len(training_ads)
        start_time = time.time()
        output_queue = Queue()

        # Start output thread
        output_thread = threading.Thread(
            target=process_output_queue,
            args=(output_queue, total, start_time)
        )
        output_thread.start()

        # Process in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            process_fn = partial(
                process_training,
                training_data=training_data,
                functions=functions,
                themes=themes,
                models=models,
                total_training_ads=total
            )

            futures = {
                executor.submit(process_fn, training, training_index=i+1): (training, i+1)
                for i, training in enumerate(training_ads)
            }

            for future in concurrent.futures.as_completed(futures):
                training, index = futures[future]
                try:
                    result, lines = future.result()
                    writer.writerow(result)
                    tsvfile.flush()
                    output_queue.put((result, lines, index))
                except Exception as e:
                    print(f"Error processing {training['id']}: {str(e)}")

        output_queue.put(None)
        output_thread.join()

    print(f"Results written to {output_file}")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
