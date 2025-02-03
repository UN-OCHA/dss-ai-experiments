"""
This script analyzes the tagging accuracy for ReliefWeb jobs, comparing
automated tagging using AWS Bedrock Titan Premier against historical manual
tagging for both career categories and themes.
"""

# pylint: disable=broad-exception-caught,too-many-statements,too-few-public-methods,too-many-arguments,too-many-positional-arguments,unused-argument

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

APPNAME = "rw-job-tagging-experiments"

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

def load_job_data() -> Dict[str, Dict[str, str]]:
    """
    Load job data from a TSV file.

    Returns:
        Dict[str, Dict[str, str]]: Dictionary of job data.
    """

    job_data = {}
    with open("job-data.tsv", "r", newline="", encoding="utf-8") as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter="\t")
        for row in reader:
            job_data[row["nid"]] = row
    return job_data

def fetch_references(reference_type: str) -> Dict[str, str]:
    """
    Fetch references (career categories or themes) from the ReliefWeb API.

    Args:
        reference_type (str): Type of reference to fetch ('career-categories' or 'themes').

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

def fetch_jobs(job_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch job details from the ReliefWeb API.

    Args:
        job_ids (List[str]): List of job IDs to fetch.

    Returns:
        List[Dict[str, Any]]: List of job details.
    """
    url = "https://api.reliefweb.int/v1/jobs"
    all_jobs = []
    batch_size = 500
    for i in range(0, len(job_ids), batch_size):
        batch_ids = job_ids[i:i+batch_size]
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
            batch_jobs = response.json().get("data", [])
            all_jobs.extend(batch_jobs)
        except requests.RequestException as e:
            print(f"Error fetching jobs batch {i//batch_size + 1}: {str(e)}")
    return all_jobs

def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from a string.

    Args:
        text (str): Input text with HTML tags.

    Returns:
        str: Text with HTML tags removed.
    """
    return re.sub("<[^<]+?>", "", text)

def generate_prompt(job: Dict[str, Any], categories: Dict[str, str], themes: Dict[str, str]) -> str:
    """
    Generate an improved prompt for job classification and theme tagging.

    Args:
        job (Dict[str, Any]): Job details.
        categories (Dict[str, str]): Career categories.
        themes (Dict[str, str]): Themes.

    Returns:
        str: Generated prompt.
    """

    job_title = strip_html_tags(job['fields']['title'])
    job_body = strip_html_tags(job['fields']['body'])

    return textwrap.dedent(f"""
You are tasked with analyzing humanitarian job offers and classifying them according to specific career categories and focus areas. Your analysis will be conducted in three steps and should be presented in a clear XML format.

Please analyze the following job offer:

<job>
Title: {job_title}
Description:
{job_body}
</job>

Before you begin, understand these key distinctions:

Career Categories in Humanitarian Work represent functional roles and professional specializations within the humanitarian sector. These categories describe the types of jobs and skill sets needed to perform humanitarian work, regardless of the specific intervention area.

Humanitarian Focus Areas describe the thematic sectors or domains in which humanitarian interventions take place. These areas represent the specific fields of action or issues addressed by humanitarian work, distinct from the job functions themselves.

For your analysis, follow these steps:

1. Provide a concise summary capturing the essence and important context of the job offer.
2. Select the best matching category (only one) from the Career Categories list. Use the corresponding item number (e.g., A1, A2).
3. Select the most relevant Humanitarian Focus Areas (minimum 0, maximum 3). Use the corresponding item numbers (e.g., B1, B2) in a comma-separated list.

Career Categories:
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

<job_analysis>
  <summary>Concise summary of the job offer</summary>
  <career_category>Single item number (A1-A9)</career_category>
  <focus_areas>Comma-separated list of item numbers (B1-B20), 0-3 items</focus_areas>
</job_analysis>

Example output:

<job_analysis>
  <summary>Managing water and sanitation projects in refugee camps.</summary>
  <career_category>A7</career_category>
  <focus_areas>B13,B11,B14</focus_areas>
</job_analysis>

IMPORTANT:
1. Use ONLY the XML structure above.
2. Ensure valid XML with proper opening and closing tags.
3. If no focus areas apply, use: <focus_areas></focus_areas>
4. Do not include ANY text or explanations outside of the XML tags.
5. Your entire response must be valid XML.
    """).strip()

def extract_thinking_and_answer(
    response: Dict[str, Any],
    model_type: str,
    categories: Dict[str, str],
    themes: Dict[str, str]
) -> Tuple[str, str, str]:
    """
    Extract thinking, category, and themes from the model response.

    Args:
        response: The raw response from the model.
        model_type: The type of model used (e.g., "titan").
        categories: Dictionary of valid career categories.
        themes: Dictionary of valid themes.

    Returns:
        A tuple containing:
            - thinking: The model's reasoning for its classification.
            - category: The selected category (or "-" if none valid).
            - themes: A pipe-separated string of valid themes (or "-" if none valid).

    Raises:
        ValueError: If an unknown model type is provided.
    """
    try:
        if model_type != "titan":
            raise ValueError(f"Unknown model type: {model_type}")

        output_text = response["results"][0]["outputText"].strip()

        extracted_thinking = extract_thinking(output_text)
        extracted_category = extract_terms(output_text, 'career_category', categories, 1, 'A')
        extracted_themes = extract_terms(output_text, 'focus_areas', themes, 3, 'B')

        formatted_category = format_terms(extracted_category)
        formatted_themes = format_terms(extracted_themes)

        return extracted_thinking, formatted_category, formatted_themes

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
    parts = ['summary', 'category_reasoning', 'themes_reasoning']
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
    Extract and validate terms (category or themes) from the output text.

    Args:
        output_text: The full output text to search in.
        tag: The XML-like tag to look for (e.g., 'career_categories' or 'themes').
        term_dict: Dictionary of valid terms (categories or themes).
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

    # Regular expression to match the format 'A1)', 'B2)', etc.
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

def get_processed_job_ids(output_file: str) -> set:
    """
    Get the set of already processed job IDs.

    Args:
        output_file (str): Path to the output file.

    Returns:
        set: Set of processed job IDs.
    """
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r", newline="", encoding="utf-8") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter="\t")
            for row in reader:
                processed_ids.add(row["Job ID"])
    return processed_ids

def process_job(
    job: Dict[str, Any],
    job_data: Dict[str, Dict[str, str]],
    categories: Dict[str, str],
    themes: Dict[str, str],
    models: List[Tuple[str, Any, str, str]],
    job_index: int,
    total_jobs: int
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Process a single job using multiple AI models for category and theme prediction.

    Args:
        job (Dict[str, Any]): The job data to be processed.
        job_data (Dict[str, Dict[str, str]]): Additional metadata for the job.
        categories (Dict[str, str]): Dictionary of valid career categories.
        themes (Dict[str, str]): Dictionary of valid themes.
        models (List[Tuple[str, Any, str, str]]): List of models to use for prediction.
            Each tuple contains (model_name, query_function, model_type, region).
        job_index (int): The current job's index in the processing queue.
        total_jobs (int): Total number of jobs to be processed.

    Returns:
        Tuple[Dict[str, Any], List[str]]: A tuple containing:
            - job_result (Dict[str, Any]): Dictionary with job details and model predictions.
            - output_lines (List[str]): List of formatted strings for logging or display.
    """

    prompt = generate_prompt(job, categories, themes)

    # Extract actual classifications.
    actual_category = job["fields"]["career_categories"][0]["name"] if job["fields"]["career_categories"] else "Not specified"
    actual_themes = ",".join(sorted([theme["name"] for theme in job["fields"].get("theme", [])])) or "Not specified"

    job_result = {
        "Job ID": job["id"],
        "Job URL": job["fields"]["url"],
        "Job Title": job["fields"]["title"],
        "Posted": job_data[job["id"]]["posted"],
        "Editor": job_data[job["id"]]["editor"],
        "Trusted": job_data[job["id"]]["trusted"],
        "Reviewed": job_data[job["id"]]["reviewed"],
        "Job actual category": actual_category,
        "Job actual themes": actual_themes
    }

    output_lines = [
        f"Processing Job {job_index}/{total_jobs}: {job['fields']['title']} - Actual category: {actual_category}, Actual themes: {actual_themes}"
    ]

    # Process through each model.
    for model_name, query_func, model_type, region in models:
        model_start_time = time.time()
        response, input_tokens, output_tokens = query_func(prompt)
        model_end_time = time.time()
        response_time = model_end_time - model_start_time

        if response is None:
            thinking, llm_category, llm_themes = "-", "-", "-"
        else:
            thinking, llm_category, llm_themes = extract_thinking_and_answer(response, model_type, categories, themes)

        # Store model results.
        job_result[f"{model_name} - Category"] = llm_category
        job_result[f"{model_name} - Themes"] = llm_themes
        job_result[f"{model_name} - Reason"] = thinking
        job_result[f"{model_name} - Region"] = region
        job_result[f"{model_name} - Time"] = f"{response_time:.2f}"
        job_result[f"{model_name} - Input Tokens"] = input_tokens
        job_result[f"{model_name} - Output Tokens"] = output_tokens

        output_lines.append(f"{model_name} - Category: {llm_category}, Themes: {llm_themes}, Time: {response_time:.2f} seconds, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")

    return job_result, output_lines

def process_output_queue(queue: Queue, total: int, start_time: float) -> None:
    """
    Process the output queue in order and print results to console.

    Args:
        queue (Queue): Queue containing job results and output lines.
        total (int): Total number of jobs to be processed.
        start_time (float): Start time of the job processing.
    """
    next_index = 1
    buffer = {}

    while next_index <= total:
        if next_index in buffer:
            result, lines, index = buffer.pop(next_index)
            print_job_result(result, lines, index, total, start_time)
            next_index += 1
        else:
            try:
                result, lines, index = queue.get(timeout=1)
                if index == next_index:
                    print_job_result(result, lines, index, total, start_time)
                    next_index += 1
                else:
                    buffer[index] = (result, lines, index)
            except Empty:
                continue

def print_job_result(
    result: Dict[str, Any],
    lines: List[str],
    index: int,
    total: int,
    start_time: float) -> None:
    """
    Print the result of a single job processing.

    Args:
        result (Dict[str, Any]): Dictionary containing job processing results.
        lines (List[str]): List of output lines to be printed.
        index (int): Index of the current job.
        total (int): Total number of jobs to be processed.
        start_time (float): Start time of the job processing.
    """
    for line in lines:
        print(line)

    elapsed = time.time() - start_time
    avg_time = elapsed / index
    remaining = avg_time * (total - index)

    print(f"Job {index}/{total} completed")
    print(f"Elapsed: {elapsed:.2f}s | Remaining: {remaining:.2f}s")
    print("=" * 50)

def main() -> None:
    """
    Main function to process jobs and classify them using parallel processing.
    """
    output_file = "job_classification_results_titan_legacy.tsv"
    processed_job_ids = get_processed_job_ids(output_file)
    job_data = load_job_data()

    # Fetch reference data.
    categories = fetch_references("career-categories")
    themes = fetch_references("themes")

    if not categories or not themes:
        print("Failed to fetch career categories or themes. Exiting.")
        return

    jobs_to_process = [job_id for job_id in job_data if job_id not in processed_job_ids]
    jobs = fetch_jobs(jobs_to_process)

    if not jobs:
        print("Failed to fetch jobs. Exiting.")
        return

    models = [
        ("AWS Bedrock Titan", lambda p: query_bedrock_titan(bedrock_east, p), "titan", "us-east-1"),
    ]

    fieldnames = ["Job ID", "Job URL", "Job Title", "Posted", "Editor", "Trusted", "Reviewed", "Job actual category", "Job actual themes"]
    for model in models:
        fieldnames.extend([
            f"{model[0]} - Category",
            f"{model[0]} - Themes",
            f"{model[0]} - Reason",
            f"{model[0]} - Region",
            f"{model[0]} - Time",
            f"{model[0]} - Input Tokens",
            f"{model[0]} - Output Tokens"
        ])

    file_exists = os.path.exists(output_file)
    with open(output_file, "a", newline="", encoding="utf-8") as tsvfile:
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            writer.writeheader()

        total_jobs = len(jobs)
        start_time = time.time()

        output_queue = Queue()
        output_thread = threading.Thread(target=process_output_queue, args=(output_queue, total_jobs, start_time))
        output_thread.start()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            process_job_partial = partial(
                process_job,
                job_data=job_data,
                categories=categories,
                themes=themes,
                models=models,
                total_jobs=total_jobs
            )
            future_to_job = {executor.submit(process_job_partial, job, job_index=i+1): (job, i+1) for i, job in enumerate(jobs)}

            for future in concurrent.futures.as_completed(future_to_job):
                job, job_index = future_to_job[future]
                try:
                    job_result, output_lines = future.result()
                    writer.writerow(job_result)
                    tsvfile.flush()
                    output_queue.put((job_result, output_lines, job_index))
                except Exception as exc:
                    print(f"Job {job['id']} generated an exception: {exc}")

        output_queue.put(None)  # Signal the output thread to finish
        output_thread.join()

    print(f"Results have been written to {output_file}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
