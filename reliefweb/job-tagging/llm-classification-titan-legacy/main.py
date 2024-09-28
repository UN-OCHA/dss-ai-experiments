"""
This script analyzes the tagging accuracy for ReliefWeb jobs posted before 2021,
comparing automated tagging using AWS Bedrock Titan Premier against historical
manual tagging. It evaluates the potential of automated job tagging systems by
assessing AI-generated tags against original human-assigned tags for legacy job
data, providing insights for future implementation of automated tagging processes.
"""

# pylint: disable=broad-exception-caught,too-many-statements

import os
import json
import time
import sys
import re
import csv
from typing import Dict, List, Tuple, Any

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

def fetch_career_categories() -> Dict[str, str]:
    """
    Fetch career categories from the ReliefWeb API.

    Returns:
        Dict[str, str]: Dictionary of career categories.
    """
    url = "https://api.reliefweb.int/v1/references/career-categories"
    params = {"appname": APPNAME}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {item["fields"]["name"]: item["fields"]["description"] for item in data["data"]}
    except requests.RequestException as e:
        print(f"Error fetching career categories: {str(e)}")
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
                "include": ["id", "url", "title", "body", "career_categories"]
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

def generate_prompt(job: Dict[str, Any], categories: Dict[str, str]) -> str:
    """
    Generate a prompt for job classification.

    Args:
        job (Dict[str, Any]): Job details.
        categories (Dict[str, str]): Career categories.

    Returns:
        str: Generated prompt.
    """
    categories_text = "\n".join([f"{name}: {desc}" for name, desc in categories.items()])
    instructions = """You are an expert in classifying job offers. Your task is to classify the given job offer into one of the ReliefWeb career categories defined between <categories> and </categories>.

Even if no category perfectly matches the job description, you MUST choose the BEST MATCHING category from the provided list. Do not create new categories or leave the classification empty.

Put your reasoning to select the category between <thinking> and </thinking> and write your answer containing the category name only between <answer> and </answer>. Do not add anything else.."""

    job_title = strip_html_tags(job['fields']['title'])
    job_body = strip_html_tags(job['fields']['body'])

    return f"""
        <instructions>
        {instructions}
        </instructions>

        <job>
        {job_title}
        {job_body}
        </job>

        <categories>
        {categories_text}
        </categories>
        """

def query_bedrock_titan(client: Any, prompt: str) -> Tuple[Dict[str, Any], int, int]:
    """
    Query AWS Bedrock Titan model.

    Args:
        client (Any): Bedrock client.
        prompt (str): Input prompt.

    Returns:
        Tuple[Dict[str, Any], int, int]: Response, input tokens, and output tokens.
    """
    try:
        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 1024,
                "stopSequences": [],
                "temperature": 0.0,
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

def extract_thinking_and_answer(response: Dict[str, Any], model_type: str, categories: Dict[str, str]) -> Tuple[str, str]:
    """
    Extract thinking and answer from model response.

    Args:
        response (Dict[str, Any]): Model response.
        model_type (str): Type of the model.
        categories (Dict[str, str]): Career categories.

    Returns:
        Tuple[str, str]: Thinking and answer.
    """
    try:
        if model_type == "titan":
            output_text = response["results"][0]["outputText"].strip()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        thinking = "-"
        if "<thinking>" in output_text and "</thinking>" in output_text:
            thinking = output_text.split("<thinking>")[1].split("</thinking>")[0].strip()

        answer = "-"
        if "<answer>" in output_text and "</answer>" in output_text:
            answer = output_text.split("<answer>")[1].split("</answer>")[0].strip()

        if answer == "-":
            for category in categories:
                if category.lower() in output_text.lower():
                    answer = category
                    break

        return thinking, answer
    except Exception as e:
        print(f"Error extracting thinking and answer for {model_type}: {str(e)}")
        return "-", "-"

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

def main() -> None:
    """
    Main function to process jobs and classify them.
    """
    output_file = "job_classification_results_titan_legacy.tsv"
    processed_job_ids = get_processed_job_ids(output_file)
    job_data = load_job_data()
    categories = fetch_career_categories()

    if not categories:
        print("Failed to fetch career categories. Exiting.")
        return

    jobs_to_process = [job_id for job_id in job_data if job_id not in processed_job_ids]
    jobs = fetch_jobs(jobs_to_process)

    if not jobs:
        print("Failed to fetch jobs. Exiting.")
        return

    models = [
        ("AWS Bedrock Titan", lambda p: query_bedrock_titan(bedrock_east, p), "titan", "us-east-1"),
    ]

    fieldnames = ["Job ID", "Job URL", "Job Title", "Posted", "Editor", "Trusted", "Reviewed", "Job actual category"]
    for model in models:
        fieldnames.extend([
            f"{model[0]} - Category",
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

        for job_index, job in enumerate(jobs, 1):
            job_start_time = time.time()
            prompt = generate_prompt(job, categories)
            actual_category = job["fields"]["career_categories"][0]["name"] if job["fields"]["career_categories"] else "Not specified"
            print(f"Processing Job {job_index}/{total_jobs}: {job['fields']['title']} - Actual category: {actual_category}")

            job_result = {
                "Job ID": job["id"],
                "Job URL": job["fields"]["url"],
                "Job Title": job["fields"]["title"],
                "Posted": job_data[job["id"]]["posted"],
                "Editor": job_data[job["id"]]["editor"],
                "Trusted": job_data[job["id"]]["trusted"],
                "Reviewed": job_data[job["id"]]["reviewed"],
                "Job actual category": actual_category
            }

            for model_name, query_func, model_type, region in models:
                model_start_time = time.time()
                response, input_tokens, output_tokens = query_func(prompt)
                model_end_time = time.time()
                response_time = model_end_time - model_start_time

                if response is None:
                    print(f"{model_name}: Error occurred during query")
                    thinking, llm_category = "-", "-"
                else:
                    thinking, llm_category = extract_thinking_and_answer(response, model_type, categories)

                job_result[f"{model_name} - Category"] = llm_category
                job_result[f"{model_name} - Reason"] = thinking
                job_result[f"{model_name} - Region"] = region
                job_result[f"{model_name} - Time"] = f"{response_time:.2f}"
                job_result[f"{model_name} - Input Tokens"] = input_tokens
                job_result[f"{model_name} - Output Tokens"] = output_tokens

                print(f"{model_name} - Category: {llm_category}, Time: {response_time:.2f} seconds, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")

            writer.writerow(job_result)
            tsvfile.flush()  # Ensure the data is written to the file

            job_end_time = time.time()
            job_elapsed_time = job_end_time - job_start_time
            total_elapsed_time = job_end_time - start_time
            avg_time_per_job = total_elapsed_time / job_index
            estimated_time_remaining = avg_time_per_job * (total_jobs - job_index)

            print(f"Job {job_index}/{total_jobs} completed in {job_elapsed_time:.2f} seconds")
            print(f"Total elapsed time: {total_elapsed_time:.2f} seconds")
            print(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds")
            print("=" * 50)

    print(f"Results have been written to {output_file}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
