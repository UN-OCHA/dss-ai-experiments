"""
ReliefWeb Job Classification Automation

This Python script automates the classification of job listings from ReliefWeb into predefined career categories using AWS Bedrock"s Titan AI model. The script processes a balanced list of 1000 jobs for each career category, fetches job details, generates prompts for the AI model, and analyzes the results. It outputs the classification accuracy and writes detailed results to a TSV file.
"""

# pylint: disable=broad-exception-caught,too-many-statements

import os
import json
import time
import sys
import re
import csv
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import boto3
import requests

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

# Load configuration
config = load_config()

# AWS Bedrock configuration
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=config["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=config["AWS_SECRET_ACCESS_KEY"]
)

APPNAME = "rw-job-tagging-experiments"

def fetch_career_categories() -> Dict[str, Dict[str, str]]:
    """
    Fetch career categories from ReliefWeb API.

    Returns:
        Dict[str, Dict[str, str]]: Dictionary of career categories.
    """
    url = "https://api.reliefweb.int/v1/references/career-categories"
    params = {"appname": APPNAME}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            item["fields"]["name"]: {
                "id": item["id"],
                "description": item["fields"].get("description", "No description available")
            } for item in data["data"]
        }
    except requests.RequestException as e:
        print(f"Error fetching career categories: {str(e)}")
        return {}

def fetch_jobs_for_category(category_id: str) -> List[Dict[str, Any]]:
    """
    Fetch jobs for a specific category from ReliefWeb API.

    Args:
        category_id (str): ID of the career category.

    Returns:
        List[Dict[str, Any]]: List of job data dictionaries.
    """
    url = "https://api.reliefweb.int/v1/jobs"
    payload = {
        "appname": APPNAME,
        "profile": "full",
        "limit": 1000,
        "sort": ["id:desc"],
        "preset": "analysis",
        "filter": {
            "field": "career_categories.id",
            "value": category_id
        },
        "fields": {
            "include": ["id", "url", "title", "body", "career_categories"]
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data["data"]
    except requests.RequestException as e:
        print(f"Error fetching jobs for category {category_id}: {str(e)}")
        return []

def strip_html_tags(text: str) -> str:
    """
    Remove HTML tags from a string.

    Args:
        text (str): Input text with HTML tags.

    Returns:
        str: Text with HTML tags removed.
    """
    return re.sub("<[^<]+?>", "", text)

def generate_prompt(job: Dict[str, Any], categories: Dict[str, Dict[str, str]]) -> str:
    """
    Generate a prompt for the AI model based on job details and categories.

    Args:
        job (Dict[str, Any]): Job data dictionary.
        categories (Dict[str, Dict[str, str]]): Dictionary of career categories.

    Returns:
        str: Generated prompt for the AI model.
    """
    categories_text = "\n".join([f"{name}: {info["description"]}" for name, info in categories.items()])

    job_title = strip_html_tags(job['fields']['title'])
    job_body = strip_html_tags(job['fields']['body'])

    return f"""<instructions>
        You are an AI expert in job classification, specializing in categorizing positions within the humanitarian and development sector. Your task is to analyze the provided job offer and assign it to the most appropriate ReliefWeb career category.

        Please follow these steps:
        1. Carefully review the job offer details.
        2. Consider ONLY the ReliefWeb career categories provided in the <categories> section.
        3. Determine the best-fitting category based on the job description, required skills, and responsibilities.

        Format your response as follows:
        1. Explain your reasoning between <thinking> and </thinking> tags.
        2. State only the chosen category name between <answer> and </answer> tags.

        Critical rules:
        - You MUST select ONLY from the given list of ReliefWeb career categories in the <categories> section.
        - Do NOT create or use any categories that are not explicitly listed.
        - If the job doesn't seem to fit perfectly into any category, choose the closest match from the provided list.
        - Ensure your classification is accurate and well-justified based on the job description.
        - Do not include any additional information or categories outside of what is requested.

        Remember: Always refer back to the provided categories before making your final decision.
        </instructions>

        <categories>
        {categories_text}
        </categories>

        <job>
        {job_title}

        {job_body}
        </job>

        """


def query_bedrock_titan(prompt: str) -> Tuple[Dict[str, Any], int, int]:
    """
    Query AWS Bedrock Titan model with the given prompt.

    Args:
        prompt (str): Input prompt for the model.

    Returns:
        Tuple[Dict[str, Any], int, int]: Model response, input token count, and output token count.
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
        response = bedrock_client.invoke_model(
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

def extract_thinking_and_answer(response: Dict[str, Any], categories: Dict[str, Dict[str, str]]) -> Tuple[str, str]:
    """
    Extract thinking and answer from the model response.

    Args:
        response (Dict[str, Any]): Model response dictionary.
        categories (Dict[str, Dict[str, str]]): Dictionary of career categories.

    Returns:
        Tuple[str, str]: Extracted thinking and answer.
    """
    try:
        output_text = response["results"][0]["outputText"].strip()
        thinking = re.search(r"<thinking>(.*?)</thinking>", output_text, re.DOTALL)
        thinking = thinking.group(1).strip() if thinking else "-"
        answer = re.search(r"<answer>(.*?)</answer>", output_text, re.DOTALL)
        answer = answer.group(1).strip() if answer else "-"
        if answer == "-":
            for category in categories:
                if category.lower() in output_text.lower():
                    answer = category
                    break
        return thinking, answer
    except Exception as e:
        print(f"Error extracting thinking and answer: {str(e)}")
        return "-", "-"

def main():
    """
    Main function to run the job classification process.
    """
    categories = fetch_career_categories()
    if not categories:
        print("Failed to fetch career categories. Exiting.")
        return

    output_file = "job_classification_results.tsv"
    fieldnames = ["Job ID", "Job URL", "Job Title", "Actual Category", "Predicted Category", "Reasoning", "Response Time", "Input Tokens", "Output Tokens"]

    file_exists = os.path.isfile(output_file)
    start_time = time.time()
    total_jobs_processed = 0
    total_correct_predictions = 0
    category_stats = defaultdict(lambda: {"processed": 0, "correct": 0})

    with open(output_file, "a", newline="", encoding="utf-8") as tsvfile:
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            writer.writeheader()

        for category_name, category_info in categories.items():
            print(f"Fetching jobs for category: {category_name}")
            jobs = fetch_jobs_for_category(category_info["id"])
            print(f"Fetched {len(jobs)} jobs for category: {category_name}")

            for job_index, job in enumerate(jobs, 1):
                total_jobs_processed += 1
                category_stats[category_name]["processed"] += 1
                job_start_time = time.time()

                prompt = generate_prompt(job, categories)
                actual_category = job["fields"]["career_categories"][0]["name"]
                print(f"Processing Job {job_index}/{len(jobs)} in category {category_name}: {job["fields"]["title"]}")

                job_result = {
                    "Job ID": job["id"],
                    "Job URL": job["fields"]["url"],
                    "Job Title": job["fields"]["title"],
                    "Actual Category": actual_category
                }

                model_start_time = time.time()
                response, input_tokens, output_tokens = query_bedrock_titan(prompt)
                model_end_time = time.time()
                response_time = model_end_time - model_start_time

                if response is None:
                    print("Error occurred during query")
                    thinking, llm_category = "-", "-"
                else:
                    thinking, llm_category = extract_thinking_and_answer(response, categories)

                job_result["Predicted Category"] = llm_category
                job_result["Reasoning"] = thinking
                job_result["Response Time"] = f"{response_time:.2f}"
                job_result["Input Tokens"] = input_tokens
                job_result["Output Tokens"] = output_tokens

                writer.writerow(job_result)
                tsvfile.flush()

                if llm_category == actual_category:
                    total_correct_predictions += 1
                    category_stats[category_name]["correct"] += 1

                job_end_time = time.time()
                job_elapsed_time = job_end_time - job_start_time
                total_elapsed_time = job_end_time - start_time
                avg_time_per_job = total_elapsed_time / total_jobs_processed
                estimated_time_remaining = avg_time_per_job * (len(jobs) - job_index)
                category_accuracy = (category_stats[category_name]["correct"] / category_stats[category_name]["processed"]) * 100
                overall_accuracy = (total_correct_predictions / total_jobs_processed) * 100

                print(f"Predicted Category: {llm_category}, Time: {response_time:.2f} seconds")
                print(f"Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")
                print(f"Job completed in {job_elapsed_time:.2f} seconds")
                print(f"Estimated time remaining for category: {estimated_time_remaining:.2f} seconds")
                print(f"Current category accuracy: {category_accuracy:.2f}%")
                print(f"Overall accuracy: {overall_accuracy:.2f}%")
                print(f"Total jobs processed: {total_jobs_processed}")
                print(f"Total elapsed time: {total_elapsed_time:.2f} seconds")
                print("=" * 50)

    print(f"All results have been written to {output_file}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print(f"Total jobs processed: {total_jobs_processed}")
    print(f"Final overall accuracy: {(total_correct_predictions / total_jobs_processed) * 100:.2f}%")
    print("\nAccuracy per career category:")
    for category, stats in category_stats.items():
        accuracy = (stats["correct"] / stats["processed"]) * 100 if stats["processed"] > 0 else 0
        print(f"{category}: {accuracy:.2f}% ({stats["correct"]}/{stats["processed"]})")

if __name__ == "__main__":
    main()
