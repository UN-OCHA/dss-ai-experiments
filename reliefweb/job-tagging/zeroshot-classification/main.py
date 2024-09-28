"""
This script classifies job listings from ReliefWeb API using zero-shot classification.
It fetches job data, predicts categories, and calculates accuracy metrics.
"""

import time
import requests
from transformers import pipeline

# Application name for ReliefWeb API.
APPNAME = "rw-ai-experiments-zeroshot-classification"

# Mapping of career categories to their descriptions.
CATEGORY_MAPPING = {
    "Administration/Finance": "This job involves financial management, budgeting, accounting, or general office administration.",
    "Donor Relations/Grants Management": "This role focuses on fundraising, proposal writing, managing donor partnerships, or overseeing grant funds.",
    "Human Resources": "This position is about managing personnel, including recruitment, training, and employee development.",
    "Information and Communications Technology": "This job deals with managing IT infrastructure, software development, or network systems.",
    "Information Management": "This role involves collecting, analyzing, and sharing data about crises or disasters, including GIS and mapping.",
    "Logistics/Procurement": "This position handles supply chain management, procurement, inventory, or transportation of goods and resources.",
    "Advocacy/Communications": "This job focuses on public relations, media engagement, or promoting organizational agendas through various communication channels.",
    "Monitoring and Evaluation": "This role assesses project quality and progress, designs evaluation tools, or reports on program effectiveness.",
    "Program/Project Management": "This position oversees the entire project lifecycle, from planning and implementation to reporting and quality assurance."
}

def fetch_jobs_for_category(category_name):
    """
    Fetch job listings for a specific category from ReliefWeb API.

    Args:
        category_name (str): The name of the category to fetch jobs for.

    Returns:
        list: A list of job data dictionaries.
    """
    url = "https://api.reliefweb.int/v1/jobs"
    payload = {
        "appname": APPNAME,
        "profile": "full",
        "limit": 10,
        "sort": ["date.created:desc"],
        "fields": {
            "include": ["id", "url", "title", "body-html", "career_categories"]
        },
        "filter": {
            "field": "career_categories.name",
            "value": category_name
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        return data['data']
    except requests.RequestException as e:
        print(f"Error fetching jobs for category {category_name}: {str(e)}")
        return []

def classify_job(classifier, job_description):
    """
    Classify a job description using zero-shot classification.

    Args:
        classifier: The zero-shot classification pipeline.
        job_description (str): The job description to classify.

    Returns:
        tuple: Predicted category, confidence score, and classification time.
    """
    hypothesis_template = "This job is about {}"
    classes_verbalized = list(CATEGORY_MAPPING.values())
    start_time = time.time()
    output = classifier(job_description, classes_verbalized, hypothesis_template=hypothesis_template, multi_label=False)
    classification_time = time.time() - start_time
    predicted_category = list(CATEGORY_MAPPING.keys())[classes_verbalized.index(output['labels'][0])]
    confidence_score = output['scores'][0]
    return predicted_category, confidence_score, classification_time

def main():
    """
    Main function to run the job classification process.
    """
    zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/bge-m3-zeroshot-v2.0-c")

    total_jobs = 0
    total_correct = 0

    for category, _ in CATEGORY_MAPPING.items():
        jobs = fetch_jobs_for_category(category)
        category_correct = 0

        for job in jobs:
            job_id = job['id']
            job_title = job['fields']['title']
            job_description = job['fields']['body-html']
            actual_category = job['fields']['career_categories'][0]['name']

            predicted_category, confidence_score, classification_time = classify_job(zeroshot_classifier, job_description)

            is_correct = predicted_category == actual_category
            if is_correct:
                category_correct += 1
                total_correct += 1

            total_jobs += 1

            # Print results for each job.
            print(f"Job ID: {job_id}")
            print(f"Job Title: {job_title}")
            print(f"Actual Category: {actual_category}")
            print(f"Predicted Category: {predicted_category}")
            print(f"Confidence Score: {confidence_score:.2f}")
            print(f"Classification Time: {classification_time:.2f} seconds")
            print(f"Accuracy for {category}: {category_correct / len(jobs) * 100:.2f}%")
            print(f"Overall Accuracy: {total_correct / total_jobs * 100:.2f}%")
            print("---")

    # Print summary statistics.
    print("\nSummary:")
    print(f"Total Jobs Processed: {total_jobs}")
    print(f"Overall Accuracy: {total_correct / total_jobs * 100:.2f}%")

if __name__ == "__main__":
    main()
