# ReliefWeb Job Classification Script

This Python script automates the classification of job listings from ReliefWeb into predefined career categories using AWS Bedrock's Titan AI model.

## Features

- Fetches career categories from ReliefWeb API
- Retrieves 1000 job listings for each career category
- Classifies jobs using AWS Bedrock's Titan AI model
- Outputs results to a TSV file
- Provides real-time progress and accuracy statistics

## Prerequisites

- Python 3.x
- AWS account with Bedrock access
- ReliefWeb API access

## Configuration

Create a `config.json` file in the same directory as the script with the following structure:

```json
{
  "AWS_ACCESS_KEY_ID": "your_aws_access_key",
  "AWS_SECRET_ACCESS_KEY": "your_aws_secret_key"
}
```

## Installation

1. Clone the repository
2. Install required packages:

```bash
pip install requests boto3
```

## Usage

Run the script with:

```bash
python main.py
```

## Output

The script generates a TSV file named `job_classification_results.tsv` with the following columns:

- Job ID
- Job URL
- Job Title
- Actual Category
- Predicted Category
- Reasoning
- Response Time
- Input Tokens
- Output Tokens

## Key Functions

- `load_config()`: Loads AWS credentials from config file
- `fetch_career_categories()`: Retrieves career categories from ReliefWeb API
- `fetch_jobs_for_category()`: Fetches 1000 jobs for each career category
- `generate_prompt()`: Creates AI prompt for job classification
- `query_bedrock_titan()`: Sends classification request to AWS Bedrock
- `extract_thinking_and_answer()`: Parses AI response

## Notes

- The script processes a balanced list of 1000 jobs for each career category
- Real-time statistics include category-specific and overall accuracy
- Estimated time remaining is calculated and displayed during execution
