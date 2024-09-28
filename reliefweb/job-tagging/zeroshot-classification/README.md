# Job Category Classifier

This Python script uses the Hugging Face Transformers library to perform zero-shot classification on job descriptions from ReliefWeb API. It categorizes jobs into predefined career categories and evaluates the accuracy of the classification.

## Notes

- This script used AI generated descriptions based of the descriptions of the original ReliefWeb career categories, which are *supposedly* more suitable to for zeroshort classification.

## Features

- Fetches job data from ReliefWeb API
- Uses zero-shot classification to categorize jobs
- Calculates and displays accuracy metrics
- Processes multiple job categories

## Requirements

- Python 3.6+
- requests
- transformers

## Installation

1. Clone this repository or download the script.
2. Install the required packages:

```bash
pip install requests transformers
```

## Usage

Run the script with:

```bash
python main.py
```

## How it works

1. The script defines a mapping of career categories and their descriptions.
2. It fetches job data for each category from the ReliefWeb API.
3. A zero-shot classification model is used to predict the category for each job based on its description.
4. The script compares the predicted category with the actual category and calculates accuracy.
5. Results are printed for each job, including job details, predicted category, confidence score, and classification time.
6. Overall accuracy is displayed at the end.

## Configuration

- `APPNAME`: Set this to your ReliefWeb API application name.
- `category_mapping`: Modify this dictionary to change or add career categories.
- The script currently fetches 10 jobs per category. Adjust the `limit` parameter in the `fetch_jobs_for_category` function to change this.

## Model

The script uses the "MoritzLaurer/bge-m3-zeroshot-v2.0-c" model for zero-shot classification. You can change this by modifying the `model` parameter in the `pipeline` function call.

## Output

The script outputs information for each processed job, including:
- Job ID and title
- Actual and predicted categories
- Confidence score
- Classification time
- Accuracy for the current category
- Overall accuracy

A summary of total jobs processed and overall accuracy is displayed at the end.

