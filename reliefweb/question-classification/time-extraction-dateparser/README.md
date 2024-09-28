# Date Extraction and Categorization Script

This Python script extracts and categorizes dates from natural language questions related to disaster events and relief efforts. It uses the `dateparser` library to identify dates within the text and categorizes them based on context.

## Features

- Extracts dates from various question formats
- Categorizes dates as single dates or date ranges
- Handles relative date expressions (e.g., "last month", "next week")
- Processes a predefined list of example questions

## Requirements

- Python 3.x
- dateparser
- datetime
- dateutil

## Installation

1. Ensure you have Python 3.x installed on your system.
2. Install the required libraries:

```bash
pip install dateparser python-dateutil
```

## Usage

1. Run the script:

```bash
python main.py
```

## How it works

The script performs the following steps:

1. Defines a list of example questions containing date information.
2. Sets a reference date for relative date parsing.
3. Defines a function `extract_and_categorize_dates` that:
   - Searches for dates in the question
   - Categorizes the date as past, future, or specific
   - Determines if it's a single date or a date range
4. Processes each question in the list and prints the extracted date information.

## Output

For each question, the script outputs:
- The original question
- The extracted date or date range, if found
- The category of the date (e.g., "past date", "next week")

If no date is found, it prints a message indicating that no date could be extracted.

## Customization

You can modify the `questions` list to include your own questions or input from an external source. Additionally, you can adjust the date categorization logic in the `extract_and_categorize_dates` function to suit your specific needs.
