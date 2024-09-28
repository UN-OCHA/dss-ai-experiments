# Date Extraction and Categorization Script

This Python script extracts and categorizes dates from natural language questions related to disaster events and relief efforts. It uses the `ctparse` library to identify and parse dates within the text.

## Features

- Extracts dates from various question formats
- Categorizes dates as single dates or date ranges
- Handles relative date expressions (e.g., "last month", "next week")
- Processes a predefined list of example questions

## Requirements

- Python 3.x
- ctparse

## Installation

1. Ensure you have Python 3.x installed on your system.
2. Install the required library:

```bash
pip install ctparse
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
   - Uses ctparse to extract date information from the question
   - Categorizes the date as single or range
   - Determines if it's a specific date, relative date, or date range
4. Processes each question in the list and prints the extracted date information.

## Output

For each question, the script outputs:
- The original question
- The extracted date or date range, if found
- The category of the date (e.g., "specific date", "past month", "next week")

If no date is found, it prints a message indicating that no date could be extracted.

## Customization

You can modify the `questions` list to include your own questions or input from an external source. Additionally, you can adjust the reference date in the script to suit your specific needs.

## Note

The script uses a fixed reference date (September 3, 2024) for demonstration purposes. In a real-world application, you might want to use the current date as the reference date.
