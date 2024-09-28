"""
This script extracts and categorizes dates from natural language questions
related to disaster events and relief efforts using the ctparse library.
"""

from ctparse import ctparse
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Union

# Example questions containing date information
QUESTIONS = [
    "When was the last earthquake reported?",
    "What are the relief efforts for the flood last month?",
    "How many people were affected by the hurricane in 2021?",
    "Can you provide updates on the drought situation from last week?",
    "What are the current relief operations for the cyclone?",
    "When did the tsunami hit the coast?",
    "What was the impact of the landslide in January?",
    "How has the situation changed since the last report on August 15th?",
    "Are there any updates on the relief efforts from two days ago?",
    "What measures are being taken for the upcoming storm next week?",
    "What were the consequences of the earthquake on March 3rd?",
    "Can you provide information on the relief efforts for the disaster last year?",
    "When is the next assessment of the flood damage scheduled?",
    "What actions were taken immediately after the earthquake last night?",
    "How many people were displaced by the flood in June?",
    "What are the predictions for the hurricane season this year?",
    "Have there been any developments since the last update on September 1st?",
    "What is the timeline for the recovery efforts after the typhoon?",
    "What was the response to the famine in 2019?",
    "When did the relief operations for the earthquake begin?",
    "Are there any reports on the flood from earlier this month?",
    "What is the status of the aid distribution for the cyclone last week?",
    "When was the last assessment of the drought situation?",
    "What are the plans for rebuilding after the earthquake in February?",
    "What was the magnitude of the earthquake that occurred yesterday?",
    "How many casualties were reported from the landslide on July 10th?",
    "What are the ongoing challenges since the flood last month?",
    "Have there been any updates on the relief efforts since last Friday?",
    "What are the expected impacts of the upcoming storm next weekend?",
    "What was the timeline of events during the flood in April?",
    "When is the next meeting scheduled to discuss the relief efforts?",
    "What were the immediate actions taken after the earthquake on May 5th?",
    "How has the situation evolved since the flood two weeks ago?",
    "What is the status of the emergency response for the hurricane this month?",
    "When did the relief efforts for the drought start?",
    "What are the plans for the upcoming cyclone season?",
    "What was the immediate response to the earthquake on December 25th?",
    "How many people were evacuated during the flood last year?",
    "What are the lessons learned from the hurricane in 2020?",
    "When was the last update on the relief efforts for the landslide?",
    "What are the projected impacts of the storm next Thursday?",
    "What was the sequence of events during the earthquake in March?",
    "Are there any updates on the aid distribution from last Tuesday?",
    "What is the current status of the recovery efforts after the cyclone?",
    "How many people were affected by the drought in 2018?",
    "What are the ongoing relief efforts for the flood last weekend?",
    "When did the emergency response for the earthquake commence?",
    "What are the future plans for disaster preparedness this year?",
    "What was the impact of the typhoon on September 10th?"
]

def extract_and_categorize_dates(
    question: str,
    ref_date: datetime
) -> Optional[Tuple[str, str, Union[datetime, Tuple[datetime, datetime]]]]:
    """
    Extract and categorize dates from a given question.

    Args:
        question (str): The input question containing date information.
        ref_date (datetime): Reference date for relative date parsing.

    Returns:
        Optional[Tuple]: A tuple containing category, description, and date(s) if found,
                         None otherwise.
    """
    result = ctparse(question, ts=ref_date)

    if result and result.resolution:
        extracted_date = result.resolution

        if isinstance(extracted_date, datetime):
            return ('single', 'specific date', extracted_date)
        if hasattr(extracted_date, 'start') and hasattr(extracted_date, 'end'):
            return ('range', 'date range', extracted_date.start, extracted_date.end)
        if hasattr(extracted_date, 'start'):
            # Handle relative dates
            if 'last' in question.lower() or 'past' in question.lower():
                if 'year' in question.lower():
                    return ('range', 'past year', extracted_date.start - timedelta(days=365), extracted_date.start)
                if 'month' in question.lower():
                    return ('range', 'past month', extracted_date.start - timedelta(days=30), extracted_date.start)
                if 'week' in question.lower():
                    return ('range', 'past week', extracted_date.start - timedelta(days=7), extracted_date.start)
            elif 'next' in question.lower() or 'upcoming' in question.lower():
                if 'year' in question.lower():
                    return ('range', 'next year', extracted_date.start, extracted_date.start + timedelta(days=365))
                if 'month' in question.lower():
                    return ('range', 'next month', extracted_date.start, extracted_date.start + timedelta(days=30))
                if 'week' in question.lower():
                    return ('range', 'next week', extracted_date.start, extracted_date.start + timedelta(days=7))
            return ('single', 'relative date', extracted_date.start)

    return None

def process_questions(questions: List[str], reference_date: datetime) -> None:
    """
    Process a list of questions and extract date information from each.

    Args:
        questions (List[str]): List of questions to process.
        reference_date (datetime): Reference date for relative date parsing.
    """
    for question in questions:
        print(f"Question: {question}")
        result = extract_and_categorize_dates(question, reference_date)
        if result:
            category, description, *dates = result
            if category == 'single':
                print(f"Extracted Date: {description} - {dates[0].date()}")
            else:
                print(f"Extracted Date Range: {description} - from {dates[0].date()} to {dates[1].date()}")
        else:
            print("No date could be extracted from the question.")
        print("-" * 50)

def main():
    """Main function to run the script."""
    reference_date = datetime(2024, 9, 3)
    process_questions(QUESTIONS, reference_date)

if __name__ == "__main__":
    main()
