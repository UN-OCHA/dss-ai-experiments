"""
ReliefWeb Theme Classifier using TinyBERT model.

This script classifies humanitarian questions into relevant ReliefWeb themes
using a pre-trained TinyBERT model from Hugging Face Transformers.
"""

import os
import time
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer.
MODEL_NAME = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
MODEL_DIR = './models'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)

# ReliefWeb themes.
THEMES = [
    "Agriculture", "Climate Change and Environment", "Contributions", "Coordination",
    "Disaster Management", "Education", "Food and Nutrition", "Gender", "Health",
    "HIV/AIDS", "Humanitarian Financing", "Logistics and Telecommunications",
    "Mine Action", "Peace and Security", "Protection and Human Rights",
    "Recovery and Reconstruction", "Shelter and Non-Food Items", "Water Sanitation Hygiene"
]

def find_relevant_themes(question: str, top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Find the most relevant themes for a given question.

    Args:
        question (str): The input question.
        top_k (int): Number of top themes to return.

    Returns:
        List[Tuple[str, float]]: List of tuples containing theme and its score.
    """
    MODEL.eval()
    with torch.no_grad():
        inputs = TOKENIZER([question] * len(THEMES), THEMES, padding=True, truncation=True, return_tensors="pt")
        scores = MODEL(**inputs).logits.squeeze()

    top_indices = torch.argsort(scores, descending=True)[:top_k]
    return [(THEMES[i], scores[i].item()) for i in top_indices]

def process_questions(questions: List[str]) -> None:
    """
    Process a list of questions and print the results.

    Args:
        questions (List[str]): List of questions to process.
    """
    total_time = 0
    for question in questions:
        start_time = time.time()
        relevant_themes = find_relevant_themes(question)
        end_time = time.time()

        print(f"Question: {question}")
        print("Relevant themes:")
        for theme, score in relevant_themes:
            print(f"  - {theme}: {score:.4f}")
        print(f"Time taken: {end_time - start_time:.4f} seconds\n")

        total_time += end_time - start_time

    avg_time = total_time / len(questions)
    print(f"Average time per question: {avg_time:.4f} seconds")
    print(f"Questions processed per second: {1/avg_time:.2f}")

def main():
    """Main function to run the script."""
    # Test questions.
    questions = [
        "How many children do not have access to clean water?",
        "What are the impacts of climate change on agriculture?",
        "How can we improve disaster preparedness in coastal areas?",
        "What are the challenges in providing education during conflicts?",
        "How does gender inequality affect humanitarian aid distribution?",
        "What are the main health concerns in refugee camps?",
        "How can we enhance food security in drought-prone regions?",
        "What are the best practices for mine clearance in post-conflict areas?",
        "How does climate change affect the spread of infectious diseases?",
        "What are the psychological impacts of displacement on children?",
        "How can we improve sanitation in overcrowded urban slums?",
        "What are the challenges in delivering aid to hard-to-reach areas?",
        "How can we promote sustainable agriculture in developing countries?",
        "What are the long-term effects of malnutrition on child development?",
        "How can we better protect human rights in humanitarian crises?",
        "What are the most effective strategies for disaster risk reduction?",
        "How can we improve access to healthcare in remote rural areas?",
        "What are the impacts of prolonged conflicts on education systems?",
        "How can we ensure the safety of humanitarian workers in conflict zones?",
        "What are the challenges in providing mental health support in emergencies?"
    ]

    process_questions(questions)

if __name__ == "__main__":
    main()
