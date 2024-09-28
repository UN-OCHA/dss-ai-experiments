"""
ReliefWeb Content Ranker.

This script provides a FastAPI application for analyzing text input
to determine relevant countries, themes, and disaster types based on
ReliefWeb's taxonomy using a pre-trained ColBERT model.
"""

import os
import json
import time
import statistics
from typing import List, Optional
from collections import OrderedDict
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rerankers import Reranker

# ReliefWeb API configuration.
RELIEFWEB_API_BASE_URL = "https://api.reliefweb.int/v1"
APP_NAME = "reliefweb-content-ranker"
DATA_DIR = './data'

# Ensure the data directory exists.
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_and_cache_data(endpoint: str, filename: str) -> List[str]:
    """Fetch data from ReliefWeb API and cache it locally."""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return json.load(file)

    url = f"{RELIEFWEB_API_BASE_URL}/{endpoint}"
    params = {"appname": APP_NAME, "limit": 1000}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = [item['fields']['name'] for item in response.json()['data']]
        with open(filepath, 'w') as file:
            json.dump(data, file)
        return data
    return []

def get_reliefweb_themes() -> List[str]:
    """Fetch ReliefWeb themes."""
    return fetch_and_cache_data("references/themes", "themes.json")

def get_reliefweb_countries() -> List[str]:
    """Fetch ReliefWeb countries."""
    return fetch_and_cache_data("countries", "countries.json")

def get_reliefweb_disaster_types() -> List[str]:
    """Fetch ReliefWeb disaster types."""
    return fetch_and_cache_data("references/disaster-types", "disaster_types.json")

def rank_items(query: str, items: List[str], ranker: Reranker) -> OrderedDict:
    """Rank items based on their relevance to the query."""
    results = ranker.rank(query=query, docs=items)
    return OrderedDict((result.text, result.score) for result in results.top_k(len(items)))

def analyze_scores(scores: OrderedDict) -> dict:
    """Analyze the distribution of scores."""
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    all_scores = list(scores.values())

    return {
        "top_3": sorted_scores[:3],
        "bottom_3": sorted_scores[-3:],
        "median_score": statistics.median(all_scores),
        "mean_score": statistics.mean(all_scores),
        "std_dev": statistics.stdev(all_scores),
        "max_score": max(all_scores),
        "min_score": min(all_scores),
        "score_range": max(all_scores) - min(all_scores),
    }

def is_term_pertinent(analysis: dict, category_name: str) -> bool:
    """Determine if the top-ranked term is pertinent based on score analysis."""
    score_range = analysis['score_range']
    top_mean_diff = analysis['top_3'][0][1] - analysis['mean_score']
    top_mean_ratio = analysis['top_3'][0][1] / analysis['mean_score']
    top_second_diff_percent = (analysis['top_3'][0][1] - analysis['top_3'][1][1]) / analysis['top_3'][0][1] * 100

    # Define thresholds.
    score_range_threshold = 0.01
    std_dev_threshold = 0.002
    top_mean_diff_threshold = 0.0075
    top_mean_ratio_threshold = 1.005
    top_second_diff_percent_threshold = 0.05

    # Print metrics and whether they meet the thresholds.
    print(f"\n{category_name} Analysis:")
    print(f"Score Range: {score_range} (Above threshold: {score_range > score_range_threshold})")
    print(f"Standard Deviation: {analysis['std_dev']} (Above threshold: {analysis['std_dev'] > std_dev_threshold})")
    print(f"Top Score - Mean Score Difference: {top_mean_diff} (Above threshold: {top_mean_diff > top_mean_diff_threshold})")
    print(f"Top Score / Mean Score Ratio: {top_mean_ratio} (Above threshold: {top_mean_ratio > top_mean_ratio_threshold})")
    print(f"Top Score - Second Top Score Percentage Difference: {top_second_diff_percent} (Above threshold: {top_second_diff_percent > top_second_diff_percent_threshold})")

    return (
        score_range > score_range_threshold and
        analysis['std_dev'] > std_dev_threshold and
        top_mean_diff > top_mean_diff_threshold and
        top_mean_ratio > top_mean_ratio_threshold and
        top_second_diff_percent > top_second_diff_percent_threshold
    )

themes, countries, disaster_types, ranker = [], [], [], None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    global themes, countries, disaster_types, ranker
    print("Fetching ReliefWeb data...")
    themes = get_reliefweb_themes()
    countries = get_reliefweb_countries()
    disaster_types = get_reliefweb_disaster_types()
    print(f"Retrieved {len(themes)} themes, {len(countries)} countries, and {len(disaster_types)} disaster types.")

    print("Initializing the ranker...")
    ranker = Reranker("answerdotai/answerai-colbert-small-v1", model_type='colbert')

    print("Warming up the model...")
    warmup_query = "This is a warm-up query to initialize the model."
    _ = rank_items(warmup_query, themes, ranker)
    print("Model warm-up complete.")

    yield

    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

class AnalyzeRequest(BaseModel):
    """Request model for text analysis."""
    text: str
    language: str

class ScoreAnalysis(BaseModel):
    """Model for score analysis results."""
    top_3: List[tuple]
    bottom_3: List[tuple]
    median_score: float
    mean_score: float
    std_dev: float
    max_score: float
    min_score: float
    score_range: float

class AnalyzeResponse(BaseModel):
    """Response model for text analysis."""
    country: Optional[str] = None
    theme: Optional[str] = None
    disaster_type: Optional[str] = None
    processing_time: float
    country_analysis: ScoreAnalysis
    theme_analysis: ScoreAnalysis
    disaster_type_analysis: ScoreAnalysis

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(request: AnalyzeRequest):
    """Endpoint for analyzing text input."""
    if not themes or not countries or not disaster_types or not ranker:
        raise HTTPException(status_code=503, detail="Service not ready. Please try again later.")

    start_time = time.time()

    ranked_countries = rank_items(request.text, countries, ranker)
    ranked_themes = rank_items(request.text, themes, ranker)
    ranked_disaster_types = rank_items(request.text, disaster_types, ranker)

    country_analysis = analyze_scores(ranked_countries)
    theme_analysis = analyze_scores(ranked_themes)
    disaster_type_analysis = analyze_scores(ranked_disaster_types)

    top_country = country_analysis['top_3'][0]
    top_theme = theme_analysis['top_3'][0]
    top_disaster_type = disaster_type_analysis['top_3'][0]

    country = top_country[0] if is_term_pertinent(country_analysis, "Country") else None
    theme = top_theme[0] if is_term_pertinent(theme_analysis, "Theme") else None
    disaster_type = top_disaster_type[0] if is_term_pertinent(disaster_type_analysis, "Disaster Type") else None

    processing_time = time.time() - start_time

    print(f"Analysis processed in {processing_time:.2f} seconds")
    print(f"Language: {request.language}")

    return AnalyzeResponse(
        country=country,
        theme=theme,
        disaster_type=disaster_type,
        processing_time=processing_time,
        country_analysis=ScoreAnalysis(**country_analysis),
        theme_analysis=ScoreAnalysis(**theme_analysis),
        disaster_type_analysis=ScoreAnalysis(**disaster_type_analysis)
    )

def main():
    """Main function to run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
