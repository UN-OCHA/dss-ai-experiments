# ReliefWeb Content Ranker

This FastAPI application analyzes text input to determine relevant countries, themes, and disaster types based on ReliefWeb's taxonomy. It uses a pre-trained ColBERT model for ranking and provides detailed score analysis.

## Features

- Fetches and caches ReliefWeb taxonomy data (themes, countries, disaster types)
- Ranks input text against ReliefWeb taxonomies using a ColBERT model
- Provides detailed score analysis for each taxonomy category
- Determines pertinence of top-ranked items based on statistical thresholds
- FastAPI endpoint for easy integration and scalability

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn
- Pydantic
- Requests
- Statistics
- Rerankers (custom module for ColBERT model)

## Installation

1. Clone the repository:
2. Install the required packages:
   ```
   pip install fastapi uvicorn pydantic requests statistics rerankers
   ```

3. Ensure you have the `rerankers` module and the ColBERT model (`answerdotai/answerai-colbert-small-v1`) available.

## Usage

1. Start the FastAPI server:
   ```
   python main.py
   ```

2. The server will start on `http://0.0.0.0:8000`.

3. Use the `/analyze` endpoint to submit text for analysis:
   ```
   POST /analyze
   {
     "text": "Your input text here",
     "language": "en"
   }
   ```

4. The API will return an analysis including relevant country, theme, and disaster type (if pertinent), along with detailed score analyses.

## Configuration

- `RELIEFWEB_API_BASE_URL`: Base URL for the ReliefWeb API
- `APP_NAME`: Application name for ReliefWeb API requests
- `DATA_DIR`: Directory for caching ReliefWeb taxonomy data

## Notes

- The application caches ReliefWeb taxonomy data to reduce API calls.
- A ColBERT model is used for ranking, which is initialized on startup.
- The pertinence of top-ranked items is determined using statistical thresholds.
