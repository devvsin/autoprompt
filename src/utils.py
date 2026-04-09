import pandas as pd
from pydantic import BaseModel
from typing import Dict, Any
import os

class Review(BaseModel):
    review_id: str
    review_text: str

class ExtractedData(BaseModel):
    review_id: str
    product: str
    sentiment: str
    reason: str
    confidence: float = 0.0
    prompt_used: str = ""

def load_reviews(file_path: str) -> pd.DataFrame:
    """Load reviews from CSV or JSON and force review_id to string"""
    if file_path.endswith('.json'):
        df = pd.read_json(file_path, orient="records")
        if 'review_id' in df.columns:
            df['review_id'] = df['review_id'].astype(str)
        return df
    
    return pd.read_csv(file_path, dtype={'review_id': str})

def save_results(results: list, output_path: str):
    """Save results to JSON"""
    df = pd.DataFrame([r.dict() for r in results])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path, orient="records", indent=2)