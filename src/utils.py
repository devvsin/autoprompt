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

def load_reviews(csv_path: str) -> pd.DataFrame:
    """Load reviews from CSV and force review_id to string"""
    # ✅ FIXED: Specify dtype to prevent integer conversion
    return pd.read_csv(csv_path, dtype={'review_id': str})

def save_results(results: list, output_path: str):
    """Save results to JSON"""
    df = pd.DataFrame([r.dict() for r in results])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path, orient="records", indent=2)