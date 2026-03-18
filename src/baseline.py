import google.generativeai as genai
from src.utils import Review, ExtractedData
from loguru import logger
import json
import re
import time

class BaselinePipeline:
    def __init__(self, config: dict):
        # Use API key from secure config
        genai.configure(api_key=config["api_key"])
        self.model = genai.GenerativeModel(config.get("generator_model", "gemini-2.5-flash"))
        
        # FIXED: Double curly braces to escape them in format string
        self.static_prompt = """
Extract the product name and sentiment from this review. 
Respond ONLY with JSON format: {{"product": "...", "sentiment": "...", "reason": "..."}}
Review: '{text}'
"""
    
    def _extract_json(self, text: str) -> dict:
        """Extract JSON from response, handling markdown code blocks"""
        # Remove markdown code blocks if present
        text = text.strip()
        
        # Try to find JSON in markdown blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        # Remove any leading/trailing text before/after JSON
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        
        return json.loads(text)
    
    def process(self, review: Review) -> ExtractedData:
        """Process a single review with static prompt"""
        prompt = self.static_prompt.format(text=review.review_text)
        
        # Retry logic for network errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.1}
                )
                
                # Use the robust JSON extraction
                data = self._extract_json(response.text)
                
                return ExtractedData(
                    review_id=review.review_id,
                    product=data.get("product", "unknown"),
                    sentiment=data.get("sentiment", "unknown"),
                    reason=data.get("reason", ""),
                    confidence=0.5,
                    prompt_used="static"
                )
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a network error
                if any(x in error_str for x in ['unavailable', 'connection', 'timeout', 'tcp']):
                    if attempt < max_retries - 1:
                        wait_time = 10 * (attempt + 1)  # 10s, 20s, 30s
                        logger.warning(f"Network error for review {review.review_id}, retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                
                logger.error(f"Baseline failed for {review.review_id}: {e}")
                return ExtractedData(
                    review_id=review.review_id,
                    product="error",
                    sentiment="error",
                    reason=str(e)[:100],
                    confidence=0.0,
                    prompt_used="static_failed"
                )