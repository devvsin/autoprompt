import json
import asyncio
from groq import AsyncGroq
from src.utils import Review, ExtractedData
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

class BaselinePipeline:
    def __init__(self, config):
        self.client = AsyncGroq(api_key=config["api_key"], max_retries=3)
        self.model_name = config.get("generator_model", "llama-3.1-8b-instant")
        
        # We explicitly request JSON format matching the schema
        self.static_prompt = """Extract the product name and sentiment from this review.
You must respond in pure JSON.
The JSON must have EXACTLY these keys: "product" (string), "sentiment" (string: positive, negative, neutral, mixed), "reason" (string).
Review: '{text}'
"""
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def process(self, review: Review) -> ExtractedData:
        """Process a single review with a static prompt using asyncio and native JSON mode."""
        prompt = self.static_prompt.format(text=review.review_text)

        try:
            # We use native JSON schema mode
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. You only output valid JSON matching the exact schema requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            raw_text = response.choices[0].message.content
            data = json.loads(raw_text)
            
            return ExtractedData(
                review_id=review.review_id,
                product=data.get("product", "unknown"),
                sentiment=data.get("sentiment", "unknown"),
                reason=data.get("reason", ""),
                confidence=0.5,
                prompt_used="static",
            )
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise  # Tenacity handles exponential backoff
                
            if any(x in error_str for x in ["unavailable", "connection", "timeout", "tcp"]):
                raise
                
            logger.error(f"Baseline parse error for {review.review_id}: {e}")
            return ExtractedData(
                review_id=review.review_id,
                product="error",
                sentiment="error",
                reason=str(e)[:100],
                confidence=0.0,
                prompt_used="static_failed",
            )