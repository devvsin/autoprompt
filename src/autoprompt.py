import random
import google.generativeai as genai
from src.utils import Review, ExtractedData
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import re
import time

class AutoPromptEngine:
    def __init__(self, config: dict):
        # Secure API key from config
        genai.configure(api_key=config["api_key"])
        self.config = config
        self.generator_model = genai.GenerativeModel(config["generator_model"])
        self.scorer_model = genai.GenerativeModel(config["scoring_model"])
        self.use_llm_scoring = config.get("use_llm_scoring", False)
        
    def _generate_prompt_variants(self, review_text: str) -> list:
        """Generate prompt variants from candidate pools"""
        variants = []
        pool = self.config["candidates"]
        
        for _ in range(self.config["max_prompts_per_item"]):
            instruction = random.choice(pool["instruction"])
            target_info = random.choice(pool["target_info"])
            
            prompt = self.config["template"].format(
                instruction=instruction,
                target_info=target_info,
                text=review_text
            )
            # Enforce JSON output
            prompt += "\nRespond ONLY with JSON: {{\"product\": \"...\", \"sentiment\": \"...\", \"reason\": \"...\"}}"
            variants.append(prompt)
        
        return variants
    
    def _extract_json(self, text: str) -> dict:
        """Extract JSON from response, handling markdown code blocks"""
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
    
    @retry(
        stop=stop_after_attempt(2),  # Reduced from 3 to save API calls
        wait=wait_exponential(multiplier=2, min=3, max=15),
        reraise=True
    )
    def _call_llm(self, prompt: str) -> dict:
        """Generate content with retry logic and exponential backoff"""
        response = self.generator_model.generate_content(
            prompt,
            generation_config={"temperature": self.config["temperature"]}
        )
        return self._extract_json(response.text)
    
    def _score_prompt(self, review_text: str, response_data: dict) -> float:
        """Score prompt quality (0-1) using heuristics only"""
        score = 0.0
        
        # Heuristic checks (weighted more heavily now)
        required_fields = ["product", "sentiment", "reason"]
        if all(field in response_data for field in required_fields):
            score += 0.4  # Increased from 0.3
        
        valid_sentiments = ["positive", "negative", "neutral", "mixed"]
        if response_data.get("sentiment", "").lower() in valid_sentiments:
            score += 0.3  # Increased from 0.2
        
        product = response_data.get("product", "")
        if 2 < len(product) < 50:
            score += 0.2  # Same
        
        reason = response_data.get("reason", "")
        if len(reason) > 10:
            score += 0.1  # Same
        
        # Optional LLM-based semantic scoring (disabled by default for free tier)
        if self.use_llm_scoring:
            try:
                time.sleep(2)  # Longer delay for rate limits
                
                check_prompt = f"""
                Review: {review_text}
                Extracted Data: {json.dumps(response_data)}
                
                Does this extraction correctly identify:
                1. The main product/service? (yes/no)
                2. The sentiment? (yes/no)
                3. A specific reason? (yes/no)
                
                Respond with a single number 0-3.
                """
                
                check_response = self.scorer_model.generate_content(
                    check_prompt,
                    generation_config={"temperature": 0}
                )
                
                llm_score = int(check_response.text.strip()) / 3.0
                score = score * 0.8 + llm_score * 0.2  # Blend heuristic and LLM
                
            except Exception as e:
                logger.warning(f"Scoring LLM failed: {e}")
        
        return min(score, 1.0)
    
    def process(self, review: Review) -> ExtractedData:
        """Process review with dynamic prompt optimization"""
        logger.info(f"Processing review {review.review_id}")
        
        prompts = self._generate_prompt_variants(review.review_text)
        
        best_score = -1
        best_response = None
        best_prompt = ""
        
        for i, prompt in enumerate(prompts):
            try:
                # CRITICAL: Add delay between API calls to respect rate limits
                # Free tier: 10 req/min = 1 request every 6 seconds
                if i > 0:
                    logger.debug("Waiting 7 seconds to respect rate limits...")
                    time.sleep(7)
                
                response_data = self._call_llm(prompt)
                score = self._score_prompt(review.review_text, response_data)
                
                logger.info(f"Variant {i}: score={score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_response = response_data
                    best_prompt = prompt
                
                # Early stopping with lower threshold
                if score >= 0.85:  # Lowered from 0.9
                    logger.info(f"Early stopping - good score achieved")
                    break
                    
            except Exception as e:
                logger.error(f"Variant {i} failed: {e}")
                # If we've failed and have a result, use it
                if best_response is not None:
                    logger.warning(f"Using best result so far due to error")
                    break
                continue
        
        if best_response is None:
            return ExtractedData(
                review_id=review.review_id,
                product="error",
                sentiment="error",
                reason="All variants failed",
                confidence=0.0,
                prompt_used="autoprompt_failed"
            )
        
        return ExtractedData(
            review_id=review.review_id,
            product=best_response.get("product", "unknown"),
            sentiment=best_response.get("sentiment", "unknown"),
            reason=best_response.get("reason", ""),
            confidence=best_score,
            prompt_used=f"autoprompt_best_of_{len(prompts)}"
        )