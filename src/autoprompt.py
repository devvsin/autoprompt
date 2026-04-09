import json
import asyncio
from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import pandas as pd
from src.utils import Review, ExtractedData

class AutoPromptEngine:
    def __init__(self, config):
        self.client = AsyncGroq(api_key=config["api_key"], max_retries=3)
        self.config = config
        self.generator_model_name = config["generator_model"]
        self.scorer_model_name = config["scoring_model"]
        
        # State created during the Optimization Phase
        self.god_prompt = ""
        self.few_shot_examples = ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=False,
    )
    async def _safe_evolve_call(self, meta_prompt: str):
        response = await self.client.chat.completions.create(
            model=self.scorer_model_name,
            messages=[{"role": "user", "content": meta_prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    async def _evolve_prompts_via_meta_llm(self) -> list:
        """Use the scorer model as a Prompt Engineer to generate completely new prompt strategies."""
        meta_prompt = (
            "You are an expert AI Prompt Engineer. Your task is to write instructions "
            "for another LLM to extract data from e-commerce product reviews. "
            "Provide 4 distinct, completely completely different prompt strategies. Some can be strict, some can act like an analyst, etc. "
            "Output valid JSON ONLY with the schema: {\"prompts\": [\"template1\", \"template2\", \"template3\", \"template4\"]} "
            "Make sure your templates include the placeholder {text} where the review will be injected. "
            "Do NOT include literal JSON brackets inside your templates."
        )
        
        try:
            data = await self._safe_evolve_call(meta_prompt)
            raw_prompts = data.get("prompts", [])
            
            prompts = []
            for p in raw_prompts:
                if isinstance(p, dict):
                    prompts.append(str(list(p.values())[0]))
                else:
                    prompts.append(str(p))
                    
            if len(prompts) < 2:
                raise ValueError("Not enough prompts generated")
            return prompts[:4]
                
        except Exception as e:
            logger.warning(f"Meta-LLM hallucinated broken JSON during prompt generation. Using fallback defaults. Error: {e}")
            return [
                "Carefully read the text and extract the 'product', its 'sentiment' and the 'reason'. Review: {text}",
                "Be a meticulous analyst. Identify the main item purchased, the emotional tone (sentiment), and why. Here is the review: {text}",
                "Identify the product, sentiment, and reason from this review text: {text}",
                "Find what the user bought (product), how they feel (sentiment), and a short explanation (reason). Text: {text}"
            ]
        

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _test_prompt_against_review(self, prompt_template: str, review: Review) -> ExtractedData:
        """Inner async execution loop to evaluate a prompt against a text."""
        # Inject the text safely using replace instead of format to avoid JSON bracket KeyErrors
        if "{text}" not in prompt_template:
            prompt_template += "\nReview: {text}"
        
        prompt = prompt_template.replace("{text}", review.review_text)
        
        try:
            # Native JSON processing
            response = await self.client.chat.completions.create(
                model=self.generator_model_name,
                messages=[
                    {"role": "system", "content": "You are a data extractor. You must output pure JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Zero for accuracy test
                response_format={"type": "json_object"}
            )
            raw_text = response.choices[0].message.content
            data = json.loads(raw_text)
            
            return ExtractedData(
                review_id=review.review_id,
                product=data.get("product", "unknown"),
                sentiment=data.get("sentiment", "unknown"),
                reason=data.get("reason", ""),
                confidence=1.0,
                prompt_used=prompt_template
            )
        except Exception as e:
            return ExtractedData(
                review_id=review.review_id,
                product="error",
                sentiment="error",
                reason=str(e),
                confidence=0.0,
                prompt_used=prompt_template
            )

    async def optimize_prompt(self, train_reviews: list, ground_truth_df: pd.DataFrame):
        """
        Phase 1 Enterprise Optimization: 
        1. Meta-generate Prompts. 
        2. Test all permutations simultaneously. 
        3. Score mathematically against true Train data.
        4. Select & Lock highest scoring prompt.
        5. Build Few-Shot RAG injection.
        """
        logger.info("[AutoPrompt] Evolving 4 completely new prompt variants...")
        candidate_prompts = await self._evolve_prompts_via_meta_llm()
        
        best_prompt = candidate_prompts[0]
        best_score = -1
        training_successes = []

        # Free-tier semaphore limit
        sem = asyncio.Semaphore(15)

        async def _bounded_test(pt, rev):
            async with sem:
                await asyncio.sleep(2.0)
                return await self._test_prompt_against_review(pt, rev)

        for idx, prompt_temp in enumerate(candidate_prompts):
            # Test this prompt concurrently against the entire training batch safely
            tasks = [_bounded_test(prompt_temp, r) for r in train_reviews]
            results = await asyncio.gather(*tasks)
            
            # Mathematical objective scoring against true labels (Data leakage fix)
            score = 0
            for res in results:
                truth_row = ground_truth_df[ground_truth_df['review_id'] == res.review_id].iloc[0]
                pred_prod = res.product.lower().strip()
                true_prod = truth_row['product'].lower().strip()
                
                # Substring matching in case the LLM generates a slightly longer/shorter string
                if pred_prod in true_prod or true_prod in pred_prod:
                    score += 1
                if res.sentiment.lower().strip() == truth_row['sentiment'].lower().strip():
                    score += 1

                # Collect perfect extractions for Few-Shot RAG
                if score >= 2:
                    training_successes.append({
                        "review": [r for r in train_reviews if r.review_id == res.review_id][0].review_text,
                        "json": {"product": res.product, "sentiment": res.sentiment, "reason": res.reason}
                    })

            logger.info(f"  Variant {idx} Train Score: {score} / {len(train_reviews) * 2}")
            
            # Always accept the highest scoring prompt
            if score > best_score:
                best_score = score
                best_prompt = prompt_temp

        logger.info(f"[AutoPrompt] Optimization complete. God Prompt locked: {best_prompt[:60]}...")
        self.god_prompt = best_prompt
        
        # Build Few-Shot Injection text
        if training_successes:
            self.few_shot_examples = "EXAMPLES OF GOOD BEHAVIOR:\n"
            for ex in training_successes[:2]: # Max 2 examples
                self.few_shot_examples += f"--- Input: {ex['review']}\n--- Output: {json.dumps(ex['json'])}\n"


    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def process(self, review: Review) -> ExtractedData:
        """Phase 2: Ultra-fast async execution using the Locked God Prompt + Few-Shot RAG"""
        
        # The ultimate industry standard injected prompt
        final_system_prompt = (
            "You are a rigorous data extraction AI. You MUST output native JSON strictly using these EXACT keys: "
            '{"product": "string", "sentiment": "positive/negative/neutral/mixed", "reason": "string"}.\n'
            f"{self.few_shot_examples}"
        )
        
        # If somehow meta prompt didn't include {text}, append it
        if "{text}" in self.god_prompt:
            user_prompt = self.god_prompt.replace("{text}", review.review_text)
        else:
            user_prompt = self.god_prompt + f"\n\nReview: {review.review_text}"

        try:
            response = await self.client.chat.completions.create(
                model=self.generator_model_name,
                messages=[
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": user_prompt}
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
                confidence=1.0,
                prompt_used="auto_evolved_rag"
            )
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                raise  # Tenacity handles exponential backoff
            if any(x in error_str for x in ["unavailable", "connection", "timeout"]):
                raise
                
            return ExtractedData(
                review_id=review.review_id,
                product="error",
                sentiment="error",
                reason=str(e),
                confidence=0.0,
                prompt_used="auto_evolved_failed"
            )